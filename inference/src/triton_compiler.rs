//! Frozen default SM89 compiler execution and original local disk-cache chain.
//! Typed frontend IR is generated on each cold specialization; no saved shape
//! list, reference IR or reference cubin is an execution input.
use crate::{
    native::{Error, Kitchen},
    triton_driver::{Driver, Kernel, Scalar},
    triton_file_cache::{CacheText, FileCache, JsonPath, Mode, os_error, uuid4},
    triton_frontend::{Fp8Body, Fp8Specialization, fp8_source_ttir},
    triton_specialization::{
        CompilerHashInputs, ast_key, cache_directory, compiler_digest, mxfp8_ast_key, rope_ast_key,
        rowwise_ast_key,
    },
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::Command,
    sync::{Arc, Mutex, RwLock},
};
fn error(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
fn field<'a>(v: &'a Value, name: &str) -> Result<&'a str, Error> {
    v[name]
        .as_str()
        .ok_or_else(|| Error::Native(format!("Missing verified compiler profile field {name}")))
}
fn read(path: &Path) -> Result<Vec<u8>, Error> {
    fs::read(path).map_err(|e| os_error(e, path, None))
}
pub(crate) fn utf8(bytes: &[u8]) -> Result<String, Error> {
    String::from_utf8(bytes.to_vec()).map_err(|e| {
        let f = e.utf8_error();
        let start = f.valid_up_to();
        let n = f.error_len().unwrap_or(bytes.len() - start);
        let first = bytes[start];
        let reason = if f.error_len().is_none() {
            "unexpected end of data"
        } else if (0xc2..=0xf4).contains(&first) {
            "invalid continuation byte"
        } else {
            "invalid start byte"
        };
        let at = if n == 1 {
            format!("byte 0x{first:02x} in position {start}")
        } else {
            format!("bytes in position {start}-{}", start + n - 1)
        };
        error(
            "UnicodeDecodeError",
            format!("'utf-8' codec can't decode {at}: {reason}"),
        )
    })
}
fn verify(path: &Path, expected: &str) -> Result<(), Error> {
    let mut input = fs::File::open(path).map_err(|e| os_error(e, path, None))?;
    let mut sha = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = input.read(&mut buf).map_err(|e| os_error(e, path, None))?;
        if n == 0 {
            break;
        }
        sha.update(&buf[..n]);
    }
    let actual = sha
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    if actual != expected {
        return Err(Error::Native(format!(
            "Verified compiler input changed: {}",
            path.display()
        )));
    }
    Ok(())
}
pub type Fp8PreRunGuard = Arc<
    dyn Fn(
            &str,
            &[crate::execution_value::Value],
            &crate::execution_value::Kwargs,
        ) -> Result<(), Error>
        + Send
        + Sync,
>;
pub struct Compiler {
    profile: Value,
    pre_run_guard: RwLock<Option<Fp8PreRunGuard>>,
    pub(crate) int8_tuners: [Mutex<crate::autotuning::Int8RuntimeState>; 2],
    native: Kitchen,
    driver: Driver,
    cache: PathBuf,
    workspace: PathBuf,
    kernels: Mutex<BTreeMap<(usize, String), Arc<CompiledKernel>>>,
    pub hooks: Arc<crate::triton_hooks::HookState>,
    modes: RwLock<crate::triton_compilation_modes::Modes>,
}
pub struct CompiledKernel {
    pub digest: String,
    pub ast_digest: String,
    pub metadata: Value,
    pub metadata_group: Vec<(CacheText, JsonPath)>,
    pub assembly: Vec<(String, Vec<u8>)>,
    pub disk_cache_hit: bool,
    driver: Driver,
    native: Kitchen,
    device: usize,
    state: Mutex<HandleState>,
    hooks: Arc<crate::triton_hooks::HookState>,
}
enum HandleState {
    Uninitialized,
    LauncherWithoutFunction,
    Ready(Arc<Kernel>),
    Sticky {
        class: &'static str,
        message: String,
        loaded: Option<Arc<Kernel>>,
    },
}
impl CompiledKernel {
    pub fn resources(&self) -> Option<[i32; 3]> {
        let state = self.state.lock().unwrap();
        let kernel = match &*state {
            HandleState::Ready(kernel) => Some(kernel),
            HandleState::Sticky { loaded, .. } => loaded.as_ref(),
            _ => None,
        }?;
        Some([kernel.registers, kernel.spills, kernel.max_threads])
    }
    pub fn driver_handle_presence(&self) -> (bool, bool) {
        let state = self.state.lock().unwrap();
        let present = matches!(
            &*state,
            HandleState::Ready(_)
                | HandleState::Sticky {
                    loaded: Some(_),
                    ..
                }
        );
        (present, present)
    }
    unsafe fn initialize(&self, metadata_initialization: bool) -> Result<Arc<Kernel>, Error> {
        {
            let mut state = self.state.lock().unwrap();
            match &*state {
                HandleState::Ready(kernel) => return Ok(kernel.clone()),
                HandleState::Sticky {
                    loaded: Some(kernel),
                    ..
                } if metadata_initialization => return Ok(kernel.clone()),
                HandleState::Sticky { loaded: None, .. } if metadata_initialization => {}
                HandleState::Sticky { class, message, .. } => {
                    return Err(error(class, message.clone()));
                }
                HandleState::LauncherWithoutFunction if !metadata_initialization => {
                    return Err(error("TypeError", "argument 5 must be int, not None"));
                }
                HandleState::Uninitialized | HandleState::LauncherWithoutFunction => {}
            }
            // Source publishes its launcher before driver load, including a
            // failed load. Release the host lock during native initialization.
            *state = HandleState::LauncherWithoutFunction;
        }
        let attribute = |name: &str| {
            self.metadata.get(name).ok_or_else(|| {
                error(
                    "AttributeError",
                    format!("'KernelMetadata' object has no attribute '{name}'"),
                )
            })
        };
        let shared = attribute("shared")?
            .as_i64()
            .ok_or_else(|| error("TypeError", "Expected integer shared-memory metadata"))?;
        let warps = attribute("num_warps")?
            .as_i64()
            .ok_or_else(|| error("TypeError", "Expected integer warp metadata"))?;
        let fail = |message: String, loaded: Option<Arc<Kernel>>| {
            *self.state.lock().unwrap() = HandleState::Sticky {
                class: "OutOfResources",
                message: message.clone(),
                loaded,
            };
            error("OutOfResources", message)
        };
        let maximum = self
            .native
            .device_properties(self.device)?
            .shared_memory_per_block_optin;
        if shared > maximum {
            return Err(fail(
                format!(
                    "out of resource: shared memory, Required: {shared}, Hardware limit: {maximum}. Reducing block sizes or `num_stages` may help."
                ),
                None,
            ));
        }
        if let Some(tmem) = self.metadata["tmem_size"].as_i64()
            && tmem > 512
        {
            return Err(fail(
                format!(
                    "out of resource: tensor memory, Required: {tmem}, Hardware limit: 512. Reducing block sizes or `num_stages` may help."
                ),
                None,
            ));
        }
        let name = attribute("name")?
            .as_str()
            .ok_or_else(|| error("TypeError", "Expected string kernel name"))?;
        let cubin = &self.assembly.iter().find(|(k, _)| k == "cubin").unwrap().1;
        if let Some(hook) = self.hooks.snapshot().load_start {
            hook(&crate::triton_hooks::LoadMetadata {
                module: None,
                function: None,
                name,
                metadata_group: &self.metadata_group,
                digest: &self.digest,
            })?;
        }
        let kernel = Arc::new(unsafe {
            self.driver
                .load(name, cubin, shared as i32, self.device as i32)?
        });
        *self.state.lock().unwrap() = HandleState::Ready(kernel.clone());
        if warps * 32 > i64::from(kernel.max_threads) {
            return Err(fail(
                format!(
                    "out of resource: threads, Required: {}, Hardware limit: {}. Reducing block sizes or `num_stages` may help.",
                    warps * 32,
                    kernel.max_threads
                ),
                Some(kernel),
            ));
        }
        if let Some(hook) = self.hooks.snapshot().load_end {
            hook(&crate::triton_hooks::LoadMetadata {
                module: Some(kernel.module.as_ptr() as u64),
                function: Some(kernel.function.as_ptr() as u64),
                name,
                metadata_group: &self.metadata_group,
                digest: &self.digest,
            })?;
        }
        Ok(kernel)
    }
    /// # Safety
    /// Arguments, current CUDA context, capacities and stream lifetimes must
    /// match this original specialization. No implicit stream recording occurs.
    pub unsafe fn run(
        &self,
        grid: [i32; 3],
        stream: u64,
        args: &mut [Scalar],
    ) -> Result<(), Error> {
        // launch_metadata initializes handles only when an enter hook exists.
        let launch_metadata = if self.hooks.snapshot().launch_enter.is_some() {
            let kernel = unsafe { self.initialize(true)? };
            Some(crate::triton_hooks::LaunchMetadata {
                name: self.metadata["name"].as_str().unwrap().into(),
                function: kernel.function.as_ptr() as u64,
                stream,
            })
        } else {
            None
        };
        let kernel = unsafe { self.initialize(false)? };
        let integer = |name: &str| {
            self.metadata[name].as_i64().ok_or_else(|| {
                error(
                    "TypeError",
                    format!("Expected integer kernel metadata {name}"),
                )
            })
        };
        let warps = integer("num_warps")? as i32;
        let ctas = integer("num_ctas")? as i32;
        let shared = integer("shared")? as i32;
        let hooks = self.hooks.snapshot();
        if let Some(hook) = hooks.launch_enter {
            hook(launch_metadata.as_ref())?;
        }
        // The caller's original object arguments remain borrowed through the
        // exit hook. Resolve into a separate buffer so callbacks can change
        // storage without either stale pointers or early TensorImpl release.
        let mut extracted = args
            .iter()
            .any(|value| matches!(value, Scalar::TensorObject(_)))
            .then(|| args.to_vec());
        let args = extracted.as_deref_mut().unwrap_or(args);
        for arg in args.iter_mut() {
            if let Scalar::TensorObject(value) = arg {
                let pointer = crate::backend_triton_fp8::info(value, "data_ptr")?.2;
                *arg = Scalar::TensorPointer(pointer);
            }
        }
        // The original native launcher enters the hook before extracting
        // tensor pointers; a pointer failure suppresses only the exit hook.
        for arg in args.iter_mut() {
            if let Scalar::TensorPointer(pointer) = arg {
                *pointer = unsafe { self.driver.tensor_pointer(*pointer)? };
            }
        }
        unsafe {
            self.driver.launch(
                &kernel,
                grid,
                warps,
                ctas,
                self.metadata["launch_cooperative_grid"] == true,
                self.metadata["launch_pdl"] == true,
                shared,
                stream,
                args,
            )?;
        }
        if let Some(hook) = hooks.launch_exit {
            hook(launch_metadata.as_ref())?;
        }
        Ok(())
    }
}
struct WorkingDirectory(PathBuf, bool);
impl Drop for WorkingDirectory {
    fn drop(&mut self) {
        if self.1 {
            if let Ok(entries) = fs::read_dir(&self.0) {
                for entry in entries.flatten() {
                    if entry.file_name() != "ptx" {
                        let _ = fs::remove_file(entry.path());
                    }
                }
            }
        } else {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
}
impl Compiler {
    /// # Safety
    /// The profile and referenced ELF files must be the locally verified frozen
    /// compiler build. The current CUDA context must follow the source contract.
    pub unsafe fn open(
        profile_path: &Path,
        cache: &Path,
        workspace: &Path,
        native: Kitchen,
    ) -> Result<Self, Error> {
        let profile: Value = serde_json::from_slice(&read(profile_path)?)
            .map_err(|e| Error::Native(e.to_string()))?;
        if profile["schema"] != 1
            || profile["target"]["backend"] != "cuda"
            || profile["target"]["arch"] != 89
        {
            return Err(Error::Native(
                "Expected frozen default SM89 compiler profile".into(),
            ));
        }
        for tool in profile["tools"]
            .as_object()
            .ok_or_else(|| Error::Native("Missing verified compiler tools".into()))?
            .values()
        {
            verify(Path::new(field(tool, "path")?), field(tool, "sha256")?)?;
        }
        for library in profile["external_libraries"]
            .as_array()
            .ok_or_else(|| Error::Native("Missing verified external compiler libraries".into()))?
        {
            verify(
                Path::new(field(library, "path")?),
                field(library, "sha256")?,
            )?;
        }
        let source = &profile["source_file"];
        verify(Path::new(field(source, "copy")?), field(source, "sha256")?)?;
        if let Some(sources) = profile["extra_source_files"].as_array() {
            for source in sources {
                verify(Path::new(field(source, "copy")?), field(source, "sha256")?)?;
            }
        }
        let driver =
            unsafe { Driver::open(Path::new(field(&profile["tools"]["cuda_driver"], "path")?))? };
        Ok(Self {
            profile,
            pre_run_guard: RwLock::new(None),
            int8_tuners: std::array::from_fn(|_| {
                Mutex::new(crate::autotuning::Int8RuntimeState::default())
            }),
            native,
            driver,
            cache: cache.into(),
            workspace: workspace.into(),
            kernels: Mutex::new(BTreeMap::new()),
            hooks: Arc::new(crate::triton_hooks::HookState::default()),
            modes: RwLock::new(crate::triton_compilation_modes::Modes::default()),
        })
    }
    /// Development metadata guard at the original pre-run-hook boundary.
    /// The normal path is None; it must not inspect unowned/unwritten payloads.
    pub fn set_pre_run_guard(&self, guard: Option<Fp8PreRunGuard>) {
        *self.pre_run_guard.write().unwrap() = guard;
    }
    pub(crate) fn pre_run(
        &self,
        op: &str,
        args: &[crate::execution_value::Value],
        kwargs: &crate::execution_value::Kwargs,
    ) -> Result<(), Error> {
        let guard = self.pre_run_guard.read().unwrap().clone();
        if let Some(guard) = guard {
            // JITFunction.run inserts these options before invoking hooks. The
            // verified frozen wrappers do not supply either keyword themselves.
            let mut kwargs = kwargs.clone();
            if !kwargs.iter().any(|(name, _)| name == "debug") {
                kwargs.push((
                    "debug".into(),
                    crate::execution_value::Value::Bool(
                        self.profile["options"]["debug"].as_bool().unwrap(),
                    ),
                ));
            }
            let instrumentation = crate::execution_value::Value::Text(
                field(&self.profile["options"], "instrumentation_mode")?.into(),
            );
            if let Some((_, value)) = kwargs
                .iter_mut()
                .find(|(name, _)| name == "instrumentation_mode")
            {
                *value = instrumentation;
            } else {
                kwargs.push(("instrumentation_mode".into(), instrumentation));
            }
            guard(op, args, &kwargs)?;
        }
        Ok(())
    }
    /// Metadata-only snapshot of the corresponding original INT8 autotuner.
    pub fn int8_autotuning_state(&self, per_row: bool) -> crate::autotuning::Int8RuntimeSnapshot {
        self.int8_tuners[usize::from(per_row)]
            .lock()
            .unwrap()
            .snapshot()
    }
    pub fn driver(&self) -> &Driver {
        &self.driver
    }
    pub fn source_path(&self) -> &str {
        self.profile["source_file"]["path"].as_str().unwrap()
    }
    fn tool(&self, name: &str) -> &str {
        self.profile["tools"][name]["path"].as_str().unwrap()
    }
    fn command(&self, name: &str, args: &[&str]) -> Result<(), Error> {
        let output = Command::new(self.tool(name))
            .args(args)
            .output()
            .map_err(|e| os_error(e, Path::new(self.tool(name)), None))?;
        if output.status.success() {
            return Ok(());
        }
        let text = utf8(&output.stderr)?;
        if name == "ptxas" {
            use std::os::unix::process::ExitStatusExt;
            let code = output
                .status
                .code()
                .unwrap_or_else(|| -output.status.signal().unwrap_or(0));
            let reason = match code {
                255 => "Internal Triton PTX codegen error".into(),
                139 => "`ptxas` raised SIGSEGV".into(),
                n => format!("`ptxas` failed with error code {n}"),
            };
            return Err(error(
                "PTXASError",
                format!(
                    "PTXAS error: {reason}\n`ptxas` stderr:\n{text}\nRepro command: {} {}\n",
                    self.tool(name),
                    args.join(" ")
                ),
            ));
        }
        if name == "llvm_driver" && output.status.code() == Some(3) {
            return Err(error(
                "ValueError",
                text.strip_suffix('\n').unwrap_or(&text),
            ));
        }
        Err(error("RuntimeError", text))
    }
    fn optimize(
        &self,
        input: &Path,
        output: &Path,
        pipeline: Option<&str>,
    ) -> Result<PathBuf, Error> {
        // Preserve the original in-memory graph, including locations omitted by
        // scf.for's custom text printer. Pretty original cache text is separate.
        let internal = output.with_file_name(format!(
            "{}.mlirbc",
            output.file_name().unwrap().to_str().unwrap()
        ));
        let mut args = vec![input.to_str().unwrap().to_string()];
        if let Some(pipeline) = pipeline {
            args.push(format!("--pass-pipeline={pipeline}"));
        }
        args.extend([
            "--emit-bytecode".into(),
            "-o".into(),
            internal.to_str().unwrap().into(),
        ]);
        self.command(
            "triton_opt",
            &args.iter().map(String::as_str).collect::<Vec<_>>(),
        )?;
        self.command(
            "triton_opt",
            &[
                internal.to_str().unwrap(),
                "--mlir-print-debuginfo",
                "--mlir-use-nameloc-as-prefix",
                "-o",
                output.to_str().unwrap(),
            ],
        )?;
        // Reproduce native module.__str__, rather than the CLI's extra newline.
        let data = read(output)?;
        let end = data.iter().rposition(|b| *b != b'\n').map_or(0, |i| i + 1);
        let mut data = data[..end].to_vec();
        data.push(b'\n');
        fs::write(output, data).map_err(|e| os_error(e, output, None))?;
        Ok(internal)
    }
    /// # Safety
    /// Device/context and later pointer/stream lifetimes must match the original
    /// typed launch. This implements the frozen default compiler configuration.
    unsafe fn fp8_source(
        &self,
        device: usize,
        s: &Fp8Specialization<'_>,
        jit: bool,
    ) -> Result<Arc<CompiledKernel>, Error> {
        let s = Fp8Specialization {
            source_path: self.source_path(),
            ..*s
        };
        let s = &s;
        let op = if matches!(s.body, Fp8Body::Quantize) {
            "quantize"
        } else {
            "dequantize"
        };
        let function = &self.profile["functions"][op];
        let ast = ast_key(s, field(function, "function_cache_key")?);
        let ast = self.source_digest(&ast, function, jit)?;
        unsafe { self.specialization(device, op, &ast, || fp8_source_ttir(s)) }
    }
    /// # Safety
    /// Device, operand capacities and launch ownership must follow the frozen
    /// original rowwise kernel. The source wrapper supplies its block size.
    unsafe fn rowwise_source(
        &self,
        device: usize,
        s: &crate::triton_frontend::rowwise::Specialization<'_>,
        jit: bool,
    ) -> Result<Arc<CompiledKernel>, Error> {
        let function = &self.profile["functions"]["rowwise"];
        let s = crate::triton_frontend::rowwise::Specialization {
            source_path: self.source_path(),
            standard_source_path: field(function, "standard_source_path")?,
            ..*s
        };
        let ast = rowwise_ast_key(&s, field(function, "function_cache_key")?);
        let ast = self.source_digest(&ast, function, jit)?;
        unsafe {
            self.specialization(device, "rowwise", &ast, || {
                crate::triton_frontend::rowwise::source_ttir(&s)
            })
        }
    }
    /// # Safety
    /// Operand capacities, launch shape and stream ownership must satisfy the
    /// original AdaLN wrapper and its original two-pass kernel.
    unsafe fn adaln_source(
        &self,
        device: usize,
        s: &crate::triton_frontend::adaln::Specialization<'_>,
        jit: bool,
        epsilon: crate::triton_frontend::scalar::ScalarType,
    ) -> Result<Arc<CompiledKernel>, Error> {
        let function = &self.profile["functions"]["adaln"];
        let s = crate::triton_frontend::adaln::Specialization {
            source_path: field(function, "source_path")?,
            standard_source_path: field(function, "standard_source_path")?,
            ..*s
        };
        let ast = crate::triton_specialization::adaln_ast_key_scalar(
            &s,
            field(function, "function_cache_key")?,
            epsilon,
        );
        let ast = self.source_digest(&ast, function, jit)?;
        unsafe {
            self.specialization(device, "adaln", &ast, || {
                crate::triton_frontend::adaln::source_ttir_scalar(&s, epsilon)
            })
        }
    }
    /// # Safety
    /// Operand capacities, grid and stream ownership must follow the original
    /// flexible-layout RoPE wrapper.
    unsafe fn rope_source(
        &self,
        device: usize,
        s: &crate::triton_frontend::rope::Specialization<'_>,
        jit: bool,
    ) -> Result<Arc<CompiledKernel>, Error> {
        let function = &self.profile["functions"]["rope"];
        let s = crate::triton_frontend::rope::Specialization {
            source_path: field(function, "source_path")?,
            ..*s
        };
        let ast = rope_ast_key(&s, field(function, "function_cache_key")?);
        let ast = self.source_digest(&ast, function, jit)?;
        unsafe {
            self.specialization(device, "rope", &ast, || {
                crate::triton_frontend::rope::source_ttir(&s)
            })
        }
    }
    /// # Safety
    /// Operand capacities and stream ownership must follow the frozen RMS-RoPE
    /// wrapper, including its normalization materialization boundary.
    unsafe fn rms_rope_source(
        &self,
        device: usize,
        s: &crate::triton_frontend::rope::rms::Specialization<'_>,
        jit: bool,
        epsilon: crate::triton_frontend::scalar::ScalarType,
    ) -> Result<Arc<CompiledKernel>, Error> {
        let function = &self.profile["functions"]["rms_rope"];
        let s = crate::triton_frontend::rope::rms::Specialization {
            source_path: field(function, "source_path")?,
            standard_source_path: field(function, "standard_source_path")?,
            ..*s
        };
        let ast = crate::triton_specialization::rms_rope_ast_key_scalar(
            &s,
            field(function, "function_cache_key")?,
            epsilon,
        );
        let ast = self.source_digest(&ast, function, jit)?;
        unsafe {
            self.specialization(device, "rms_rope", &ast, || {
                crate::triton_frontend::rope::rms::source_ttir_scalar(&s, epsilon)
            })
        }
    }
    /// # Safety
    /// Operand capacities, grid and stream ownership must follow the original
    /// MXFP8 wrapper, including its incomplete-block no-write behavior.
    unsafe fn mxfp8_source(
        &self,
        device: usize,
        s: &crate::triton_frontend::mxfp8::Specialization<'_>,
        jit: bool,
    ) -> Result<Arc<CompiledKernel>, Error> {
        let function = &self.profile["functions"]["mxfp8"];
        let s = crate::triton_frontend::mxfp8::Specialization {
            source_path: self.source_path(),
            standard_source_path: field(function, "standard_source_path")?,
            ..*s
        };
        let ast = mxfp8_ast_key(&s, field(function, "function_cache_key")?);
        let ast = self.source_digest(&ast, function, jit)?;
        unsafe {
            self.specialization(device, "mxfp8", &ast, || {
                crate::triton_frontend::mxfp8::source_ttir(&s)
            })
        }
    }
    /// # Safety
    /// Operand capacities and launch ownership follow the original wrapper.
    /// SM89 assembly errors propagate from the actual original ptxas binary.
    unsafe fn nvfp4_source(
        &self,
        device: usize,
        s: &crate::triton_frontend::nvfp4::Specialization<'_>,
        jit: bool,
        literal: Option<(&crate::triton_arguments::ConstexprKey, Option<&str>)>,
    ) -> Result<Arc<CompiledKernel>, Error> {
        let function = &self.profile["functions"]["nvfp4"];
        let s = crate::triton_frontend::nvfp4::Specialization {
            source_path: self.source_path(),
            standard_source_path: field(function, "standard_source_path")?,
            ..*s
        };
        let ast =
            crate::triton_specialization::nvfp4_ast_key(&s, field(function, "function_cache_key")?);
        let (ast, memory) =
            self.conditional_digests(&ast, function, jit, literal.map(|(k, _)| k))?;
        unsafe {
            self.specialization_with_memory_key(
                device,
                "nvfp4",
                &ast,
                &memory,
                &self.profile,
                || {
                    if let Some(failure) = crate::triton_frontend::nvfp4::condition_failure(
                        false,
                        literal.and_then(|(_, kind)| kind),
                    ) {
                        return Err(Error::ReferenceException {
                            class: failure.class,
                            message: failure.message,
                        });
                    }
                    Ok(crate::triton_frontend::nvfp4::source_ttir(&s))
                },
            )
        }
    }
    /// # Safety
    /// Packed inputs, scale capacities and stream ownership follow the original.
    unsafe fn dequant_nvfp4_source(
        &self,
        device: usize,
        s: &crate::triton_frontend::nvfp4::dequantize::Specialization<'_>,
        jit: bool,
        literal: Option<(&crate::triton_arguments::ConstexprKey, Option<&str>)>,
    ) -> Result<Arc<CompiledKernel>, Error> {
        let function = &self.profile["functions"]["dequant_nvfp4"];
        let s = crate::triton_frontend::nvfp4::dequantize::Specialization {
            source_path: self.source_path(),
            ..*s
        };
        let ast = crate::triton_specialization::dequant_nvfp4_ast_key(
            &s,
            field(function, "function_cache_key")?,
        );
        let (ast, memory) =
            self.conditional_digests(&ast, function, jit, literal.map(|(k, _)| k))?;
        unsafe {
            self.specialization_with_memory_key(
                device,
                "dequant_nvfp4",
                &ast,
                &memory,
                &self.profile,
                || {
                    if let Some(failure) = crate::triton_frontend::nvfp4::condition_failure(
                        true,
                        literal.and_then(|(_, kind)| kind),
                    ) {
                        return Err(Error::ReferenceException {
                            class: failure.class,
                            message: failure.message,
                        });
                    }
                    Ok(crate::triton_frontend::nvfp4::dequantize::source_ttir(&s))
                },
            )
        }
    }

    /// Compile one unchanged candidate of either original INT8 autotuner.
    /// # Safety
    /// The original candidate index, specialization, operand capacities and
    /// device/stream ownership must correspond to the frozen typed launch.
    unsafe fn int8_source(
        &self,
        device: usize,
        s: &crate::triton_frontend::int8::Specialization<'_>,
        configuration_index: usize,
        jit: bool,
    ) -> Result<Arc<CompiledKernel>, Error> {
        let op = if s.per_row {
            "int8_per_row"
        } else {
            "int8_tensorwise"
        };
        let function = &self.profile["functions"][op];
        let configuration = &function["configurations"][configuration_index];
        let s = crate::triton_frontend::int8::Specialization {
            source_path: self.source_path(),
            standard_source_path: field(function, "standard_source_path")?,
            ..*s
        };
        let ast =
            crate::triton_specialization::int8_ast_key(&s, field(function, "function_cache_key")?);
        let ast = self.source_digest(&ast, function, jit)?;
        unsafe {
            self.specialization_with_configuration(device, op, &ast, configuration, || {
                crate::triton_frontend::int8::source_ttir(&s)
            })
        }
    }

    /// Compile the frozen INT4/INT6 grouped unpack kernel.
    /// # Safety
    /// Operand capacities, specialization and device/stream must match the source launch.
    unsafe fn w4a8_unpack_source(
        &self,
        device: usize,
        s: &crate::triton_frontend::w4a8::Specialization<'_>,
        jit: bool,
    ) -> Result<Arc<CompiledKernel>, Error> {
        let f = &self.profile["functions"]["w4a8_unpack"];
        let s = crate::triton_frontend::w4a8::Specialization {
            source_path: field(f, "source_path")?,
            ..*s
        };
        let ast = crate::triton_specialization::w4a8_ast_key(&s, field(f, "function_cache_key")?);
        let ast = self.source_digest(&ast, f, jit)?;
        unsafe {
            self.specialization(device, "w4a8_unpack", &ast, || {
                crate::triton_frontend::w4a8::source_ttir(&s)
            })
        }
    }

    /// Compile the frozen online-softmax neighborhood attention kernel.
    /// # Safety
    /// Original operand capacities, specializations and stream/device must match.
    unsafe fn na3d_source(
        &self,
        device: usize,
        s: &crate::triton_frontend::na::Specialization<'_>,
        jit: bool,
    ) -> Result<Arc<CompiledKernel>, Error> {
        let f = &self.profile["functions"]["na3d"];
        let s = crate::triton_frontend::na::Specialization {
            source_path: field(f, "source_path")?,
            standard_source_path: field(f, "standard_source_path")?,
            ..*s
        };
        let ast = crate::triton_specialization::na_ast_key(&s, field(f, "function_cache_key")?);
        let ast = self.source_digest(&ast, &self.profile["functions"]["na3d"], jit)?;
        unsafe {
            self.specialization_with_configuration_result(
                device,
                "na3d",
                &ast,
                &self.profile,
                || {
                    crate::triton_frontend::na::checked_source(&s).map_err(|e| {
                        Error::ReferenceException {
                            class: e.class,
                            message: e.message,
                        }
                    })
                },
            )
        }
    }
    /// # Safety
    /// Device/context and later pointer/stream lifetimes must match the original
    /// typed launch. This implements the frozen default compiler configuration.
    pub unsafe fn fp8(
        &self,
        device: usize,
        s: &Fp8Specialization<'_>,
    ) -> Result<Arc<CompiledKernel>, Error> {
        unsafe { self.fp8_source(device, s, false) }
    }
    /// # Safety
    /// Device/context and later pointer/stream lifetimes must match the original
    /// typed launch. This implements the frozen default compiler configuration.
    pub unsafe fn jit_fp8(
        &self,
        device: usize,
        s: &Fp8Specialization<'_>,
    ) -> Result<Arc<CompiledKernel>, Error> {
        unsafe { self.fp8_source(device, s, true) }
    }

    /// # Safety
    /// Device, operand capacities and launch ownership must follow the frozen
    /// original rowwise kernel. The source wrapper supplies its block size.
    pub unsafe fn rowwise(
        &self,
        device: usize,
        s: &crate::triton_frontend::rowwise::Specialization<'_>,
    ) -> Result<Arc<CompiledKernel>, Error> {
        unsafe { self.rowwise_source(device, s, false) }
    }
    /// # Safety
    /// Device, operand capacities and launch ownership must follow the frozen
    /// original rowwise kernel. The source wrapper supplies its block size.
    pub unsafe fn jit_rowwise(
        &self,
        device: usize,
        s: &crate::triton_frontend::rowwise::Specialization<'_>,
    ) -> Result<Arc<CompiledKernel>, Error> {
        unsafe { self.rowwise_source(device, s, true) }
    }

    /// # Safety
    /// Operand capacities, launch shape and stream ownership must satisfy the
    /// original AdaLN wrapper and its original two-pass kernel.
    pub unsafe fn adaln(
        &self,
        device: usize,
        s: &crate::triton_frontend::adaln::Specialization<'_>,
    ) -> Result<Arc<CompiledKernel>, Error> {
        unsafe {
            self.adaln_source(
                device,
                s,
                false,
                crate::triton_frontend::scalar::ScalarType::F32,
            )
        }
    }
    /// # Safety
    /// Operand capacities, launch shape and stream ownership must satisfy the
    /// original AdaLN wrapper and its original two-pass kernel.
    pub unsafe fn jit_adaln(
        &self,
        device: usize,
        s: &crate::triton_frontend::adaln::Specialization<'_>,
    ) -> Result<Arc<CompiledKernel>, Error> {
        unsafe {
            self.adaln_source(
                device,
                s,
                true,
                crate::triton_frontend::scalar::ScalarType::F32,
            )
        }
    }

    /// # Safety
    /// Operand capacities, grid and stream ownership must follow the original
    /// flexible-layout RoPE wrapper.
    pub unsafe fn rope(
        &self,
        device: usize,
        s: &crate::triton_frontend::rope::Specialization<'_>,
    ) -> Result<Arc<CompiledKernel>, Error> {
        unsafe { self.rope_source(device, s, false) }
    }
    /// # Safety
    /// Operand capacities, grid and stream ownership must follow the original
    /// flexible-layout RoPE wrapper.
    pub unsafe fn jit_rope(
        &self,
        device: usize,
        s: &crate::triton_frontend::rope::Specialization<'_>,
    ) -> Result<Arc<CompiledKernel>, Error> {
        unsafe { self.rope_source(device, s, true) }
    }

    /// # Safety
    /// Operand capacities and stream ownership must follow the frozen RMS-RoPE
    /// wrapper, including its normalization materialization boundary.
    pub unsafe fn rms_rope(
        &self,
        device: usize,
        s: &crate::triton_frontend::rope::rms::Specialization<'_>,
    ) -> Result<Arc<CompiledKernel>, Error> {
        unsafe {
            self.rms_rope_source(
                device,
                s,
                false,
                crate::triton_frontend::scalar::ScalarType::F32,
            )
        }
    }
    /// # Safety
    /// Operand capacities and stream ownership must follow the frozen RMS-RoPE
    /// wrapper, including its normalization materialization boundary.
    pub unsafe fn jit_rms_rope(
        &self,
        device: usize,
        s: &crate::triton_frontend::rope::rms::Specialization<'_>,
    ) -> Result<Arc<CompiledKernel>, Error> {
        unsafe {
            self.rms_rope_source(
                device,
                s,
                true,
                crate::triton_frontend::scalar::ScalarType::F32,
            )
        }
    }

    /// # Safety
    /// Operand capacities, grid and stream ownership must follow the original
    /// MXFP8 wrapper, including its incomplete-block no-write behavior.
    pub unsafe fn mxfp8(
        &self,
        device: usize,
        s: &crate::triton_frontend::mxfp8::Specialization<'_>,
    ) -> Result<Arc<CompiledKernel>, Error> {
        unsafe { self.mxfp8_source(device, s, false) }
    }
    /// # Safety
    /// Operand capacities, grid and stream ownership must follow the original
    /// MXFP8 wrapper, including its incomplete-block no-write behavior.
    pub unsafe fn jit_mxfp8(
        &self,
        device: usize,
        s: &crate::triton_frontend::mxfp8::Specialization<'_>,
    ) -> Result<Arc<CompiledKernel>, Error> {
        unsafe { self.mxfp8_source(device, s, true) }
    }

    /// # Safety
    /// Operand capacities and launch ownership follow the original wrapper.
    /// SM89 assembly errors propagate from the actual original ptxas binary.
    pub unsafe fn nvfp4(
        &self,
        device: usize,
        s: &crate::triton_frontend::nvfp4::Specialization<'_>,
    ) -> Result<Arc<CompiledKernel>, Error> {
        unsafe { self.nvfp4_source(device, s, false, None) }
    }
    /// # Safety
    /// Operand capacities and launch ownership follow the original wrapper.
    /// SM89 assembly errors propagate from the actual original ptxas binary.
    pub unsafe fn jit_nvfp4(
        &self,
        device: usize,
        s: &crate::triton_frontend::nvfp4::Specialization<'_>,
    ) -> Result<Arc<CompiledKernel>, Error> {
        unsafe { self.nvfp4_source(device, s, true, None) }
    }

    /// # Safety
    /// Packed inputs, scale capacities and stream ownership follow the original.
    pub unsafe fn dequant_nvfp4(
        &self,
        device: usize,
        s: &crate::triton_frontend::nvfp4::dequantize::Specialization<'_>,
    ) -> Result<Arc<CompiledKernel>, Error> {
        unsafe { self.dequant_nvfp4_source(device, s, false, None) }
    }
    /// # Safety
    /// Packed inputs, scale capacities and stream ownership follow the original.
    pub unsafe fn jit_dequant_nvfp4(
        &self,
        device: usize,
        s: &crate::triton_frontend::nvfp4::dequantize::Specialization<'_>,
    ) -> Result<Arc<CompiledKernel>, Error> {
        unsafe { self.dequant_nvfp4_source(device, s, true, None) }
    }

    /// Original dynamic constexpr identity, distinct from its branch value.
    /// # Safety
    /// Operands and launch ownership must follow the frozen NVFP4 wrapper.
    pub unsafe fn jit_nvfp4_literal(
        &self,
        device: usize,
        s: &crate::triton_frontend::nvfp4::Specialization<'_>,
        literal: &crate::triton_arguments::ConstexprKey,
        invalid_kind: Option<&str>,
    ) -> Result<Arc<CompiledKernel>, Error> {
        unsafe { self.nvfp4_source(device, s, true, Some((literal, invalid_kind))) }
    }
    /// # Safety
    /// Packed operands and launch ownership must follow the frozen wrapper.
    pub unsafe fn jit_dequant_nvfp4_literal(
        &self,
        device: usize,
        s: &crate::triton_frontend::nvfp4::dequantize::Specialization<'_>,
        literal: &crate::triton_arguments::ConstexprKey,
        invalid_kind: Option<&str>,
    ) -> Result<Arc<CompiledKernel>, Error> {
        unsafe { self.dequant_nvfp4_source(device, s, true, Some((literal, invalid_kind))) }
    }
    fn conditional_digests(
        &self,
        key: &str,
        function: &Value,
        jit: bool,
        literal: Option<&crate::triton_arguments::ConstexprKey>,
    ) -> Result<(String, String), Error> {
        let Some(literal) = literal else {
            let ast = self.source_digest(key, function, jit)?;
            return Ok((ast.clone(), ast));
        };
        let names = function["argument_names"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_str().unwrap())
            .collect::<Vec<_>>();
        let index = names.iter().position(|n| *n == "hi_first").unwrap();
        let memory = crate::triton_specialization::source_constexpr_key(
            key,
            &names,
            &[(index, literal.memory.clone())],
        );
        let key = crate::triton_specialization::source_constexpr_key(
            key,
            &names,
            &[(index, literal.source.clone())],
        );
        Ok((
            self.source_digest(&key, function, jit)?,
            self.source_digest(&memory, function, jit)?,
        ))
    }

    /// Compile one unchanged candidate of either original INT8 autotuner.
    /// # Safety
    /// The original candidate index, specialization, operand capacities and
    /// device/stream ownership must correspond to the frozen typed launch.
    pub unsafe fn int8(
        &self,
        device: usize,
        s: &crate::triton_frontend::int8::Specialization<'_>,
        configuration_index: usize,
    ) -> Result<Arc<CompiledKernel>, Error> {
        unsafe { self.int8_source(device, s, configuration_index, false) }
    }
    /// Compile one unchanged candidate of either original INT8 autotuner.
    /// # Safety
    /// The original candidate index, specialization, operand capacities and
    /// device/stream ownership must correspond to the frozen typed launch.
    pub unsafe fn jit_int8(
        &self,
        device: usize,
        s: &crate::triton_frontend::int8::Specialization<'_>,
        configuration_index: usize,
    ) -> Result<Arc<CompiledKernel>, Error> {
        unsafe { self.int8_source(device, s, configuration_index, true) }
    }

    /// Compile the frozen INT4/INT6 grouped unpack kernel.
    /// # Safety
    /// Operand capacities, specialization and device/stream must match the source launch.
    pub unsafe fn w4a8_unpack(
        &self,
        device: usize,
        s: &crate::triton_frontend::w4a8::Specialization<'_>,
    ) -> Result<Arc<CompiledKernel>, Error> {
        unsafe { self.w4a8_unpack_source(device, s, false) }
    }
    /// Compile the frozen INT4/INT6 grouped unpack kernel.
    /// # Safety
    /// Operand capacities, specialization and device/stream must match the source launch.
    pub unsafe fn jit_w4a8_unpack(
        &self,
        device: usize,
        s: &crate::triton_frontend::w4a8::Specialization<'_>,
    ) -> Result<Arc<CompiledKernel>, Error> {
        unsafe { self.w4a8_unpack_source(device, s, true) }
    }

    /// Compile the frozen online-softmax neighborhood attention kernel.
    /// # Safety
    /// Original operand capacities, specializations and stream/device must match.
    pub unsafe fn na3d(
        &self,
        device: usize,
        s: &crate::triton_frontend::na::Specialization<'_>,
    ) -> Result<Arc<CompiledKernel>, Error> {
        unsafe { self.na3d_source(device, s, false) }
    }
    /// Compile the frozen online-softmax neighborhood attention kernel.
    /// # Safety
    /// Original operand capacities, specializations and stream/device must match.
    pub unsafe fn jit_na3d(
        &self,
        device: usize,
        s: &crate::triton_frontend::na::Specialization<'_>,
    ) -> Result<Arc<CompiledKernel>, Error> {
        unsafe { self.na3d_source(device, s, true) }
    }
    /// # Safety
    /// The original scalar signature, argument widths and operand/stream
    /// lifetimes must match the RMS-RoPE specialization.
    pub unsafe fn jit_rms_rope_scalar(
        &self,
        device: usize,
        s: &crate::triton_frontend::rope::rms::Specialization<'_>,
        epsilon: crate::triton_frontend::scalar::ScalarType,
    ) -> Result<Arc<CompiledKernel>, Error> {
        unsafe { self.rms_rope_source(device, s, true, epsilon) }
    }
    /// # Safety
    /// The original Python numeric scalar signature, widths and operand/stream
    /// lifetimes must match the AdaLN specialization.
    pub unsafe fn jit_adaln_scalar(
        &self,
        device: usize,
        s: &crate::triton_frontend::adaln::Specialization<'_>,
        epsilon: crate::triton_frontend::scalar::ScalarType,
    ) -> Result<Arc<CompiledKernel>, Error> {
        unsafe { self.adaln_source(device, s, true, epsilon) }
    }
    /// # Safety
    /// Original scalar/structured argument widths, capacities and stream
    /// ownership must match the frozen wrapper and its typed launch.
    pub unsafe fn jit_adaln_argument(
        &self,
        device: usize,
        s: &crate::triton_frontend::adaln::Specialization<'_>,
        argument: &crate::triton_arguments::ArgumentType,
    ) -> Result<Arc<CompiledKernel>, Error> {
        if let crate::triton_arguments::ArgumentType::Numeric(kind) = argument {
            return unsafe { self.adaln_source(device, s, true, *kind) };
        }
        let function = &self.profile["functions"]["adaln"];
        let key =
            crate::triton_specialization::adaln_ast_key(s, field(function, "function_cache_key")?);
        let position =
            crate::triton_scalar_sources::ADALN_EPSILON_POSITIONS[usize::from(!s.subtract_mean)];
        let failure = crate::triton_frontend::errors::CompilationFailure::at(
            crate::triton_scalar_sources::ADALN_SOURCE,
            crate::triton_scalar_sources::ADALN_DEFINITION,
            position,
            &argument.unsupported_cause().unwrap(),
        );
        unsafe { self.scalar_argument_failure(device, "adaln", &key, 11, argument, failure) }
    }
    /// # Safety
    /// Original scalar/structured argument widths, capacities and stream
    /// ownership must match the frozen wrapper and its typed launch.
    pub unsafe fn jit_rms_rope_argument(
        &self,
        device: usize,
        s: &crate::triton_frontend::rope::rms::Specialization<'_>,
        argument: &crate::triton_arguments::ArgumentType,
    ) -> Result<Arc<CompiledKernel>, Error> {
        if let crate::triton_arguments::ArgumentType::Numeric(kind) = argument {
            return unsafe { self.rms_rope_source(device, s, true, *kind) };
        }
        let function = &self.profile["functions"]["rms_rope"];
        let key = crate::triton_specialization::rms_rope_ast_key(
            s,
            field(function, "function_cache_key")?,
        );
        let failure = crate::triton_frontend::errors::CompilationFailure::at(
            crate::triton_scalar_sources::RMS_ROPE_SOURCE,
            crate::triton_scalar_sources::RMS_ROPE_DEFINITION,
            crate::triton_scalar_sources::RMS_ROPE_EPSILON_POSITIONS[0],
            &argument.unsupported_cause().unwrap(),
        );
        unsafe { self.scalar_argument_failure(device, "rms_rope", &key, 20, argument, failure) }
    }
    /// Original NA constexpr values and scalar/structured scale specialization.
    /// # Safety
    /// Values, typed pointers and capacities must match the frozen wrapper.
    pub unsafe fn jit_na3d_arguments(
        &self,
        device: usize,
        s: &crate::triton_frontend::na::Specialization<'_>,
        kernels: [crate::triton_frontend::na::KernelValue; 3],
        literals: &[(usize, crate::triton_arguments::ConstexprKey)],
        scale: &crate::triton_arguments::ArgumentType,
        failure: Option<crate::triton_frontend::errors::CompilationFailure>,
    ) -> Result<Arc<CompiledKernel>, Error> {
        let function = &self.profile["functions"]["na3d"];
        let s = crate::triton_frontend::na::Specialization {
            source_path: field(function, "source_path")?,
            standard_source_path: field(function, "standard_source_path")?,
            ..*s
        };
        let names = function["argument_names"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_str().unwrap())
            .collect::<Vec<_>>();
        let key =
            crate::triton_specialization::na_ast_key(&s, field(function, "function_cache_key")?);
        let key = crate::triton_specialization::source_argument_key(&key, &names, 13, scale);
        let memory_key = crate::triton_specialization::source_constexpr_key(
            &key,
            &names,
            &literals
                .iter()
                .map(|(i, k)| (*i, k.memory.clone()))
                .collect::<Vec<_>>(),
        );
        let memory = self.source_digest(&memory_key, function, true)?;
        let key = crate::triton_specialization::source_constexpr_key(
            &key,
            &names,
            &literals
                .iter()
                .map(|(i, k)| (*i, k.source.clone()))
                .collect::<Vec<_>>(),
        );
        let ast = self.source_digest(&key, function, true)?;
        unsafe {
            self.specialization_with_memory_key(
                device,
                "na3d",
                &ast,
                &memory,
                &self.profile,
                || {
                    let failure = failure
                        .or_else(|| crate::triton_frontend::na::kernel_failure(&s, kernels))
                        .or_else(|| {
                            scale.unsupported_cause().map(|cause| {
                                crate::triton_frontend::errors::CompilationFailure::at(
                                    crate::triton_frontend::na::FUNCTION_SOURCE,
                                    crate::triton_frontend::na::FUNCTION_DEFINITION,
                                    if s.input == crate::triton_frontend::Element::F32 {
                                        (93, 24)
                                    } else {
                                        (95, 24)
                                    },
                                    &cause,
                                )
                            })
                        });
                    if let Some(failure) = failure {
                        return Err(Error::ReferenceException {
                            class: failure.class,
                            message: failure.message,
                        });
                    }
                    crate::triton_frontend::na::checked_source(&s).map_err(|e| {
                        Error::ReferenceException {
                            class: e.class,
                            message: e.message,
                        }
                    })?;
                    Ok(crate::triton_frontend::na::source_ttir_kernels(&s, kernels))
                },
            )
        }
    }
    unsafe fn scalar_argument_failure(
        &self,
        device: usize,
        operation: &str,
        key: &str,
        index: usize,
        argument: &crate::triton_arguments::ArgumentType,
        failure: crate::triton_frontend::errors::CompilationFailure,
    ) -> Result<Arc<CompiledKernel>, Error> {
        let function = &self.profile["functions"][operation];
        let names = function["argument_names"]
            .as_array()
            .unwrap()
            .iter()
            .map(|name| name.as_str().unwrap())
            .collect::<Vec<_>>();
        let key = crate::triton_specialization::source_argument_key(key, &names, index, argument);
        let ast = self.source_digest(&key, function, true)?;
        unsafe {
            self.specialization_with_configuration_result(
                device,
                operation,
                &ast,
                &self.profile,
                || {
                    Err(Error::ReferenceException {
                        class: failure.class,
                        message: failure.message,
                    })
                },
            )
        }
    }
    fn source_digest(&self, key: &str, function: &Value, jit: bool) -> Result<String, Error> {
        if !jit {
            return Ok(crate::triton_specialization::source_key_digest(key));
        }
        let names = function["argument_names"]
            .as_array()
            .ok_or_else(|| Error::Native("Missing original JIT argument names".into()))?
            .iter()
            .map(|n| n.as_str().unwrap())
            .collect::<Vec<_>>();
        Ok(crate::triton_specialization::jit_source_digest(key, &names))
    }
    /// Native representation of the original compilation knob scope.
    /// JIT memory hits precede these lower-level compilation modes.
    pub fn set_compilation_modes(&self, modes: crate::triton_compilation_modes::Modes) {
        *self.modes.write().unwrap() = modes;
    }
    /// Clear the original-equivalent device specialization maps. Disk files and
    /// already returned kernel handles retain their original ownership.
    pub fn clear_specializations(&self) {
        self.kernels.lock().unwrap().clear();
    }
    unsafe fn specialization(
        &self,
        device: usize,
        op: &str,
        ast: &str,
        source: impl FnOnce() -> String,
    ) -> Result<Arc<CompiledKernel>, Error> {
        unsafe { self.specialization_with_configuration(device, op, ast, &self.profile, source) }
    }
    unsafe fn specialization_with_configuration(
        &self,
        device: usize,
        op: &str,
        ast: &str,
        configuration: &Value,
        source: impl FnOnce() -> String,
    ) -> Result<Arc<CompiledKernel>, Error> {
        unsafe {
            self.specialization_with_configuration_result(device, op, ast, configuration, || {
                Ok(source())
            })
        }
    }
    unsafe fn specialization_with_configuration_result(
        &self,
        device: usize,
        op: &str,
        ast: &str,
        configuration: &Value,
        source: impl FnOnce() -> Result<String, Error>,
    ) -> Result<Arc<CompiledKernel>, Error> {
        unsafe { self.specialization_with_memory_key(device, op, ast, ast, configuration, source) }
    }
    unsafe fn specialization_with_memory_key(
        &self,
        device: usize,
        op: &str,
        ast: &str,
        memory_ast: &str,
        configuration: &Value,
        source: impl FnOnce() -> Result<String, Error>,
    ) -> Result<Arc<CompiledKernel>, Error> {
        let function = &self.profile["functions"][op];
        let hashes = CompilerHashInputs {
            triton_version_key: field(&self.profile, "triton_version_key")?,
            backend_hash: field(&self.profile, "backend_hash")?,
            options_hash: field(configuration, "options_hash")?,
        };
        // The JIT specialization/options key does not include invalidating
        // environment values; those are consulted only by the cold compiler.
        let key = (
            device,
            compiler_digest(&hashes, memory_ast, &BTreeMap::new()),
        );
        if let Some(kernel) = self.kernels.lock().unwrap().get(&key) {
            return Ok(kernel.clone());
        }
        let modes = self.modes.read().unwrap().clone();
        let environment = match &modes.environment {
            Some(values) => values.clone(),
            None => crate::triton_environment::from_process()?,
        };
        let digest = compiler_digest(&hashes, ast, &environment);
        let cache = FileCache::new(&self.cache, &cache_directory(&digest)?, Mode::Cache)?;
        let overrides = modes
            .override_directory
            .as_ref()
            .map(|root| FileCache::new(root, &cache_directory(ast)?, Mode::Override))
            .transpose()?;
        let dump = modes
            .dump_directory
            .as_ref()
            .map(|root| FileCache::new(root, &cache_directory(ast)?, Mode::Dump))
            .transpose()?;
        let name = field(function, "function_name")?;
        let metadata_name = format!("{name}.json");
        let existing = cache.get_group(&metadata_name)?.unwrap_or_default();
        let cached = existing
            .iter()
            .any(|(key, _)| key.to_utf8().as_deref() == Some(&metadata_name));
        let cached = cached && !modes.always_compile;
        let group = if cached {
            existing
        } else {
            self.compile(
                &cache,
                existing,
                name,
                &source()?,
                function,
                configuration,
                &digest,
                &modes,
                overrides.as_ref(),
                dump.as_ref(),
                &environment,
            )?
        };
        let compiled = unsafe { self.load_group(device, &digest, ast, group, cached)? };
        if let Some(listener) = &modes.listener {
            listener(&crate::triton_compilation_modes::Compilation {
                ast_digest: ast,
                metadata: &compiled.metadata,
                metadata_group: &compiled.metadata_group,
                cache_hit: cached,
                lowering_stages: if cached {
                    &[]
                } else {
                    &["ttir", "ttgir", "llir", "ptx", "cubin"]
                },
            })?;
        }
        let result = Arc::new(compiled);
        self.kernels.lock().unwrap().insert(key, result.clone());
        Ok(result)
    }
    #[allow(clippy::too_many_arguments)]
    fn compile(
        &self,
        cache: &FileCache,
        mut group: Vec<(CacheText, JsonPath)>,
        name: &str,
        source: &str,
        function: &Value,
        configuration: &Value,
        digest: &str,
        modes: &crate::triton_compilation_modes::Modes,
        overrides: Option<&FileCache>,
        dump: Option<&FileCache>,
        environment: &BTreeMap<String, String>,
    ) -> Result<Vec<(CacheText, JsonPath)>, Error> {
        fs::create_dir_all(&self.workspace).map_err(|e| os_error(e, &self.workspace, None))?;
        let mut work = WorkingDirectory(
            self.workspace
                .join(format!("tmp.pid_{}_{}", std::process::id(), uuid4()?)),
            false,
        );
        fs::create_dir(&work.0).map_err(|e| os_error(e, &work.0, None))?;
        let raw = work.0.join("raw.ttir");
        fs::write(&raw, source).map_err(|e| os_error(e, &raw, None))?;
        let pretty_source = work.0.join("source");
        let mut previous = self.optimize(&raw, &pretty_source, None)?;
        let save = |group: &mut Vec<(CacheText, JsonPath)>,
                    extension: &str,
                    data: &[u8]|
         -> Result<(), Error> {
            Self::save(cache, group, &format!("{name}.{extension}"), data)
        };
        save(&mut group, "source", &read(&pretty_source)?)?;
        for (extension, pipeline) in [("ttir", "ttir_pipeline"), ("ttgir", "ttgir_pipeline")] {
            let output = work.0.join(extension);
            previous = self.optimize(&previous, &output, Some(field(configuration, pipeline)?))?;
            if let Some(path) =
                overrides.and_then(|manager| manager.get_file(&format!("{name}.{extension}")))
            {
                println!("\nOverriding kernel with file {}", path.display());
                previous = self.optimize(&path, &output, None)?;
            }
            let data = read(&output)?;
            if !modes.store_binary_only {
                save(&mut group, extension, &data)?;
            }
            if let Some(dump) = dump {
                dump.put(&data, &format!("{name}.{extension}"))?;
            }
        }
        let llvm_mlir_path = work.0.join("llvm.mlir");
        previous = self.optimize(
            &previous,
            &llvm_mlir_path,
            Some(field(configuration, "llir_pipeline")?),
        )?;
        let llvm = work.0.join("unoptimized.ll");
        self.command(
            "mlir_translate",
            &[
                "--mlir-to-llvmir",
                previous.to_str().unwrap(),
                "-o",
                llvm.to_str().unwrap(),
            ],
        )?;
        let llir = work.0.join("llir");
        let mut optimizer_args = vec![
            "optimize",
            llvm.to_str().unwrap(),
            llir.to_str().unwrap(),
            "sm_89",
            "+ptx87",
            "87",
        ];
        for library in self.profile["external_libraries"].as_array().unwrap() {
            optimizer_args.push(field(library, "path")?);
        }
        self.command("llvm_driver", &optimizer_args)?;
        self.finish_stage(
            cache, &mut group, name, "llir", &llir, modes, overrides, dump,
        )?;
        let ptx = work.0.join("ptx");
        self.command(
            "llvm_driver",
            &[
                "ptx",
                llir.to_str().unwrap(),
                ptx.to_str().unwrap(),
                "sm_89",
                "+ptx87",
                "87",
            ],
        )?;
        self.finish_stage(cache, &mut group, name, "ptx", &ptx, modes, overrides, dump)?;
        let cubin = work.0.join("cubin");
        let assembled = self.command(
            "ptxas",
            &[
                "-lineinfo",
                "-v",
                "--regAllocOptLevel=2",
                "--gpu-name=sm_89",
                ptx.to_str().unwrap(),
                "-o",
                cubin.to_str().unwrap(),
            ],
        );
        if let Err(e) = assembled {
            if let Error::ReferenceException {
                class: "PTXASError",
                message,
            } = e.root()
            {
                let diagnostic = message.strip_prefix("PTXAS error: ").unwrap_or(message);
                let source = utf8(&read(&ptx)?)?;
                print!(
                    "\n\n================================================================\n{diagnostic}\n\n{source}\n================================================================\nplease share the reproducer above with Triton project.\n\n"
                );
            }
            work.1 = matches!(
                e.root(),
                Error::ReferenceException {
                    class: "PTXASError",
                    ..
                }
            );
            return Err(e);
        }
        self.finish_stage(
            cache, &mut group, name, "cubin", &cubin, modes, overrides, dump,
        )?;
        let mut metadata = field(
            if configuration.get("metadata_json_template").is_some() {
                configuration
            } else {
                function
            },
            "metadata_json_template",
        )?
        .replace("@COMPILER_DIGEST@", digest);
        if !environment.is_empty() {
            let text = environment
                .iter()
                .map(|(k, v)| {
                    format!(
                        "{}: {}",
                        crate::triton_file_cache::json_text(&CacheText::from(k.as_str())),
                        crate::triton_file_cache::json_text(&CacheText::from(v.as_str()))
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            metadata = metadata.replace(
                "\"triton_version\":",
                &format!("{text}, \"triton_version\":"),
            );
        }
        let llvm_mlir = utf8(&read(&work.0.join("llvm.mlir"))?)?;
        let attrs = llvm_mlir
            .lines()
            .find(|line| line.starts_with("module attributes "))
            .unwrap_or("");
        let integer = |name: &str| -> Option<i64> {
            let needle = format!("{name} = ");
            let quoted = format!("\"{name}\" = ");
            let value = attrs
                .split_once(&needle)
                .or_else(|| attrs.split_once(&quoted))?
                .1;
            value.split(" : ").next()?.parse().ok()
        };
        for (name, attr, default) in [
            ("num_warps", "ttg.total-num-warps", None),
            ("shared", "ttg.shared", None),
            ("tmem_size", "ttg.tensor_memory_size", None),
            (
                "global_scratch_size",
                "ttg.global_scratch_memory_size",
                None,
            ),
            (
                "global_scratch_align",
                "ttg.global_scratch_memory_alignment",
                None,
            ),
            (
                "profile_scratch_size",
                "ttg.profile_scratch_memory_size",
                Some(0),
            ),
            (
                "profile_scratch_align",
                "ttg.profile_scratch_memory_alignment",
                Some(1),
            ),
        ] {
            let value = if default.is_some() {
                integer(attr).filter(|n| *n != 0).or(default)
            } else {
                integer(attr)
            };
            if name == "num_warps" && value.is_none() {
                continue;
            }
            let prefix = format!("\"{name}\": ");
            let begin = metadata
                .find(&prefix)
                .ok_or_else(|| Error::Native(format!("Missing original metadata field {name}")))?
                + prefix.len();
            let end = begin
                + metadata[begin..]
                    .find([',', '}'])
                    .ok_or_else(|| Error::Native("Malformed original metadata template".into()))?;
            metadata.replace_range(
                begin..end,
                &value.map_or_else(|| "null".into(), |n| n.to_string()),
            );
        }
        save(&mut group, "json", metadata.as_bytes())?;
        cache.put_group_values(&format!("{name}.json"), &group)?;
        Ok(group)
    }
    fn save(
        cache: &FileCache,
        group: &mut Vec<(CacheText, JsonPath)>,
        filename: &str,
        data: &[u8],
    ) -> Result<(), Error> {
        let path = cache.put(data, filename)?;
        let key = CacheText::from(filename);
        let value = JsonPath::Text(CacheText::from(path.to_str().unwrap()));
        if let Some((_, old)) = group.iter_mut().find(|(k, _)| k == &key) {
            *old = value;
        } else {
            group.push((key, value));
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn finish_stage(
        &self,
        cache: &FileCache,
        group: &mut Vec<(CacheText, JsonPath)>,
        name: &str,
        extension: &str,
        output: &Path,
        modes: &crate::triton_compilation_modes::Modes,
        overrides: Option<&FileCache>,
        dump: Option<&FileCache>,
    ) -> Result<(), Error> {
        let filename = format!("{name}.{extension}");
        if let Some(path) = overrides.and_then(|manager| manager.get_file(&filename)) {
            println!("\nOverriding kernel with file {}", path.display());
            fs::write(output, read(&path)?).map_err(|e| os_error(e, output, None))?;
        }
        let data = read(output)?;
        if !modes.store_binary_only || extension == "cubin" {
            Self::save(cache, group, &filename, &data)?;
        }
        if let Some(dump) = dump {
            dump.put(&data, &filename)?;
            if extension == "cubin" {
                let tool = Path::new(self.tool("cuobjdump"));
                let result = Command::new(tool)
                    .arg("-sass")
                    .arg(output)
                    .output()
                    .map_err(|e| os_error(e, tool, None))?;
                use std::io::Write;
                std::io::stderr()
                    .write_all(&result.stderr)
                    .map_err(|e| Error::Native(e.to_string()))?;
                if !result.status.success() {
                    return Err(error(
                        "CalledProcessError",
                        format!(
                            "Command '{}' returned non-zero exit status {}.",
                            tool.display(),
                            result.status.code().unwrap_or(1)
                        ),
                    ));
                }
                let text = utf8(&result.stdout)?;
                let sass =
                    crate::triton_compilation_modes::sass(&text).unwrap_or_else(|| "None".into());
                dump.put(sass.as_bytes(), &format!("{name}.sass"))?;
            }
        }
        Ok(())
    }
    unsafe fn load_group(
        &self,
        device: usize,
        digest: &str,
        ast: &str,
        group: Vec<(CacheText, JsonPath)>,
        cache_hit: bool,
    ) -> Result<CompiledKernel, Error> {
        let path = |p: &JsonPath| -> Result<PathBuf, Error> {
            match p {
                JsonPath::Text(s) => s.filesystem_path().ok_or_else(|| {
                    error("UnicodeEncodeError", "Unencodable cache filesystem path")
                }),
                _ => Err(error(
                    "TypeError",
                    "expected str, bytes or os.PathLike object",
                )),
            }
        };
        let (_, metadata_path) = group
            .iter()
            .find(|(k, _)| k.to_utf8().is_some_and(|k| k.ends_with(".json")))
            .ok_or_else(|| error("StopIteration", ""))?;
        let metadata: Value = serde_json::from_str(&utf8(&read(&path(metadata_path)?)?)?)
            .map_err(|e| error("JSONDecodeError", e.to_string()))?;
        for name in ["num_warps", "num_ctas", "shared", "name"] {
            if metadata.get(name).is_none() {
                return Err(error(
                    "AttributeError",
                    format!("'KernelMetadata' object has no attribute '{name}'"),
                ));
            }
        }
        let mut assembly = vec![];
        for (key, value) in &group {
            if key.to_utf8().is_some_and(|k| k.ends_with(".json")) {
                continue;
            }
            let path = path(value)?;
            let extension = path
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_string();
            let bytes = read(&path)?;
            if extension != "cubin" {
                utf8(&bytes)?;
            }
            if let Some((_, old)) = assembly.iter_mut().find(|(k, _)| k == &extension) {
                *old = bytes;
            } else {
                assembly.push((extension, bytes));
            }
        }
        let _cubin = assembly.iter().find(|(k, _)| k == "cubin").ok_or_else(|| {
            error(
                "KeyError",
                crate::triton_specialization::python_str_repr("Unknown key: 'cubin'"),
            )
        })?;
        Ok(CompiledKernel {
            digest: digest.into(),
            ast_digest: ast.into(),
            metadata,
            metadata_group: group,
            assembly,
            disk_cache_hit: cache_hit,
            driver: self.driver.clone(),
            native: self.native.clone(),
            device,
            state: Mutex::new(HandleState::Uninitialized),
            hooks: self.hooks.clone(),
        })
    }
}
