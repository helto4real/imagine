//! Executable registry and general QuantizedTensor dispatch order.
//! Unbound reference operations remain explicit development gaps, never fallbacks.
use crate::{
    backend_contract, boxed_aten,
    execution_value::{Kwargs, Value, get, type_error},
    native::{Error, Kitchen},
    registry::{BackendRegistry, CallArgs, FunctionConstraints},
    storage::QuantizedStorage,
};
use std::{
    collections::BTreeMap,
    sync::{Arc, RwLock},
};
use tch::Tensor;

pub type Callable = Arc<dyn Fn(&Runtime, &[Value], &Kwargs) -> Result<Value, Error> + Send + Sync>;
pub type Decoder = Arc<dyn Fn(&Runtime, &QuantizedStorage) -> Result<Tensor, Error> + Send + Sync>;
pub struct Resolved {
    pub backend: String,
    pub operation: String,
    body: Callable,
}
impl Resolved {
    /// # Safety
    /// All selected native operands must satisfy the frozen launch contract.
    pub unsafe fn call(
        &self,
        rt: &Runtime,
        args: &[Value],
        kwargs: &Kwargs,
    ) -> Result<Value, Error> {
        (self.body)(rt, args, kwargs)
    }
}
pub struct ExecutableRegistry {
    pub policy: BackendRegistry,
    bodies: RwLock<BTreeMap<(String, String), Callable>>,
    observer: RwLock<Option<SelectionObserver>>,
}
pub type SelectionObserver = Arc<dyn Fn(&str, &str) + Send + Sync>;
impl ExecutableRegistry {
    pub fn new(policy: BackendRegistry) -> Self {
        Self {
            policy,
            bodies: RwLock::new(BTreeMap::new()),
            observer: RwLock::new(None),
        }
    }
    pub fn register(
        &self,
        name: &str,
        constraints: BTreeMap<String, FunctionConstraints>,
        bodies: BTreeMap<String, Callable>,
    ) {
        {
            let mut table = self.bodies.write().unwrap();
            table.retain(|(backend, _), _| backend != name);
            for (op, body) in bodies {
                table.insert((name.into(), op), body);
            }
        }
        self.policy.register(name, constraints);
    }
    pub fn bind(&self, backend: &str, op: &str, body: Callable) {
        self.bodies
            .write()
            .unwrap()
            .insert((backend.into(), op.into()), body);
    }
    /// The frozen Python callable validates binding before entering its body.
    /// User replacements installed through `bind` retain their own contract.
    pub(crate) fn bind_reference(&self, backend: &str, op: &str, body: Callable) {
        let Some((names, required)) = crate::backend_signatures::signature(backend, op) else {
            self.bind(backend, op, body);
            return;
        };
        let name = op.to_string();
        self.bind(
            backend,
            op,
            Arc::new(move |rt, args, kwargs| {
                crate::execution_value::bind_arguments(&name, names, required, args, kwargs)?;
                if args
                    .iter()
                    .chain(kwargs.iter().map(|(_, v)| v))
                    .any(crate::execution_value::small_unsigned)
                {
                    let args = args
                        .iter()
                        .map(crate::execution_value::python_integer_value)
                        .collect::<Vec<_>>();
                    let kwargs = kwargs
                        .iter()
                        .map(|(k, v)| (k.clone(), crate::execution_value::python_integer_value(v)))
                        .collect();
                    body(rt, &args, &kwargs)
                } else {
                    body(rt, args, kwargs)
                }
            }),
        );
    }
    pub(crate) unsafe fn call_bound(
        &self,
        rt: &Runtime,
        backend: &str,
        op: &str,
        kwargs: &Kwargs,
    ) -> Result<Value, Error> {
        let f = self
            .bodies
            .read()
            .unwrap()
            .get(&(backend.into(), op.into()))
            .cloned()
            .ok_or_else(|| Error::BindingGap {
                backend: backend.into(),
                operation: op.into(),
            })?;
        f(rt, &[], kwargs)
    }
    pub fn resolve(
        &self,
        op: &str,
        explicit: Option<&str>,
        metadata: &CallArgs,
    ) -> Result<Resolved, Error> {
        let backend = self.policy.select(op, explicit, metadata)?;
        let body = self
            .bodies
            .read()
            .unwrap()
            .get(&(backend.clone(), op.into()))
            .cloned()
            .ok_or_else(|| Error::BindingGap {
                backend: backend.clone(),
                operation: op.into(),
            })?;
        let observer = self.observer.read().unwrap().clone();
        if let Some(observer) = observer {
            observer(op, &backend);
        }
        Ok(Resolved {
            backend,
            operation: op.into(),
            body,
        })
    }
    pub fn observe_selection(&self, observer: Option<SelectionObserver>) {
        *self.observer.write().unwrap() = observer;
    }
    pub fn gaps(&self) -> Vec<(String, String)> {
        let bodies = self.bodies.read().unwrap();
        self.policy
            .list_backends()
            .into_iter()
            .flat_map(|(b, s)| s.capabilities.into_iter().map(move |op| (b.clone(), op)))
            .filter(|k| !bodies.contains_key(k))
            .collect()
    }
}
#[derive(Clone)]
struct Layout {
    mro: Vec<String>,
    decoder: Option<Decoder>,
}
/// Isolated development observer; ordinary execution has no observer installed.
/// `before` is true before the original scatter and false after it, before division.
pub type SolScatterObserver =
    Arc<dyn Fn(bool, i64, &mut Tensor, &Tensor, &Tensor, u64) -> Result<(), Error> + Send + Sync>;
pub struct Runtime {
    pub native: Kitchen,
    /// The original CUDA module import-time COMFY_KITCHEN_DISABLE_CUTLASS flag.
    pub disable_cutlass_int8: bool,
    /// Frozen CUDA module flag; changing it reproduces that module's state.
    pub w4a8_chunked: bool,
    pub force_int4_int8_fallback: bool,
    pub int4_weight_chunk_n: Option<i64>,
    pub backends: ExecutableRegistry,
    sol_scatter_observer: RwLock<Option<SolScatterObserver>>,
    layouts: RwLock<BTreeMap<String, Layout>>,
    generic: RwLock<BTreeMap<String, Callable>>,
    handlers: RwLock<BTreeMap<(String, String), Callable>>,
}
impl Runtime {
    /// Test-only controlled-intermediate inspection (D-05-SOL-ATTN-EAGER-FP32-SCATTER).
    /// The caller must enforce the approved scope; defaults and ordinary calls use None.
    pub fn observe_sol_scatter(&self, observer: Option<SolScatterObserver>) {
        *self.sol_scatter_observer.write().unwrap() = observer;
    }
    pub(crate) fn sol_scatter(
        &self,
        out: &mut Tensor,
        index: &Tensor,
        src: &Tensor,
    ) -> Result<(), Error> {
        let observer = self.sol_scatter_observer.read().unwrap().clone();
        let stream = if observer.is_some() {
            match out.device() {
                tch::Device::Cuda(i) => self.native.current_stream(i)?,
                _ => 0,
            }
        } else {
            0
        };
        if let Some(f) = &observer {
            f(true, 2, out, index, src, stream)?;
        }
        let _ = out.f_scatter_add_(2, index, src)?;
        if let Some(f) = &observer {
            let current = match out.device() {
                tch::Device::Cuda(i) => self.native.current_stream(i)?,
                _ => 0,
            };
            f(false, 2, out, index, src, current)?;
        }
        Ok(())
    }

    pub fn layout_mros(&self) -> BTreeMap<String, Vec<String>> {
        self.layouts
            .read()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.mro.clone()))
            .collect()
    }
    pub fn generic_slots(&self) -> Vec<String> {
        self.generic.read().unwrap().keys().cloned().collect()
    }
    pub fn handler_slots(&self) -> BTreeMap<String, Vec<String>> {
        let mut result = BTreeMap::<String, Vec<String>>::new();
        for (op, layout) in self.handlers.read().unwrap().keys() {
            result.entry(op.clone()).or_default().push(layout.clone());
        }
        result
    }
    pub fn new(native: Kitchen, policy: BackendRegistry) -> Self {
        Self {
            native,
            disable_cutlass_int8: std::env::var("COMFY_KITCHEN_DISABLE_CUTLASS")
                .is_ok_and(|s| s == "1"),
            w4a8_chunked: std::env::var_os("COMFY_KITCHEN_W4A8_CHUNKED").is_none_or(|s| s != "0"),
            force_int4_int8_fallback: std::env::var_os("COMFY_KITCHEN_FORCE_INT4_INT8_FALLBACK")
                .is_some_and(|s| s == "1"),
            int4_weight_chunk_n: std::env::var("COMFY_KITCHEN_INT4_INT8_WEIGHT_CHUNK_N")
                .unwrap_or_else(|_| "4096".into())
                .parse::<i64>()
                .ok()
                .map(|n| n.max(1)),
            backends: ExecutableRegistry::new(policy),
            sol_scatter_observer: RwLock::new(None),
            layouts: RwLock::new(BTreeMap::new()),
            generic: RwLock::new(BTreeMap::new()),
            handlers: RwLock::new(BTreeMap::new()),
        }
    }
    /// Install every registered body in the frozen initial-platform reference.
    /// Remaining argument/configuration parity is tracked by the Stage 1 ledger.
    pub fn frozen(
        native: Kitchen,
        compiler: Arc<crate::triton_compiler::Compiler>,
    ) -> Result<Self, Error> {
        let rt = Self::frozen_partial(native)?;
        rt.bind_triton_fp8(compiler.clone());
        rt.bind_triton_rowwise(compiler.clone());
        rt.bind_triton_adaln(compiler.clone());
        rt.bind_triton_rope(compiler.clone());
        rt.bind_triton_rms_rope(compiler.clone());
        rt.bind_triton_mxfp8(compiler.clone());
        rt.bind_triton_nvfp4(compiler.clone());
        rt.bind_triton_int8(compiler.clone());
        rt.bind_triton_w4a8(compiler.clone());
        rt.bind_triton_na(compiler);
        Ok(rt)
    }
    /// Reference metadata is complete even while this development runtime has
    /// binding gaps. It must not be published as a complete inference backend.
    pub fn frozen_partial(native: Kitchen) -> Result<Self, Error> {
        let chunk_bound = crate::python_numbers::chunk_bound_from_process()?;
        let probe = native.clone();
        let mut rt = Self::new(
            native,
            BackendRegistry::with_capability_probe(move || probe.current_compute_capability().ok()),
        );
        rt.int4_weight_chunk_n = Some(chunk_bound);
        let lt = rt.native.cublaslt_available()?;
        for b in ["cuda", "triton", "eager"] {
            rt.backends.register(
                b,
                backend_contract::constraints(b, lt).unwrap(),
                BTreeMap::new(),
            );
        }
        rt.backends
            .policy
            .mark_unavailable("hip", "PyTorch ROCm/HIP runtime not available");
        rt.backends.policy.mark_unavailable(
            "ascend",
            "torch-npu is not installed: No module named 'torch_npu'",
        );
        crate::execution_bindings::install(&rt);
        Ok(rt)
    }
    pub fn register_layout(&self, name: &str, mro: Vec<String>, decoder: Option<Decoder>) {
        self.layouts
            .write()
            .unwrap()
            .insert(name.into(), Layout { mro, decoder });
    }
    pub fn register_generic(&self, op: &str, handler: Callable) {
        self.generic.write().unwrap().insert(op.into(), handler);
    }
    pub fn register_handler(&self, op: &str, layout: &str, handler: Callable) {
        self.handlers
            .write()
            .unwrap()
            .insert((op.into(), layout.into()), handler);
    }
    pub fn mro(&self, name: &str) -> Result<Vec<String>, Error> {
        self.layouts
            .read()
            .unwrap()
            .get(name)
            .map(|l| l.mro.clone())
            .ok_or_else(|| Error::ReferenceException {
                class: "KeyError",
                message: format!("'{name}'"),
            })
    }
    /// # Safety
    /// Selected native operands, capacities and active device/stream must satisfy
    /// the frozen operation contract. General validation does not prove raw capacities.
    pub unsafe fn decode(&self, q: &QuantizedStorage) -> Result<Tensor, Error> {
        let mro = self.mro(&q.layout)?;
        let decoder = {
            let defs = self.layouts.read().unwrap();
            mro.iter()
                .find_map(|name| defs.get(name).and_then(|d| d.decoder.clone()))
        };
        decoder.ok_or_else(|| Error::BindingGap {
            backend: "layout".into(),
            operation: format!("{}.dequantize", q.layout),
        })?(self, q)
    }
    /// # Safety
    /// Selected native operands, capacities and active device/stream must satisfy
    /// the frozen operation contract. General validation does not prove raw capacities.
    pub unsafe fn dequantize_args(&self, v: &Value) -> Result<Value, Error> {
        unsafe {
            Ok(match v {
                Value::Quantized(q) => {
                    let snapshot = q.lock().unwrap().shallow_parts();
                    Value::tensor(self.decode(&snapshot)?)
                }
                Value::Tuple(v) => Value::Tuple(
                    v.iter()
                        .map(|v| self.dequantize_args(v))
                        .collect::<Result<_, _>>()?,
                ),
                Value::List(v) => Value::List(
                    v.iter()
                        .map(|v| self.dequantize_args(v))
                        .collect::<Result<_, _>>()?,
                ),
                Value::Dict(v) => Value::Dict(self.decode_kwargs(v)?),
                _ => v.clone(),
            })
        }
    }
    fn decode_kwargs(&self, v: &Kwargs) -> Result<Kwargs, Error> {
        v.iter()
            .map(|(k, v)| Ok((k.clone(), unsafe { self.dequantize_args(v)? })))
            .collect()
    }
    /// # Safety
    /// Selected native operands, capacities and active device/stream must satisfy
    /// the frozen operation contract. General validation does not prove raw capacities.
    pub unsafe fn call_kitchen(&self, op: &str, kwargs: &Kwargs) -> Result<Value, Error> {
        unsafe {
            // Custom ops without a layout handler use the same generic recursive
            // dequantization before their registry lookup, including quantized bias.
            let kwargs = self.decode_kwargs(kwargs)?;
            let metadata = kwargs
                .iter()
                .map(|(k, v)| Ok((k.clone(), v.metadata_for_call(self)?)))
                .collect::<Result<_, Error>>()?;
            self.backends
                .resolve(op, None, &metadata)?
                .call(self, &[], &kwargs)
        }
    }
    /// # Safety
    /// Same selected native contracts as dispatch. Ordinary values bypass the
    /// tensor-subclass dispatcher, as LibTorch does in the reference.
    pub unsafe fn op(&self, op: &str, args: &[Value], kwargs: &Kwargs) -> Result<Value, Error> {
        fn has(v: &Value) -> bool {
            match v {
                Value::Quantized(_) => true,
                Value::Tuple(v) | Value::List(v) => v.iter().any(has),
                Value::Dict(v) => v.iter().any(|(_, v)| has(v)),
                _ => false,
            }
        }
        unsafe {
            if args.iter().any(has) || kwargs.iter().any(|(_, v)| has(v)) {
                self.dispatch(op, args, kwargs)
            } else {
                self.ordinary(op, args, kwargs)
            }
        }
    }
    /// # Safety
    /// Selected native operands, capacities and active device/stream must satisfy
    /// the frozen operation contract. General validation does not prove raw capacities.
    pub unsafe fn dispatch(
        &self,
        op: &str,
        args: &[Value],
        kwargs: &Kwargs,
    ) -> Result<Value, Error> {
        unsafe {
            let generic = self.generic.read().unwrap().get(op).cloned();
            if let Some(f) = generic {
                return f(self, args, kwargs);
            }
            // Frozen _get_layout_from_args examines direct arguments and exactly
            // one list/tuple level; it does not select a layout from kwargs/dicts.
            let mut selected = None;
            for arg in args {
                match arg {
                    Value::Quantized(q) => {
                        selected = Some(q.lock().unwrap().layout.clone());
                        break;
                    }
                    Value::List(v) | Value::Tuple(v) => {
                        if let Some(q) = v.iter().find_map(|v| {
                            if let Value::Quantized(q) = v {
                                Some(q)
                            } else {
                                None
                            }
                        }) {
                            selected = Some(q.lock().unwrap().layout.clone());
                            break;
                        }
                    }
                    _ => (),
                }
            }
            if let Some(layout) = selected {
                for parent in self.mro(&layout)? {
                    let handler = self
                        .handlers
                        .read()
                        .unwrap()
                        .get(&(op.into(), parent))
                        .cloned();
                    if let Some(f) = handler {
                        return f(self, args, kwargs);
                    }
                }
            }
            let args = args
                .iter()
                .map(|v| self.dequantize_args(v))
                .collect::<Result<Vec<_>, _>>()?;
            let kwargs = self.decode_kwargs(kwargs)?;
            self.ordinary(op, &args, &kwargs)
        }
    }
    /// # Safety
    /// Selected native operands, capacities and active device/stream must satisfy
    /// the frozen operation contract. General validation does not prove raw capacities.
    pub unsafe fn ordinary(
        &self,
        op: &str,
        args: &[Value],
        kwargs: &Kwargs,
    ) -> Result<Value, Error> {
        unsafe {
            let (name, overload) = op.rsplit_once('.').unwrap_or((op, ""));
            let overload = if overload == "default" { "" } else { overload };
            boxed_aten::call(&self.native, name, overload, args, kwargs)
        }
    }
}

pub(crate) fn required<'a>(
    args: &'a [Value],
    kwargs: &'a Kwargs,
    index: usize,
    name: &str,
) -> Result<&'a Value, Error> {
    get(args, kwargs, index, name)
        .ok_or_else(|| type_error(format!("missing required argument '{name}'")))
}
