//! Kitchen's backend-selection policy, independently of kernel arithmetic.
use crate::native::ShellDType;
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    marker::PhantomData,
    rc::Rc,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};
use tch::Kind;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Failure {
    pub parameter: String,
    pub reason: String,
    /// Original rule raised instead of returning ValidationResult.fail.
    pub raised: Option<String>,
}
impl Failure {
    pub fn new(parameter: &str, reason: impl Into<String>) -> Self {
        Self {
            parameter: parameter.into(),
            reason: reason.into(),
            raised: None,
        }
    }
    pub fn raised(class: &str, reason: impl Into<String>) -> Self {
        Self {
            parameter: String::new(),
            reason: reason.into(),
            raised: Some(class.into()),
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SelectionError {
    CallRuleRaised(Failure),
    BackendNotFound {
        backend: String,
        reason: String,
    },
    BackendNotImplemented {
        backend: String,
        function: String,
    },
    NoCapableBackend {
        function: String,
        // Python dict order is the first insertion order, even after updates.
        failures: Vec<(String, Failure)>,
    },
}
impl SelectionError {
    pub fn exception_class(&self) -> &str {
        match self {
            Self::CallRuleRaised(error) => error
                .raised
                .as_deref()
                .expect("CallRuleRaised must carry its original exception class"),
            Self::BackendNotFound { .. } => "BackendNotFoundError",
            Self::BackendNotImplemented { .. } => "BackendNotImplementedError",
            Self::NoCapableBackend { .. } => "NoCapableBackendError",
        }
    }
}
impl std::fmt::Display for SelectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CallRuleRaised(error) => f.write_str(&error.reason),
            Self::BackendNotFound { backend, reason } => {
                write!(f, "Backend '{backend}' is not available")?;
                if !reason.is_empty() {
                    write!(f, ": {reason}")?;
                }
                Ok(())
            }
            Self::BackendNotImplemented { backend, function } => {
                write!(f, "Backend '{backend}' does not implement '{function}'")
            }
            Self::NoCapableBackend { function, failures } => {
                if failures.is_empty() {
                    return write!(f, "No backend available for '{function}'");
                }
                write!(f, "No backend can handle '{function}': ")?;
                for (i, (backend, error)) in failures.iter().enumerate() {
                    if i > 0 {
                        f.write_str("; ")?;
                    }
                    write!(f, "{backend}: ")?;
                    if !error.parameter.is_empty() {
                        write!(f, "{}: ", error.parameter)?;
                    }
                    f.write_str(&error.reason)?;
                }
                Ok(())
            }
        }
    }
}
impl std::error::Error for SelectionError {}

fn record_failure(failures: &mut Vec<(String, Failure)>, backend: String, failure: Failure) {
    if let Some((_, previous)) = failures.iter_mut().find(|(name, _)| *name == backend) {
        *previous = failure;
    } else {
        failures.push((backend, failure));
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScalarType {
    Int,
    Float,
    Bool,
    Str,
    NoneType,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AllowedType {
    DType(Kind),
    ShellDType(ShellDType),
    Scalar(ScalarType),
}
impl AllowedType {
    pub(crate) fn python_name(&self) -> &'static str {
        match self {
            Self::DType(kind) => match kind {
                Kind::Uint8 => "torch.uint8",
                Kind::Int8 => "torch.int8",
                Kind::Int16 => "torch.int16",
                Kind::Int => "torch.int32",
                Kind::Int64 => "torch.int64",
                Kind::Half => "torch.float16",
                Kind::Float => "torch.float32",
                Kind::Double => "torch.float64",
                Kind::ComplexHalf => "torch.complex32",
                Kind::ComplexFloat => "torch.complex64",
                Kind::ComplexDouble => "torch.complex128",
                Kind::Bool => "torch.bool",
                Kind::QInt8 => "torch.qint8",
                Kind::QUInt8 => "torch.quint8",
                Kind::QInt32 => "torch.qint32",
                Kind::BFloat16 => "torch.bfloat16",
                Kind::Float8e5m2 => "torch.float8_e5m2",
                Kind::Float8e4m3fn => "torch.float8_e4m3fn",
                Kind::Float8e5m2fnuz => "torch.float8_e5m2fnuz",
                Kind::Float8e4m3fnuz => "torch.float8_e4m3fnuz",
            },
            Self::ShellDType(ShellDType::E8M0) => "torch.float8_e8m0fnu",
            Self::ShellDType(ShellDType::PackedE2M1) => "torch.float4_e2m1fn_x2",
            Self::Scalar(kind) => match kind {
                ScalarType::Int => "<class 'int'>",
                ScalarType::Float => "<class 'float'>",
                ScalarType::Bool => "<class 'bool'>",
                ScalarType::Str => "<class 'str'>",
                ScalarType::NoneType => "<class 'NoneType'>",
            },
        }
    }
}

pub(crate) fn python_float(value: f64) -> String {
    if value.is_nan() {
        return "nan".into();
    }
    // Debug retains the .0 and signed zero of a float. Python also spells
    // exponents with a sign and at least two digits.
    let text = crate::python_float_repr::shortest(value);
    if let Some((mantissa, exponent)) = text.split_once('e') {
        let exponent: i32 = exponent.parse().unwrap();
        format!("{mantissa}e{exponent:+03}")
    } else {
        text
    }
}
#[derive(Clone, Debug)]
pub enum Argument<'rt> {
    Source(crate::registry_source::SourceArgument<'rt>),
    None,
    Tensor {
        kind: Kind,
        device: String,
        shape: Vec<i64>,
    },
    DType(Kind),
    ShellTensor {
        kind: ShellDType,
        device: String,
        shape: Vec<i64>,
    },
    ShellDType(ShellDType),
    Scalar(ScalarType),
    Integer(i64),
    Unsigned(u64),
    FloatValue(f64),
    BoolValue(bool),
    Text(String),
    Sequence(Vec<i64>),
    BoolSequence(Vec<bool>),
}
impl Argument<'_> {
    pub fn integer_sequence(&self) -> Option<Vec<i64>> {
        match self {
            Self::Source(v) => v.integers(),
            Self::Sequence(v) => Some(v.clone()),
            Self::BoolSequence(v) => Some(v.iter().map(|b| i64::from(*b)).collect()),
            _ => None,
        }
    }
    pub fn sequence_repr(&self) -> Option<String> {
        match self {
            Self::Source(v) if v.integers().is_some() => v.list_repr().ok(),
            Self::Sequence(v) => Some(format!("{v:?}")),
            Self::BoolSequence(v) => Some(format!(
                "[{}]",
                v.iter()
                    .map(|b| if *b { "True" } else { "False" })
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
            _ => None,
        }
    }
    pub(crate) fn dtype_failure_value(&self) -> Result<String, Failure> {
        Ok(match self {
            Self::Source(v) => return v.display(),
            Self::Tensor { kind, .. } | Self::DType(kind) => {
                AllowedType::DType(*kind).python_name().into()
            }
            Self::ShellTensor { kind, .. } | Self::ShellDType(kind) => {
                AllowedType::ShellDType(*kind).python_name().into()
            }
            Self::Scalar(kind) => AllowedType::Scalar(kind.clone()).python_name().into(),
            Self::Integer(value) => value.to_string(),
            Self::Unsigned(value) => value.to_string(),
            Self::FloatValue(value) => python_float(*value),
            Self::BoolValue(value) => if *value { "True" } else { "False" }.into(),
            Self::Text(value) => value.clone(),
            Self::None | Self::Sequence(_) | Self::BoolSequence(_) => {
                unreachable!("checked before dtype failure")
            }
        })
    }
    pub fn tensor_parts(&self) -> Option<(AllowedType, &str, &[i64])> {
        match self {
            Self::Source(v) => v.metadata.as_deref().and_then(Self::tensor_parts),
            Self::Tensor {
                kind,
                device,
                shape,
            } => Some((AllowedType::DType(*kind), device, shape)),
            Self::ShellTensor {
                kind,
                device,
                shape,
            } => Some((AllowedType::ShellDType(*kind), device, shape)),
            _ => None,
        }
    }
    pub fn allowed_type(&self) -> Option<AllowedType> {
        Some(match self {
            Self::Source(v) => return v.metadata.as_deref().and_then(Self::allowed_type),
            Self::None => return None,
            Self::Tensor { kind, .. } | Self::DType(kind) => AllowedType::DType(*kind),
            Self::ShellTensor { kind, .. } | Self::ShellDType(kind) => {
                AllowedType::ShellDType(*kind)
            }
            Self::Scalar(kind) => AllowedType::Scalar(kind.clone()),
            Self::Integer(_) | Self::Unsigned(_) => AllowedType::Scalar(ScalarType::Int),
            Self::FloatValue(_) => AllowedType::Scalar(ScalarType::Float),
            Self::BoolValue(_) => AllowedType::Scalar(ScalarType::Bool),
            Self::Text(_) => AllowedType::Scalar(ScalarType::Str),
            Self::Sequence(_) | Self::BoolSequence(_) => return None,
        })
    }
}
pub type CallArgs<'rt> = BTreeMap<String, Argument<'rt>>;
#[derive(Clone, Debug)]
pub enum ShapeRule {
    ExactDims(usize),
    MinDims(usize),
    DivisibleBy { dim: usize, factor: i64 },
}
#[derive(Clone, Default)]
pub struct ParamConstraint {
    pub types: Option<Vec<AllowedType>>,
    pub devices: Option<BTreeSet<String>>,
    pub shape_rules: Vec<ShapeRule>,
}
pub type CallRule = Arc<dyn for<'rt> Fn(&CallArgs<'rt>) -> Result<(), Failure> + Send + Sync>;
#[derive(Clone)]
pub struct FunctionConstraints {
    // Vec retains the reference dictionary's validation order.
    pub params: Vec<(String, ParamConstraint)>,
    pub default_devices: BTreeSet<String>,
    pub min_capability: Option<(u32, u32)>,
    pub call_rules: Vec<CallRule>,
}
impl Default for FunctionConstraints {
    fn default() -> Self {
        Self {
            params: Vec::new(),
            default_devices: BTreeSet::from(["cuda".into(), "cpu".into()]),
            min_capability: None,
            call_rules: Vec::new(),
        }
    }
}
impl FunctionConstraints {
    pub fn validate(&self, args: &CallArgs, capability: Option<(u32, u32)>) -> Result<(), Failure> {
        if let Some(minimum) = self.min_capability {
            match capability {
                None => {
                    return Err(Failure::new(
                        "__hardware__",
                        "CUDA compute capability required but not available",
                    ));
                }
                Some(current) if current < minimum => {
                    return Err(Failure::new(
                        "__hardware__",
                        format!(
                            "compute capability {}.{} < required {}.{}",
                            current.0, current.1, minimum.0, minimum.1
                        ),
                    ));
                }
                _ => {}
            }
        }
        for (name, constraint) in &self.params {
            let Some(value) = args.get(name) else {
                continue;
            };
            if matches!(value, Argument::None) {
                continue;
            }
            let dtype = value.allowed_type();
            if constraint.types.is_some()
                && let Argument::Source(v) = value
            {
                v.hashable()?;
            }
            if constraint.types.is_some()
                && matches!(value, Argument::Sequence(_) | Argument::BoolSequence(_))
            {
                return Err(Failure::raised("TypeError", "unhashable type: 'list'"));
            }
            if constraint
                .types
                .as_ref()
                .is_some_and(|types| dtype.as_ref().is_none_or(|kind| !types.contains(kind)))
            {
                // D-05-REGISTRY-TYPE-SET-ORDER: only this type-set literal has
                // an approved deterministic ordering; retain all surrounding text.
                let names: BTreeSet<_> = constraint
                    .types
                    .as_ref()
                    .unwrap()
                    .iter()
                    .map(AllowedType::python_name)
                    .collect();
                let allowed = if names.is_empty() {
                    "set()".into()
                } else {
                    format!("{{{}}}", names.into_iter().collect::<Vec<_>>().join(", "))
                };
                return Err(Failure::new(
                    name,
                    format!("dtype {} not in {allowed}", value.dtype_failure_value()?),
                ));
            }
            if let Some((_, device, shape)) = value.tensor_parts() {
                let devices = constraint.devices.as_ref().unwrap_or(&self.default_devices);
                let device_type = device.split(':').next().unwrap();
                if !devices.contains("*") && !devices.contains(device_type) {
                    // D-05-REGISTRY-DEVICE-SET-ORDER: only the allowed-device
                    // literal has an approved fixed order. Device names in the
                    // frozen tables are plain ASCII identifiers.
                    let allowed = if devices.is_empty() {
                        "set()".into()
                    } else {
                        format!(
                            "{{{}}}",
                            devices
                                .iter()
                                .map(|name| format!("'{name}'"))
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    };
                    return Err(Failure::new(
                        name,
                        format!("device {device_type} not in {allowed}"),
                    ));
                }
                for rule in &constraint.shape_rules {
                    let valid = match rule {
                        ShapeRule::ExactDims(n) => shape.len() == *n,
                        ShapeRule::MinDims(n) => shape.len() >= *n,
                        ShapeRule::DivisibleBy { dim, factor } => {
                            shape.get(*dim).is_some_and(|size| size % factor == 0)
                        }
                    };
                    if !valid {
                        let description = match rule {
                            ShapeRule::ExactDims(n) => format!("exactly {n}D"),
                            ShapeRule::MinDims(n) => format!("at least {n}D"),
                            ShapeRule::DivisibleBy { dim, factor } => {
                                format!("dim[{dim}] divisible by {factor}")
                            }
                        };
                        return Err(Failure::new(
                            name,
                            format!("shape {shape:?} fails: {description}"),
                        ));
                    }
                }
            }
        }
        for rule in &self.call_rules {
            rule(args)?;
        }
        Ok(())
    }
}
#[derive(Clone)]
struct State {
    priority: Vec<String>,
    backends: BTreeMap<String, BTreeMap<String, FunctionConstraints>>,
    constraints: BTreeMap<(String, String), FunctionConstraints>,
    disabled: BTreeSet<String>,
    unavailable: BTreeMap<String, String>,
}
impl State {
    fn available(&self, name: &str) -> bool {
        self.backends.contains_key(name) && !self.disabled.contains(name)
    }
}
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
thread_local! {static OVERRIDES:RefCell<BTreeMap<u64,String>>=const {RefCell::new(BTreeMap::new())};}

#[derive(Clone)]
pub struct BackendRegistry {
    id: u64,
    state: Arc<Mutex<State>>,
    capability: Arc<OnceLock<Option<(u32, u32)>>>,
    capability_probe: Arc<dyn Fn() -> Option<(u32, u32)> + Send + Sync>,
}
pub struct BackendStatus {
    pub available: bool,
    pub disabled: bool,
    pub unavailable_reason: Option<String>,
    pub capabilities: Vec<String>,
}
pub struct BackendOverride {
    id: u64,
    previous: Option<String>,
    _same_thread: PhantomData<Rc<()>>,
}
impl Drop for BackendOverride {
    fn drop(&mut self) {
        OVERRIDES.with(|overrides| {
            let mut map = overrides.borrow_mut();
            if let Some(previous) = self.previous.take() {
                map.insert(self.id, previous);
            } else {
                map.remove(&self.id);
            }
        });
    }
}
impl BackendRegistry {
    pub fn new(capability: Option<(u32, u32)>) -> Self {
        let registry = Self::with_capability_probe(|| None);
        registry.capability.set(capability).unwrap();
        registry
    }
    /// Like cached_property: probe the current device on first validation only.
    pub fn with_capability_probe(
        probe: impl Fn() -> Option<(u32, u32)> + Send + Sync + 'static,
    ) -> Self {
        Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            capability: Arc::new(OnceLock::new()),
            capability_probe: Arc::new(probe),
            state: Arc::new(Mutex::new(State {
                priority: ["ascend", "cuda", "triton", "eager"]
                    .map(str::to_owned)
                    .to_vec(),
                backends: BTreeMap::new(),
                constraints: BTreeMap::new(),
                disabled: BTreeSet::new(),
                unavailable: BTreeMap::new(),
            })),
        }
    }
    fn compute_capability(&self) -> Option<(u32, u32)> {
        *self.capability.get_or_init(|| (self.capability_probe)())
    }
    pub fn validate_backend_for_call(
        &self,
        backend: &str,
        function: &str,
        args: &CallArgs,
    ) -> Result<(), Failure> {
        let state = self.state.lock().unwrap().clone();
        if !state.available(backend) {
            return Err(Failure::new(
                "__backend__",
                state
                    .unavailable
                    .get(backend)
                    .map(String::as_str)
                    .unwrap_or("not available"),
            ));
        }
        let rules = state.backends[backend]
            .get(function)
            .ok_or_else(|| Failure::new("__function__", "not implemented"))?;
        rules.validate(args, self.compute_capability())
    }
    pub fn register(&self, name: &str, functions: BTreeMap<String, FunctionConstraints>) {
        let mut state = self.state.lock().unwrap();
        for (function, rules) in &functions {
            state
                .constraints
                .insert((name.into(), function.clone()), rules.clone());
        }
        state.backends.insert(name.into(), functions);
        state.unavailable.remove(name);
    }
    pub fn is_available(&self, name: &str) -> bool {
        self.state.lock().unwrap().available(name)
    }
    pub(crate) fn current_override(&self) -> Option<String> {
        OVERRIDES.with(|overrides| overrides.borrow().get(&self.id).cloned())
    }
    pub fn get_constraints(&self, backend: &str, function: &str) -> Option<FunctionConstraints> {
        self.state
            .lock()
            .unwrap()
            .constraints
            .get(&(backend.into(), function.into()))
            .cloned()
    }
    pub fn list_backends(&self) -> BTreeMap<String, BackendStatus> {
        let state = self.state.lock().unwrap();
        let names: BTreeSet<_> = state
            .priority
            .iter()
            .chain(state.backends.keys())
            .chain(state.unavailable.keys())
            .cloned()
            .collect();
        names
            .into_iter()
            .map(|name| {
                let status = BackendStatus {
                    available: state.backends.contains_key(&name),
                    disabled: state.disabled.contains(&name),
                    unavailable_reason: state.unavailable.get(&name).cloned(),
                    capabilities: state
                        .backends
                        .get(&name)
                        .map(|functions| functions.keys().cloned().collect())
                        .unwrap_or_default(),
                };
                (name, status)
            })
            .collect()
    }
    pub fn set_priority(&self, names: Vec<String>) {
        self.state.lock().unwrap().priority = names;
    }
    pub fn mark_unavailable(&self, name: &str, reason: &str) {
        self.state
            .lock()
            .unwrap()
            .unavailable
            .insert(name.into(), reason.into());
    }
    pub fn disable(&self, name: &str) {
        self.state.lock().unwrap().disabled.insert(name.into());
    }
    pub fn enable(&self, name: &str) {
        self.state.lock().unwrap().disabled.remove(name);
    }
    pub fn use_backend(&self, name: &str) -> Result<BackendOverride, SelectionError> {
        let state = self.state.lock().unwrap();
        if !state.available(name) {
            let reason = state.unavailable.get(name).cloned().unwrap_or_else(|| {
                if state.disabled.contains(name) {
                    "disabled".into()
                } else {
                    "not registered".into()
                }
            });
            return Err(SelectionError::BackendNotFound {
                backend: name.into(),
                reason,
            });
        }
        let previous =
            OVERRIDES.with(|overrides| overrides.borrow_mut().insert(self.id, name.into()));
        Ok(BackendOverride {
            id: self.id,
            previous,
            _same_thread: PhantomData,
        })
    }
    pub fn select(
        &self,
        function: &str,
        explicit: Option<&str>,
        args: &CallArgs,
    ) -> Result<String, SelectionError> {
        // Run call rules outside the mutex, as the reference's rule callbacks
        // are allowed to consult registry state themselves.
        let state = self.state.lock().unwrap().clone();
        let check = |name: &str| -> Result<(), Failure> {
            if !state.available(name) {
                return Err(Failure::new("__backend__", "not available"));
            }
            let rules = state
                .backends
                .get(name)
                .and_then(|f| f.get(function))
                .ok_or_else(|| Failure::new("__function__", "not implemented"))?;
            if !args.is_empty() {
                rules.validate(args, self.compute_capability())?;
            }
            Ok(())
        };
        if let Some(backend) = explicit.filter(|name| !name.is_empty()) {
            if !state.backends.contains_key(backend) {
                let reason = state
                    .unavailable
                    .get(backend)
                    .cloned()
                    .unwrap_or_else(|| "not registered".into());
                return Err(SelectionError::BackendNotFound {
                    backend: backend.into(),
                    reason,
                });
            }
            if state.disabled.contains(backend) {
                return Err(SelectionError::BackendNotFound {
                    backend: backend.into(),
                    reason: "disabled".into(),
                });
            }
            if !state.backends[backend].contains_key(function) {
                return Err(SelectionError::BackendNotImplemented {
                    backend: backend.into(),
                    function: function.into(),
                });
            }
            return match check(backend) {
                Ok(()) => Ok(backend.into()),
                Err(failure) if failure.raised.is_some() => {
                    Err(SelectionError::CallRuleRaised(failure))
                }
                Err(failure) => Err(SelectionError::NoCapableBackend {
                    function: function.into(),
                    failures: vec![(backend.into(), failure)],
                }),
            };
        }
        let mut failures = Vec::new();
        if let Some(backend) = OVERRIDES.with(|overrides| overrides.borrow().get(&self.id).cloned())
        {
            let result = if !state.available(&backend) {
                Err(Failure::new("", "not available"))
            } else if !state.backends[&backend].contains_key(function) {
                Err(Failure::new("", "not implemented"))
            } else {
                check(&backend)
            };
            match result {
                Ok(()) => return Ok(backend),
                Err(failure) if failure.raised.is_some() => {
                    return Err(SelectionError::CallRuleRaised(failure));
                }
                Err(failure) => {
                    record_failure(&mut failures, backend, failure);
                }
            }
        }
        for backend in &state.priority {
            if !state.available(backend) || !state.backends[backend].contains_key(function) {
                continue;
            }
            match check(backend) {
                Ok(()) => return Ok(backend.clone()),
                Err(failure) if failure.raised.is_some() => {
                    return Err(SelectionError::CallRuleRaised(failure));
                }
                Err(failure) => {
                    record_failure(&mut failures, backend.clone(), failure);
                }
            }
        }
        Err(SelectionError::NoCapableBackend {
            function: function.into(),
            failures,
        })
    }
}
