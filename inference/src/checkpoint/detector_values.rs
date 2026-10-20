//! Value operations for the statically generated, locked checkpoint detector.
//! This is internal metadata handling, not a runtime source/bytecode evaluator.
use super::detection;
use crate::{
    execution::Runtime, execution_value::Value as V, native::Error, patching::options::Options,
};
use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

pub(super) fn none() -> Options {
    Options::Atom(V::None)
}
pub(super) fn boolean(v: bool) -> Options {
    Options::Atom(V::Bool(v))
}
pub(super) fn float(v: f64) -> Options {
    Options::Atom(V::Float(v))
}
pub(super) fn text(v: &str) -> Options {
    Options::Atom(V::Text(v.into()))
}
pub(super) fn list(v: Vec<Options>) -> Options {
    Options::List(Rc::new(RefCell::new(v)))
}
pub(super) fn tuple(v: Vec<Options>) -> Options {
    Options::Tuple(Rc::new(v))
}
pub(super) fn integer(v: i128) -> Result<Options, Error> {
    if let Ok(v) = i64::try_from(v) {
        return Ok(Options::Atom(V::Int(v)));
    }
    if let Ok(v) = u64::try_from(v) {
        return Ok(Options::Atom(V::UInt(v)));
    }
    Err(gap(
        "detector integer exceeds the admitted native value range",
    ))
}
fn error(class: &'static str, message: impl Into<String>) -> Error {
    Error::ReferenceException {
        class,
        message: message.into(),
    }
}
fn gap(message: impl Into<String>) -> Error {
    Error::BindingGap {
        backend: "model_detection".into(),
        operation: message.into(),
    }
}
struct KeyMap(RefCell<Vec<(Options, Options)>>);
// The frozen detector immediately calls int() on this power result. Retain
// its Python type instead of substituting a floating NaN for a complex value.
struct ComplexPower;
#[derive(Clone)]
pub(super) struct Context<'a> {
    runtime: &'a Runtime,
    locals: BTreeMap<String, Options>,
}
impl<'a> Context<'a> {
    pub fn new(runtime: &'a Runtime, state: Options, prefix: &str, metadata: Options) -> Self {
        Self {
            runtime,
            locals: BTreeMap::from([
                ("state_dict".into(), state),
                ("key_prefix".into(), text(prefix)),
                ("metadata".into(), metadata),
            ]),
        }
    }
    pub fn get(&self, name: &str) -> Result<Options, Error> {
        self.locals.get(name).cloned().or_else(||(name=="torch").then(||text("__locked_torch_module__"))).ok_or_else(||error("UnboundLocalError",format!("cannot access local variable '{name}' where it is not associated with a value")))
    }
    pub fn set(&mut self, name: &str, value: Options) {
        self.locals.insert(name.into(), value);
    }
    pub fn truth(&self, value: &Options) -> Result<bool, Error> {
        if let Some(value) = value.metadata_truth() {
            return value;
        }
        Ok(match value {
            Options::Dict(v) => !v.borrow().is_empty(),
            Options::List(v) => !v.borrow().is_empty(),
            Options::Tuple(v) => !v.is_empty(),
            Options::Object(v) => v
                .downcast_ref::<KeyMap>()
                .is_none_or(|v| !v.0.borrow().is_empty()),
            Options::Atom(V::None) => false,
            Options::Atom(V::Bool(v)) => *v,
            Options::Atom(V::Int(v)) => *v != 0,
            Options::Atom(V::UInt(v)) => *v != 0,
            Options::Atom(V::Float(v)) => *v != 0.,
            Options::Atom(V::Text(v)) => !v.is_empty(),
            Options::Atom(V::Tensor(_) | V::Quantized(_)) => crate::conditioning::is_equal(
                &unsafe {
                    self.runtime
                        .op("aten::is_nonzero.default", &[value.snapshot()?], &vec![])
                }?,
                &V::Bool(true),
            )?,
            _ => true,
        })
    }
    pub fn iter(&self, value: &Options) -> Result<Vec<Options>, Error> {
        match value {
            Options::List(v) => Ok(v.borrow().clone()),
            Options::Tuple(v) => Ok(v.as_ref().clone()),
            Options::Dict(v) => Ok(v.borrow().iter().map(|(k, _)| text(k)).collect()),
            Options::Atom(V::Text(v)) => Ok(v.chars().map(|c| text(&c.to_string())).collect()),
            Options::Object(v) => v
                .downcast_ref::<KeyMap>()
                .map(|v| v.0.borrow().iter().map(|(k, _)| k.clone()).collect())
                .ok_or_else(|| gap("unknown detector iterable object")),
            _ => Err(error(
                "TypeError",
                format!("'{}' object is not iterable", value.typename()),
            )),
        }
    }
    pub fn mapping(&self, values: Vec<(Options, Options)>) -> Result<Options, Error> {
        if values
            .iter()
            .all(|(key, _)| matches!(key, Options::Atom(V::Text(_))))
        {
            let out = Options::dictionary(vec![]);
            for (key, value) in values {
                let Options::Atom(V::Text(key)) = key else {
                    unreachable!()
                };
                out.set(&key, value)?;
            }
            Ok(out)
        } else {
            Ok(Options::Object(Rc::new(KeyMap(RefCell::new(values)))))
        }
    }
    pub fn index(&self, value: &Options, key: &Options) -> Result<Options, Error> {
        match value {
            Options::Dict(_) => {
                if let Options::Atom(V::Text(key)) = key {
                    return value.get(key)?.ok_or_else(|| {
                        error(
                            "KeyError",
                            crate::triton_specialization::python_str_repr(key),
                        )
                    });
                }
                Err(error(
                    "KeyError",
                    crate::python_value::repr(&key.snapshot()?)?,
                ))
            }
            Options::Object(object) => {
                if let Some(map) = object.downcast_ref::<KeyMap>() {
                    for (candidate, value) in map.0.borrow().iter() {
                        if self.equal(candidate, key)? {
                            return Ok(value.clone());
                        }
                    }
                    return Err(error(
                        "KeyError",
                        crate::python_value::repr(&key.snapshot()?)?,
                    ));
                }
                Err(gap("unknown detector mapping object"))
            }
            Options::List(_) | Options::Tuple(_) | Options::Atom(V::Text(_)) => {
                let values = self.iter(value)?;
                let index = self.int(key)?;
                let index = if index < 0 {
                    index + values.len() as i128
                } else {
                    index
                };
                values
                    .get(usize::try_from(index).unwrap_or(usize::MAX))
                    .cloned()
                    .ok_or_else(|| {
                        error(
                            "IndexError",
                            if matches!(value, Options::List(_)) {
                                "list index out of range"
                            } else if matches!(value, Options::Tuple(_)) {
                                "tuple index out of range"
                            } else {
                                "string index out of range"
                            },
                        )
                    })
            }
            _ => Err(error(
                "TypeError",
                format!("'{}' object is not subscriptable", value.typename()),
            )),
        }
    }
    pub fn set_item(&self, value: &Options, key: &Options, item: Options) -> Result<(), Error> {
        if let Options::Atom(V::Text(key)) = key
            && matches!(value, Options::Dict(_))
        {
            return value.set(key, item);
        }
        if let Options::List(values) = value {
            let n = self.int(key)?;
            let mut values = values.borrow_mut();
            let n = if n < 0 { n + values.len() as i128 } else { n };
            let slot = values
                .get_mut(usize::try_from(n).unwrap_or(usize::MAX))
                .ok_or_else(|| error("IndexError", "list assignment index out of range"))?;
            *slot = item;
            return Ok(());
        }
        Err(gap(
            "detector assignment outside string-key dictionary or native list",
        ))
    }
    pub fn slice(
        &self,
        value: &Options,
        start: Option<Options>,
        end: Option<Options>,
        step: Option<Options>,
    ) -> Result<Options, Error> {
        match value {
            Options::List(_) | Options::Tuple(_) | Options::Atom(V::Text(_)) => {}
            Options::Dict(_) => return Err(error("KeyError", "slice(None, None, None)")),
            Options::Object(_) | Options::Atom(V::Quantized(_)) => {
                return Err(gap(
                    "source metadata slice requires original subscription binding",
                ));
            }
            Options::Atom(V::Tensor(_)) => {
                return Err(gap(
                    "source tensor slice requires original indexing binding",
                ));
            }
            _ => {
                return Err(error(
                    "TypeError",
                    format!("'{}' object is not subscriptable", value.typename()),
                ));
            }
        }
        let values = self.iter(value)?;
        let n = values.len() as i128;
        let step = step.as_ref().map(|v| self.int(v)).transpose()?.unwrap_or(1);
        if step == 0 {
            return Err(error("ValueError", "slice step cannot be zero"));
        }
        let bound = |value: Option<&Options>, default: i128| -> Result<i128, Error> {
            Ok(if let Some(value) = value {
                let i = self.int(value)?;
                let i = if i < 0 { i + n } else { i };
                if step > 0 {
                    i.clamp(0, n)
                } else {
                    i.clamp(-1, n - 1)
                }
            } else {
                default
            })
        };
        let mut index = bound(start.as_ref(), if step > 0 { 0 } else { n - 1 })?;
        let stop = bound(end.as_ref(), if step > 0 { n } else { -1 })?;
        let mut out = vec![];
        while if step > 0 { index < stop } else { index > stop } {
            out.push(values[index as usize].clone());
            index += step;
        }
        if matches!(value, Options::Atom(V::Text(_))) {
            return self.concatenate_text(&out);
        }
        Ok(if matches!(value, Options::Tuple(_)) {
            if start.is_none() && end.is_none() && step == 1 {
                value.clone()
            } else {
                tuple(out)
            }
        } else {
            list(out)
        })
    }
    pub fn string(&self, value: &Options) -> Result<Options, Error> {
        Ok(text(&crate::python_value::display(&value.snapshot()?)?))
    }
    pub fn concatenate_text(&self, values: &[Options]) -> Result<Options, Error> {
        let mut out = String::new();
        for value in values {
            let Options::Atom(V::Text(value)) = value else {
                return Err(gap("detector text concatenation operand"));
            };
            out.push_str(value);
        }
        Ok(text(&out))
    }
    fn int(&self, value: &Options) -> Result<i128, Error> {
        if let Options::Object(value) = value
            && value.is::<ComplexPower>()
        {
            return Err(error(
                "TypeError",
                "int() argument must be a string, a bytes-like object or a real number, not 'complex'",
            ));
        }
        crate::python_numbers::integer_value(self.runtime, &value.snapshot()?)
    }
    fn index_integer(&self, value: &Options) -> Result<i128, Error> {
        match value {
            Options::Atom(V::Int(v)) => Ok(i128::from(*v)),
            Options::Atom(V::UInt(v)) => Ok(i128::from(*v)),
            Options::Atom(V::Bool(v)) => Ok(i128::from(*v)),
            Options::Atom(V::Tensor(v)) => {
                {
                    let guard = v.lock().unwrap();
                    let tensor = &guard.tensor;
                    if tensor.numel() != 1
                        || !matches!(
                            tensor.kind(),
                            tch::Kind::Bool
                                | tch::Kind::Uint8
                                | tch::Kind::Int8
                                | tch::Kind::Int16
                                | tch::Kind::Int
                                | tch::Kind::Int64
                        )
                    {
                        return Err(error(
                            "TypeError",
                            "only integer tensors of a single element can be converted to an index",
                        ));
                    }
                }
                self.int(value)
            }
            Options::Object(_) | Options::Atom(V::Quantized(_)) => Err(gap(
                "metadata __index__ requires original native object binding",
            )),
            _ => Err(error(
                "TypeError",
                format!(
                    "'{}' object cannot be interpreted as an integer",
                    value.typename()
                ),
            )),
        }
    }
    fn number(&self, value: &Options) -> Result<f64, Error> {
        match value {
            Options::Atom(V::Int(v)) => Ok(*v as f64),
            Options::Atom(V::UInt(v)) => Ok(*v as f64),
            Options::Atom(V::Bool(v)) => Ok(f64::from(u8::from(*v))),
            Options::Atom(V::Float(v)) => Ok(*v),
            _ => Err(gap("non-numeric detector arithmetic")),
        }
    }
    pub fn binary(&self, op: &str, left: &Options, right: &Options) -> Result<Options, Error> {
        if op == "Div"
            && !matches!(left, Options::Atom(V::Float(_)))
            && !matches!(right, Options::Atom(V::Float(_)))
        {
            return Ok(float(crate::python_numbers::integer_ratio(
                self.int(left)?,
                self.int(right)?,
            )?));
        }
        if op == "Add" {
            if let (Options::Atom(V::Text(a)), Options::Atom(V::Text(b))) = (left, right) {
                return Ok(text(&(a.clone() + b)));
            }
            if matches!(
                (left, right),
                (Options::List(_), Options::List(_)) | (Options::Tuple(_), Options::Tuple(_))
            ) {
                let mut values = self.iter(left)?;
                values.extend(self.iter(right)?);
                return Ok(if matches!(left, Options::Tuple(_)) {
                    tuple(values)
                } else {
                    list(values)
                });
            }
        }
        if op == "Mult" {
            let (sequence, count) = if matches!(
                left,
                Options::List(_) | Options::Tuple(_) | Options::Atom(V::Text(_))
            ) {
                (left, right)
            } else {
                (right, left)
            };
            if matches!(
                sequence,
                Options::List(_) | Options::Tuple(_) | Options::Atom(V::Text(_))
            ) {
                let count = usize::try_from(self.int(count)?.max(0))
                    .map_err(|_| gap("detector repetition exceeds usize"))?;
                let input = self.iter(sequence)?;
                let mut values = vec![];
                for _ in 0..count {
                    values.extend(input.iter().cloned());
                }
                return Ok(if matches!(sequence, Options::List(_)) {
                    list(values)
                } else if matches!(sequence, Options::Tuple(_)) {
                    tuple(values)
                } else {
                    return self.concatenate_text(&values);
                });
            }
        }
        if matches!(left, Options::Atom(V::Float(_)))
            || matches!(right, Options::Atom(V::Float(_)))
            || op == "Div"
        {
            let a = self.number(left)?;
            let b = self.number(right)?;
            if op == "Pow" && a < 0. && b.fract() != 0. {
                return Ok(Options::Object(Rc::new(ComplexPower)));
            }
            if matches!(op, "Div" | "FloorDiv") && b == 0. {
                return Err(error(
                    "ZeroDivisionError",
                    if op == "Div" {
                        "float division by zero"
                    } else {
                        "float floor division by zero"
                    },
                ));
            }
            return Ok(float(match op {
                "Add" => a + b,
                "Sub" => a - b,
                "Mult" => a * b,
                "Div" => a / b,
                "FloorDiv" => (a / b).floor(),
                "Pow" => a.powf(b),
                _ => return Err(gap(format!("detector float operation {op}"))),
            }));
        }
        let a = self.int(left)?;
        let b = self.int(right)?;
        let result = match op {
            "Add" => a.checked_add(b),
            "Sub" => a.checked_sub(b),
            "Mult" => a.checked_mul(b),
            "FloorDiv" => {
                if b == 0 {
                    return Err(error(
                        "ZeroDivisionError",
                        "integer division or modulo by zero",
                    ));
                }
                a.checked_div(b).map(|n| {
                    if (a % b != 0) && ((a < 0) != (b < 0)) {
                        n - 1
                    } else {
                        n
                    }
                })
            }
            "Pow" => u32::try_from(b).ok().and_then(|b| a.checked_pow(b)),
            _ => return Err(gap(format!("detector integer operation {op}"))),
        };
        integer(result.ok_or_else(|| gap("detector integer arithmetic beyond i128"))?)
    }
    pub fn unary(&self, op: &str, value: &Options) -> Result<Options, Error> {
        match op {
            "Not" => Ok(boolean(!self.truth(value)?)),
            "USub" => {
                if matches!(value, Options::Atom(V::Float(_))) {
                    Ok(float(-self.number(value)?))
                } else {
                    integer(-self.int(value)?)
                }
            }
            _ => Err(gap(format!("detector unary operation {op}"))),
        }
    }
    fn equal(&self, a: &Options, b: &Options) -> Result<bool, Error> {
        if a.same_object(b) {
            return Ok(true);
        }
        crate::conditioning::is_equal(&a.snapshot()?, &b.snapshot()?)
    }
    pub fn compare(&self, op: &str, left: &Options, right: &Options) -> Result<Options, Error> {
        let result = match op {
            "Eq" => self.equal(left, right)?,
            "NotEq" => !self.equal(left, right)?,
            "Is" | "IsNot" => {
                let same = matches!(
                    (left, right),
                    (Options::Atom(V::None), Options::Atom(V::None))
                ) || left.same_object(right);
                if op == "Is" { same } else { !same }
            }
            "In" | "NotIn" => {
                let mut found = false;
                for value in self.iter(right)? {
                    if self.equal(left, &value)? {
                        found = true;
                        break;
                    }
                }
                if op == "In" { found } else { !found }
            }
            "Lt" | "Gt" | "GtE" | "LtE" => {
                let a = self.number(left)?;
                let b = self.number(right)?;
                match op {
                    "Lt" => a < b,
                    "Gt" => a > b,
                    "GtE" => a >= b,
                    _ => a <= b,
                }
            }
            _ => return Err(gap(format!("detector comparison {op}"))),
        };
        Ok(boolean(result))
    }
    pub fn attribute(&self, value: &Options, name: &str) -> Result<Options, Error> {
        if name == "float32"
            && matches!(value,Options::Atom(V::Text(v))if v=="__locked_torch_module__")
        {
            return Ok(Options::Atom(V::DType(tch::Kind::Float)));
        }
        if matches!(value, Options::Atom(V::Tensor(_) | V::Quantized(_))) {
            let (shape, kind) = match value {
                Options::Atom(V::Tensor(v)) => {
                    let v = v.lock().unwrap();
                    (v.size(), v.kind())
                }
                Options::Atom(V::Quantized(v)) => {
                    let v = v.lock().unwrap();
                    (v.params.original_shape.clone(), v.params.original_kind)
                }
                _ => unreachable!(),
            };
            return match name {
                "shape" => Ok(tuple(
                    shape
                        .into_iter()
                        .map(|n| Options::Atom(V::Int(n)))
                        .collect(),
                )),
                "ndim" => integer(shape.len() as i128),
                "dtype" => Ok(Options::Atom(V::DType(kind))),
                _ => Err(gap(format!("detector tensor attribute {name}"))),
            };
        }
        Err(error(
            "AttributeError",
            format!("'{}' object has no attribute '{name}'", value.typename()),
        ))
    }
    pub fn method(
        &self,
        value: &Options,
        name: &str,
        args: Vec<Options>,
        kwargs: Vec<(&str, Options)>,
    ) -> Result<Options, Error> {
        if !kwargs.is_empty() {
            return Err(gap(format!("detector keyword method {name}")));
        }
        match name {
            "pop" => {
                if let Options::List(values) = value {
                    let index = args
                        .first()
                        .map(|v| self.index_integer(v))
                        .transpose()?
                        .unwrap_or(-1);
                    let mut values = values.borrow_mut();
                    if values.is_empty() {
                        return Err(error("IndexError", "pop from empty list"));
                    }
                    let index = if index < 0 {
                        index + values.len() as i128
                    } else {
                        index
                    };
                    let index = usize::try_from(index)
                        .ok()
                        .filter(|&i| i < values.len())
                        .ok_or_else(|| error("IndexError", "pop index out of range"))?;
                    return Ok(values.remove(index));
                }
            }
            "get" => {
                if let Options::Dict(_) = value {
                    let Options::Atom(V::Text(key)) = &args[0] else {
                        return Err(gap("non-string metadata get key"));
                    };
                    return Ok(value
                        .get(key)?
                        .unwrap_or_else(|| args.get(1).cloned().unwrap_or_else(none)));
                }
                if let Options::Object(v) = value
                    && let Some(map) = v.downcast_ref::<KeyMap>()
                {
                    for (k, v) in map.0.borrow().iter() {
                        if self.equal(k, &args[0])? {
                            return Ok(v.clone());
                        }
                    }
                    return Ok(args.get(1).cloned().unwrap_or_else(none));
                }
            }
            "keys" => {
                if matches!(value, Options::Dict(_)) {
                    return Ok(list(self.iter(value)?));
                }
            }
            "append" => {
                if let Options::List(values) = value {
                    values.borrow_mut().push(args[0].clone());
                    return Ok(none());
                }
            }
            "update" => {
                if matches!(value, Options::Dict(_)) {
                    for (key, item) in args[0].dict_items()? {
                        value.set(&key, item)?;
                    }
                    return Ok(none());
                }
            }
            "format" => {
                if let Options::Atom(V::Text(template)) = value {
                    return self.format(template, &args);
                }
            }
            "startswith" | "endswith" => {
                if let (Options::Atom(V::Text(value)), Options::Atom(V::Text(prefix))) =
                    (value, &args[0])
                {
                    return Ok(boolean(if name == "startswith" {
                        value.starts_with(prefix)
                    } else {
                        value.ends_with(prefix)
                    }));
                }
            }
            "split" => {
                if let (Options::Atom(V::Text(value)), Options::Atom(V::Text(separator))) =
                    (value, &args[0])
                {
                    if separator.is_empty() {
                        return Err(error("ValueError", "empty separator"));
                    }
                    return Ok(list(value.split(separator).map(text).collect()));
                }
            }
            "size" => {
                if matches!(value, Options::Atom(V::Tensor(_) | V::Quantized(_))) {
                    let shape = self.attribute(value, "shape")?;
                    return if let Some(index) = args.first() {
                        self.index(&shape, index)
                    } else {
                        Ok(shape)
                    };
                }
            }
            "item" => {
                if let Options::Atom(V::Tensor(_)) = value {
                    return Ok(Options::Atom(unsafe {
                        self.runtime
                            .op("aten::item.default", &[value.snapshot()?], &vec![])
                    }?));
                }
            }
            _ => {}
        }
        Err(error(
            "AttributeError",
            format!("'{}' object has no attribute '{name}'", value.typename()),
        ))
    }
    fn format(&self, template: &str, args: &[Options]) -> Result<Options, Error> {
        let mut out = String::new();
        let mut chars = template.chars().peekable();
        let mut index = 0;
        while let Some(c) = chars.next() {
            if c == '{' {
                if chars.peek() == Some(&'{') {
                    chars.next();
                    out.push('{');
                    continue;
                }
                if chars.next() != Some('}') {
                    return Err(gap(
                        "detector format field beyond locked automatic placeholders",
                    ));
                }
                let value = args.get(index).ok_or_else(|| {
                    error(
                        "IndexError",
                        format!("Replacement index {index} out of range for positional args tuple"),
                    )
                })?;
                let Options::Atom(V::Text(value)) = self.string(value)? else {
                    unreachable!()
                };
                out.push_str(&value);
                index += 1;
            } else if c == '}' {
                if chars.next() != Some('}') {
                    return Err(error(
                        "ValueError",
                        "Single '}' encountered in format string",
                    ));
                }
                out.push('}');
            } else {
                out.push(c);
            }
        }
        Ok(text(&out))
    }
    pub fn function(
        &self,
        name: &str,
        args: Vec<Options>,
        kwargs: Vec<(&str, Options)>,
    ) -> Result<Options, Error> {
        match name {
            "sum" => self.collect("sum", self.iter(&args[0])?),
            "enumerate" => {
                let start = args
                    .get(1)
                    .map(|v| self.index_integer(v))
                    .transpose()?
                    .unwrap_or(0);
                Ok(list(
                    self.iter(&args[0])?
                        .into_iter()
                        .enumerate()
                        .map(|(i, v)| Ok(tuple(vec![integer(start + i as i128)?, v])))
                        .collect::<Result<Vec<_>, Error>>()?,
                ))
            }
            "reversed" => {
                let mut values = self.iter(&args[0])?;
                values.reverse();
                Ok(list(values))
            }
            "list" => Ok(list(self.iter(&args[0])?)),
            "tuple" => Ok(tuple(self.iter(&args[0])?)),
            "len" => match &args[0] {
                Options::List(v) => integer(v.borrow().len() as i128),
                Options::Tuple(v) => integer(v.len() as i128),
                Options::Dict(v) => integer(v.borrow().len() as i128),
                Options::Atom(V::Text(v)) => integer(v.chars().count() as i128),
                Options::Atom(V::Tensor(v)) => v
                    .lock()
                    .unwrap()
                    .tensor
                    .size()
                    .first()
                    .copied()
                    .map(|v| integer(i128::from(v)))
                    .unwrap_or_else(|| Err(error("TypeError", "len() of a 0-d tensor"))),
                Options::Object(v) if v.is::<KeyMap>() => {
                    integer(v.downcast_ref::<KeyMap>().unwrap().0.borrow().len() as i128)
                }
                Options::Object(_) | Options::Atom(V::Quantized(_)) => Err(gap(
                    "metadata length requires original native object binding",
                )),
                value => Err(error(
                    "TypeError",
                    format!("object of type '{}' has no len()", value.typename()),
                )),
            },
            "int" => integer(self.int(&args[0])?),
            "range" => {
                let (start, end) = if args.len() == 1 {
                    (0, self.index_integer(&args[0])?)
                } else {
                    (self.index_integer(&args[0])?, self.index_integer(&args[1])?)
                };
                Ok(list(
                    (start..end).map(integer).collect::<Result<Vec<_>, _>>()?,
                ))
            }
            "sorted" => {
                let mut values = self.iter(&args[0])?;
                let mut keyed = values
                    .drain(..)
                    .map(|v| {
                        let Options::Atom(V::Text(key)) = &v else {
                            return Err(gap("detector sort outside source string keys"));
                        };
                        Ok((key.clone(), v))
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                keyed.sort_by(|a, b| a.0.cmp(&b.0));
                Ok(list(keyed.into_iter().map(|(_, v)| v).collect()))
            }
            "math.prod" => {
                let mut product = integer(1)?;
                for value in self.iter(&args[0])? {
                    product = self.binary("Mult", &product, &value)?;
                }
                Ok(product)
            }
            "math.sqrt" => {
                let value = self.number(&args[0])?;
                if value < 0. {
                    return Err(error("ValueError", "math domain error"));
                }
                Ok(float(value.sqrt()))
            }
            "round" => {
                let value = self.number(&args[0])?;
                if !value.is_finite() {
                    return Err(error(
                        if value.is_nan() {
                            "ValueError"
                        } else {
                            "OverflowError"
                        },
                        if value.is_nan() {
                            "cannot convert float NaN to integer"
                        } else {
                            "cannot convert float infinity to integer"
                        },
                    ));
                }
                integer(value.round_ties_even() as i128)
            }
            "count_blocks" => {
                let keys = self.iter(&args[0])?;
                let Options::Atom(V::Text(template)) = &args[1] else {
                    return Err(gap("detector block format is not text"));
                };
                let mut count = 0_i128;
                loop {
                    let prefix = self.format(template, &[integer(count)?])?;
                    let Options::Atom(V::Text(prefix)) = prefix else {
                        unreachable!()
                    };
                    if !keys.iter().any(
                        |key| matches!(key,Options::Atom(V::Text(key))if key.starts_with(&prefix)),
                    ) {
                        break;
                    }
                    count += 1;
                }
                integer(count)
            }
            "convert_config" => detection::convert_config(&args[0]),
            "any_suffix_in" => {
                let keys = strings(&self.iter(&args[0])?)?;
                let prefix = string(&args[1])?;
                let main = string(&args[2])?;
                let tails = if let Some(tails) = args.get(3) {
                    strings(&self.iter(tails)?)?
                } else {
                    vec![]
                };
                Ok(boolean(detection::any_suffix_in(
                    &keys, &prefix, &main, &tails,
                )))
            }
            "calculate_transformer_depth" => {
                let prefix = string(&args[0])?;
                let keys = strings(&self.iter(&args[1])?)?;
                Ok(
                    detection::calculate_transformer_depth(&prefix, &keys, &args[2])?.map_or_else(
                        none,
                        |d| {
                            tuple(vec![
                                Options::Atom(V::UInt(d.depth as u64)),
                                Options::Atom(V::Int(d.context_dimension)),
                                boolean(d.linear),
                                boolean(d.time_stack),
                                boolean(d.time_stack_cross),
                            ])
                        },
                    ),
                )
            }
            "torch.std" => {
                let tensor = args[0].snapshot()?;
                let unbiased = kwargs
                    .iter()
                    .find(|(key, _)| *key == "unbiased")
                    .map_or(Ok(true), |(_, v)| self.truth(v))?;
                Ok(Options::Atom(unsafe {
                    self.runtime
                        .op("aten::std.default", &[tensor, V::Bool(unbiased)], &vec![])
                }?))
            }
            "json.loads" => parse_json(&args[0]),
            _ => Err(gap(format!("detector function {name}"))),
        }
    }
    pub fn collect(&self, kind: &str, values: Vec<Options>) -> Result<Options, Error> {
        match kind {
            "list" => Ok(list(values)),
            "sum" => {
                let mut total = integer(0)?;
                for value in values {
                    total = self.binary("Add", &total, &value)?;
                }
                Ok(total)
            }
            "unique_len" => {
                let mut unique: Vec<Options> = vec![];
                for value in values {
                    let mut found = false;
                    for old in &unique {
                        if self.equal(&value, old)? {
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        unique.push(value);
                    }
                }
                integer(unique.len() as i128)
            }
            _ => Err(gap(format!("detector collection {kind}"))),
        }
    }
    pub fn comprehension(
        &self,
        items: Vec<Options>,
        kind: &str,
        mut evaluate: impl FnMut(&mut Self, Options) -> Result<Option<Options>, Error>,
    ) -> Result<Options, Error> {
        let mut cx = self.clone();
        let mut values = vec![];
        for item in items {
            if let Some(value) = evaluate(&mut cx, item)? {
                if kind == "any" && self.truth(&value)? {
                    return Ok(boolean(true));
                }
                if kind == "all" && !self.truth(&value)? {
                    return Ok(boolean(false));
                }
                values.push(value);
            }
        }
        if matches!(kind, "any" | "all") {
            Ok(boolean(kind == "all"))
        } else {
            self.collect(kind, values)
        }
    }
    pub fn attempt(
        &mut self,
        body: impl FnOnce(&mut Self) -> Result<(), Error>,
    ) -> Result<(), Error> {
        match body(self) {
            Err(e) if matches!(e.root(), Error::BindingGap { .. }) => Err(e),
            _ => Ok(()),
        }
    }
    pub fn exception(&self, class: &str, message: &Options) -> Result<Error, Error> {
        let message = string(message)?;
        Ok(error(
            match class {
                "ValueError" => "ValueError",
                _ => return Err(gap(format!("detector exception class {class}"))),
            },
            message,
        ))
    }
}
fn string(value: &Options) -> Result<String, Error> {
    if let Options::Atom(V::Text(value)) = value {
        Ok(value.clone())
    } else {
        Err(gap("non-string detector argument"))
    }
}
fn strings(values: &[Options]) -> Result<Vec<String>, Error> {
    values.iter().map(string).collect()
}

fn parse_json(value: &Options) -> Result<Options, Error> {
    super::metadata_json::load_options(value)
}
