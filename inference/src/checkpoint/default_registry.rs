//! Original _get_allowed_globals body. Native producers supply real modules,
//! iterators, callable owners and literal keys; no global is fabricated here.
use crate::{
    native::Error,
    patching::options::{Options, SourceText},
    source_exception::Frame,
};
use std::rc::Rc;
pub trait Bindings {
    fn load(&mut self, path: &str) -> Result<Options, Error>;
    fn attribute(&mut self, value: &Options, name: &str) -> Result<Options, Error>;
    fn call(&mut self, function: &Options, args: &[Options]) -> Result<Options, Error>;
    fn iter(&mut self, value: &Options) -> Result<Options, Error>;
    fn next(&mut self, iterator: &Options) -> Result<Option<Options>, Error>;
    fn string(&mut self, value: &Options) -> Result<Options, Error>;
    fn format(&mut self, value: &Options) -> Result<Options, Error>;
    fn equal(&mut self, left: &Options, right: &Options) -> Result<bool, Error>;
    /// Original BUILD_CONST_KEY_MAP after all RHS objects have been evaluated.
    fn literal_map(&mut self, entries: Vec<(&'static str, Options)>) -> Result<Options, Error>;
    fn literal_key(&mut self, text: &'static str) -> Result<Options, Error>;
    fn set(&mut self, dictionary: &Options, key: Options, value: Options) -> Result<(), Error>;
}
const INITIAL: &[(&str, &str)] = &[
    ("collections.OrderedDict", "OrderedDict"),
    ("collections.Counter", "Counter"),
    ("torch.nn.parameter.Parameter", "torch.nn.Parameter"),
    (
        "torch.serialization._get_layout",
        "torch.serialization._get_layout",
    ),
    ("torch.Size", "torch.Size"),
    ("torch.Tensor", "torch.Tensor"),
    ("torch.device", "torch.device"),
    ("_codecs.encode", "encode"),
    ("builtins.bytearray", "bytearray"),
    ("builtins.set", "set"),
    ("builtins.complex", "complex"),
];
pub(crate) fn join(left: &Options, right: &Options) -> Result<Options, Error> {
    let (Some(left), Some(right)) = (SourceText::get(left), SourceText::get(right)) else {
        return Err(Error::BindingGap {
            backend: "default_registry".into(),
            operation: "formatted fields require actual Unicode owners".into(),
        });
    };
    let mut points = left.codepoints().to_vec();
    points.push(u32::from('.'));
    points.extend(right.codepoints());
    SourceText::from_codepoints(points)
}
fn class_key(value: &Options, b: &mut dyn Bindings) -> Result<Options, Error> {
    let raw_module = b.attribute(value, "__module__")?;
    let module = b.format(&raw_module)?;
    drop(raw_module);
    let raw_name = b.attribute(value, "__name__")?;
    let name = b.format(&raw_name)?;
    drop(raw_name);
    join(&module, &name)
}
fn each(
    value: Options,
    slot: &mut Option<Options>,
    b: &mut dyn Bindings,
    mut body: impl FnMut(&Options, &mut dyn Bindings) -> Result<(), Error>,
) -> Result<(), Error> {
    let iterator = b.iter(&value)?;
    drop(value);
    while let Some(value) = b.next(&iterator)? {
        drop(slot.replace(value));
        body(slot.as_ref().unwrap(), b)?;
    }
    Ok(())
}
#[derive(Default)]
struct Locals {
    rc: Option<Options>,
    t: Option<Options>,
    tt: Option<Options>,
    ts: Option<Options>,
    qt: Option<Options>,
    f: Option<Options>,
}
fn body(b: &mut dyn Bindings, locals: &mut Locals) -> Result<Options, Error> {
    let mut initial = vec![];
    for (key, path) in INITIAL {
        initial.push((*key, b.load(path)?));
    }
    let rc = b.literal_map(initial)?;
    locals.rc = Some(rc.clone());
    for path in [
        "torch.storage._dtype_to_storage_type_map",
        "torch.storage._new_dtypes",
    ] {
        let function = b.load(path)?;
        let values = b.call(&function, &[])?;
        drop(function);
        each(values, &mut locals.t, b, |value, b| {
            let key = b.string(value)?;
            b.set(&rc, key, value.clone())
        })?;
    }
    for prefix in ["uint", "int"] {
        let mut values = vec![];
        for number in 1..8 {
            values.push(b.load(&format!("torch.{prefix}{number}"))?);
        }
        // Source eagerly builds this exact list before its first str call.
        for value in &values {
            drop(locals.t.replace(value.clone()));
            let value = locals.t.as_ref().unwrap();
            let key = b.string(value)?;
            b.set(&rc, key, value.clone())?;
        }
    }
    let classes = b.load("torch._tensor_classes")?;
    each(classes, &mut locals.tt, b, |value, b| {
        let key = class_key(value, b)?;
        b.set(&rc, key, value.clone())
    })?;
    let classes = b.load("torch._storage_classes")?;
    each(classes, &mut locals.ts, b, |value, b| {
        let types = [
            b.load("torch.storage.TypedStorage")?,
            b.load("torch.storage.UntypedStorage")?,
        ];
        let mut builtin = false;
        for ty in &types {
            if ty.same_object(value) || b.equal(ty, value)? {
                builtin = true;
                break;
            }
        }
        drop(types);
        if builtin {
            let key = class_key(value, b)?;
            return b.set(&rc, key, value.clone());
        }
        // Assignment evaluates the RHS constructor and name before the LHS
        // module/name expression. The second __name__ getter is independent.
        let constructor = b.load("torch.serialization.StorageType")?;
        let name = b.attribute(value, "__name__")?;
        let wrapped = b.call(&constructor, &[name])?;
        drop(constructor);
        let key = class_key(value, b)?;
        b.set(&rc, key, wrapped)
    })?;
    let mut schemes = vec![];
    for path in [
        "torch.per_tensor_affine",
        "torch.per_tensor_symmetric",
        "torch.per_channel_affine",
        "torch.per_channel_symmetric",
        "torch.per_channel_affine_float_qparams",
    ] {
        schemes.push(b.load(path)?);
    }
    for value in &schemes {
        drop(locals.qt.replace(value.clone()));
        let value = locals.qt.as_ref().unwrap();
        let key = b.string(value)?;
        b.set(&rc, key, value.clone())?;
    }
    let function = b.load("_tensor_rebuild_functions")?;
    let functions = b.call(&function, &[])?;
    drop(function);
    each(functions, &mut locals.f, b, |value, b| {
        let raw_name = b.attribute(value, "__name__")?;
        let name = b.format(&raw_name)?;
        drop(raw_name);
        let prefix = b.literal_key("torch._utils")?;
        let key = join(&prefix, &name)?;
        b.set(&rc, key, value.clone())
    })?;
    let function = b.load("torch._tensor._rebuild_from_type_v2")?;
    let key = b.literal_key("torch._tensor._rebuild_from_type_v2")?;
    b.set(&rc, key, function)?;
    Ok(rc)
}
pub fn build(bindings: &mut dyn Bindings) -> Result<Options, Error> {
    let mut locals = Locals::default();
    body(bindings, &mut locals).map_err(|error| {
        let frame = Frame::new("_get_allowed_globals");
        for (name, value) in [
            ("rc", locals.rc),
            ("t", locals.t),
            ("tt", locals.tt),
            ("ts", locals.ts),
            ("qt", locals.qt),
            ("f", locals.f),
        ] {
            if let Some(value) = value {
                frame.set(name, Rc::new(value));
            }
        }
        frame.finish();
        error.retain_frame(frame)
    })
}
