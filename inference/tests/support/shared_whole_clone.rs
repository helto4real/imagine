use imagine_inference::{
    execution_value::Value as V,
    memory::{Handle, Model, Patcher, source::OwnerBindings},
    native::Error,
    patching::{
        attributes::AttributeStore,
        deepcopy::DeepCopyBindings,
        model_clone::{CloneArguments, CloneBindings, clone_patcher},
        model_patcher::{Initialization, InitializationBindings, PatcherState},
        options::{GeneralMapping, Options, SourceText},
        relationships::RelationshipCloneBindings,
    },
};
use serde_json::{Value, json};
use std::{
    any::Any,
    cell::{Cell, RefCell},
    rc::Rc,
};

struct Owner {
    label: String,
    dynamic: bool,
}
enum Object {
    Model(String, Options),
    Token(String),
    Set,
    Factory,
}
fn object(value: &Options) -> &Object {
    let Options::Object(v) = value else {
        panic!("own object")
    };
    if let Some(model) =
        v.downcast_ref::<imagine_inference::memory_finalizers::ModelObject<'static, Object>>()
    {
        return &model.data;
    }
    v.downcast_ref::<Object>().expect("own foreign object")
}
fn canonical(value: &Options) -> Option<&Patcher<'static>> {
    let Options::Object(v) = value else {
        return None;
    };
    v.downcast_ref::<Patcher<'static>>()
}
fn owner(value: &Options) -> &PatcherState {
    canonical(value)
        .expect("one canonical memory owner")
        .source_state()
        .unwrap()
}
fn metadata(value: &Options) -> &Owner {
    canonical(value)
        .unwrap()
        .source_bindings()
        .unwrap()
        .as_any()
        .downcast_ref::<Owner>()
        .unwrap()
}
fn label(value: &Options) -> &str {
    if canonical(value).is_some() {
        return &metadata(value).label;
    }
    match object(value) {
        Object::Model(v, _) | Object::Token(v) => v,
        _ => panic!("own labelled object"),
    }
}
impl OwnerBindings<'static> for Owner {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn load_device(&self, state: &PatcherState) -> Result<tch::Device, Error> {
        assert_eq!(label(&state.fields.get("load_device")?), "load");
        Ok(tch::Device::Cuda(0))
    }
    fn parent(&self, state: &PatcherState) -> Result<Option<Handle<'static>>, Error> {
        match state.fields.get("parent")? {
            Options::Atom(V::None) => Ok(None),
            Options::Object(v) => {
                v.downcast::<Patcher<'static>>()
                    .map(Some)
                    .map_err(|_| Error::BindingGap {
                        backend: "own_whole_clone_fixture".into(),
                        operation: "noncanonical parent".into(),
                    })
            }
            _ => panic!("own source parent"),
        }
    }
    fn model_patches_models(&self, _: &PatcherState) -> Result<Vec<Handle<'static>>, Error> {
        Err(Error::BindingGap {
            backend: "own_whole_clone_fixture".into(),
            operation: "metadata clone fixture defines no kernel patch dependency methods".into(),
        })
    }
    // The oracle's own Owner class includes original constructor/clone bodies,
    // without ModelPatcher.__del__. This is not a production destructor binding.
    fn finalize(&self, _: &Patcher<'static>) -> Result<(), Error> {
        Ok(())
    }
    fn unraisable(&self, error: Error) {
        panic!("unexpected own source finalizer error: {error:?}")
    }
}
struct ModelView {
    fields: AttributeStore,
    dynamic: bool,
}
impl Model<'static> for ModelView {
    fn loaded_ram_size(&self) -> Result<f64, Error> {
        Ok(0.)
    }
    fn model_dtype(&self) -> Result<Option<tch::Kind>, Error> {
        Err(Error::BindingGap {
            backend: "own_metadata_fixture".into(),
            operation: "no dtype method in this fixture class".into(),
        })
    }
    fn model_patches_to(
        &self,
        _target: imagine_inference::memory::PatchTarget,
    ) -> Result<(), Error> {
        Err(Error::BindingGap {
            backend: "own_metadata_fixture".into(),
            operation: "no patch-movement method in this fixture class".into(),
        })
    }

    fn offload_device(&self) -> Result<tch::Device, Error> {
        Ok(tch::Device::Cpu)
    }
    fn storage(&self) -> imagine_inference::memory_finalizers::Handle<'static> {
        let model = self.fields.get("model").unwrap();
        let Options::Object(model) = model else {
            panic!("own physical model")
        };
        model
            .downcast::<imagine_inference::memory_finalizers::ModelObject<'static, Object>>()
            .unwrap_or_else(|_| panic!("own canonical physical model"))
    }
    fn load_device(&self) -> tch::Device {
        assert_eq!(label(&self.fields.get("load_device").unwrap()), "load");
        tch::Device::Cuda(0)
    }
    fn current_device(&self) -> Result<tch::Device, Error> {
        Ok(tch::Device::Cpu)
    }
    fn size(&self) -> Result<f64, Error> {
        Ok({
            let Options::Atom(V::Int(size)) = self.fields.get("size").unwrap() else {
                unreachable!()
            };
            size as f64
        })
    }
    fn loaded_size(&self) -> Result<f64, Error> {
        Ok(0.)
    }
    fn partially_load(&self, _device: tch::Device, _: f64, _: bool) -> Result<f64, Error> {
        panic!("own clone fixture does not define weight loading")
    }
    fn partially_unload(&self, _device: tch::Device, _: f64) -> Result<f64, Error> {
        panic!("own clone fixture does not define weight loading")
    }
    fn detach(&self, _: bool) -> Result<(), Error> {
        panic!("own clone fixture does not define source detachment")
    }
    fn dynamic(&self) -> Result<bool, Error> {
        Ok(self.dynamic)
    }
}
fn list(values: Vec<Options>) -> Options {
    Options::List(Rc::new(RefCell::new(values)))
}
fn tuple(values: Vec<Options>) -> Options {
    Options::Tuple(Rc::new(values))
}
fn atom(value: V) -> Options {
    Options::Atom(value)
}
fn same(a: &Options, b: &Options) -> bool {
    a.same_object(b)
        || match (a, b) {
            (Options::Atom(V::None), Options::Atom(V::None)) => true,
            (Options::Atom(V::Bool(a)), Options::Atom(V::Bool(b))) => a == b,
            // This corpus's integer objects are all source cached small integers.
            (Options::Atom(V::Int(a)), Options::Atom(V::Int(b))) => a == b,
            _ => false,
        }
}
struct Bindings {
    keys: Rc<super::own_mapping_keys::OwnMappingKeys>,
    owners: RefCell<Vec<Options>>,
    tokens: RefCell<Vec<(String, Options)>>,
    uuid: Cell<usize>,
    child: Cell<usize>,
    depth: Cell<usize>,
    settings: Value,
    events: Vec<Value>,
    position: Cell<usize>,
    recording: Cell<bool>,
    action: Value,
    factory: Options,
}
impl Bindings {
    fn token(&self, name: &str) -> Options {
        let mut pool = self.tokens.borrow_mut();
        if let Some((_, v)) = pool.iter().find(|(k, _)| k == name) {
            return v.clone();
        }
        let value = Options::Object(Rc::new(Object::Token(name.into())));
        pool.push((name.into(), value.clone()));
        value
    }
    fn model(&self, name: &str) -> Options {
        Options::Object(
            imagine_inference::memory_finalizers::ModelObject::with_data(Object::Model(
                name.into(),
                Options::dictionary(vec![("device".into(), self.token("offload"))]),
            )),
        )
    }
    fn dictionary(&self, pairs: Vec<(Options, Options)>) -> Options {
        let out = self.keys.empty();
        for (k, v) in pairs {
            out.set_item(k, v).unwrap()
        }
        out
    }
    fn text(&self, name: &str) -> Options {
        SourceText::from_utf8(name)
    }
    fn strings(&self, pairs: Vec<(&str, Options)>) -> Options {
        self.dictionary(pairs.into_iter().map(|(k, v)| (self.text(k), v)).collect())
    }
    fn event(&self, op: &str, args: Vec<(&str, Value)>) -> Result<(), Error> {
        if !self.recording.get() {
            return Ok(());
        }
        let index = self.position.get();
        self.position.set(index + 1);
        let event = self.events.get(index).expect("extra whole-clone operation");
        assert_eq!(event["op"], op, "event {index}");
        for (k, v) in args {
            assert_eq!(event[k], v, "event {index} input {k}")
        }
        if event.get("error").is_some() {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: "own whole clone failure".into(),
            });
        }
        Ok(())
    }
    fn create(&self, dynamic: bool, args: &Initialization) -> Result<Options, Error> {
        let index = self.owners.borrow().len();
        let state = PatcherState::uninitialized(if dynamic { "Dynamic" } else { "Owner" });
        let model = ModelView {
            fields: state.fields.clone(),
            dynamic,
        };
        let value = Options::Object(Patcher::from_source(
            model,
            state,
            Rc::new(Owner {
                label: if index == 0 {
                    "source".into()
                } else {
                    format!("created:{}", index - 1)
                },
                dynamic,
            }),
        ));
        self.owners.borrow_mut().push(value.clone());
        owner(&value).initialize(args, self)?;
        Ok(value)
    }
    fn key(value: &Options) -> Value {
        if let Some(v) = SourceText::get(value) {
            return json!(["text", v.utf8().unwrap()]);
        }
        match value {
            Options::Atom(V::None) => json!(["none"]),
            Options::Atom(V::Text(v)) => json!(["text", v]),
            Options::Tuple(v) => json!(["tuple", v.iter().map(Self::key).collect::<Vec<_>>()]),
            _ => json!(["object", label(value)]),
        }
    }
    fn snapshot(&self) -> Value {
        fn index(objects: &mut Vec<Options>, value: &Options) -> usize {
            if let Some(i) = objects.iter().position(|v| same(v, value)) {
                i
            } else {
                objects.push(value.clone());
                objects.len() - 1
            }
        }
        let mut objects = vec![];
        let roots = self
            .owners
            .borrow()
            .iter()
            .map(|v| index(&mut objects, v))
            .collect::<Vec<_>>();
        let mut records = vec![];
        let mut position = 0;
        while position < objects.len() {
            let value = objects[position].clone();
            let record = if canonical(&value).is_some() {
                let v = metadata(&value);
                json!([
                    "owner",
                    v.label,
                    if v.dynamic { "dynamic" } else { "base" },
                    owner(&value)
                        .fields
                        .entries()
                        .iter()
                        .map(|(k, v)| json!([k, index(&mut objects, v)]))
                        .collect::<Vec<_>>()
                ])
            } else if let Some(mapping) = GeneralMapping::get(&value) {
                json!([
                    "dict",
                    mapping
                        .dictionary
                        .entries()
                        .iter()
                        .map(|(k, v)| json!([Self::key(k), index(&mut objects, v)]))
                        .collect::<Vec<_>>()
                ])
            } else {
                match &value {
                    Options::Object(_) => match object(&value) {
                        Object::Model(name, fields) => {
                            let Options::Dict(fields) = fields else {
                                unreachable!()
                            };
                            json!([
                                "model",
                                name,
                                fields
                                    .borrow()
                                    .iter()
                                    .map(|(k, v)| json!([k, index(&mut objects, v)]))
                                    .collect::<Vec<_>>()
                            ])
                        }
                        Object::Token(name) => json!(["object", name]),
                        Object::Set => json!(["set"]),
                        Object::Factory => json!(["factory"]),
                    },
                    Options::Dict(v) => json!([
                        "dict",
                        v.borrow()
                            .iter()
                            .map(|(k, v)| json!([["text", k], index(&mut objects, v)]))
                            .collect::<Vec<_>>()
                    ]),
                    Options::List(v) => json!([
                        "list",
                        v.borrow()
                            .iter()
                            .map(|v| index(&mut objects, v))
                            .collect::<Vec<_>>()
                    ]),
                    Options::Tuple(v) => json!([
                        "tuple",
                        v.iter().map(|v| index(&mut objects, v)).collect::<Vec<_>>()
                    ]),
                    Options::Atom(V::None) => json!(["none"]),
                    Options::Atom(V::Bool(v)) => json!(["bool", v]),
                    Options::Atom(V::Int(v)) => json!(["int", v]),
                    _ => panic!("own graph value"),
                }
            };
            records.push(record);
            position += 1;
        }
        json!({"roots":roots,"objects":records})
    }
    fn setup(&self) -> Options {
        let original = self
            .create(
                self.settings["dynamic"].as_bool().unwrap(),
                &Initialization {
                    model: self.model("original"),
                    load_device: self.token("load"),
                    offload_device: self.token("offload"),
                    size: atom(V::Int(0)),
                    weight_inplace_update: atom(V::Bool(false)),
                    fast_disk: atom(V::Bool(false)),
                },
            )
            .unwrap();
        let f = &owner(&original).fields;
        f.set(
            "patches",
            self.strings(vec![(
                "weight",
                list(vec![self.token("patch:0"), self.token("patch:1")]),
            )]),
        );
        f.set(
            "object_patches",
            self.strings(vec![("object", self.token("object:0"))]),
        );
        f.set(
            "weight_wrapper_patches",
            self.strings(vec![("weight", list(vec![self.token("wrapper:0")]))]),
        );
        let shared = list(vec![self.token("option:0")]);
        f.set(
            "model_options",
            self.strings(vec![
                (
                    "transformer_options",
                    self.strings(vec![("patches", shared.clone()), ("again", shared.clone())]),
                ),
                ("opaque", tuple(vec![shared])),
            ]),
        );
        f.set(
            "attachments",
            self.strings(vec![
                ("a", self.token("attachment:0")),
                ("plain", self.token("plain")),
            ]),
        );
        f.set(
            "additional_models",
            self.strings(vec![(
                "g",
                list(vec![self.token("child:0"), self.token("child:0")]),
            )]),
        );
        f.set(
            "callbacks",
            self.strings(vec![(
                "on_clone",
                self.dictionary(vec![
                    (atom(V::None), list(vec![self.token("callback:0")])),
                    (self.text("extra"), list(vec![self.token("callback:1")])),
                ]),
            )]),
        );
        f.set(
            "wrappers",
            self.strings(vec![(
                "sample",
                self.dictionary(vec![(atom(V::None), list(vec![self.token("wrapper:0")]))]),
            )]),
        );
        f.set(
            "injections",
            self.strings(vec![("injection", list(vec![self.token("injection:0")]))]),
        );
        let patches = self.dictionary(vec![(
            self.token("hook-ref"),
            self.strings(vec![(
                "weight",
                list(vec![tuple(vec![self.token("patch:0")])]),
            )]),
        )]);
        f.set("hook_patches", patches.clone());
        let empty = self.settings["empty"].as_bool().unwrap();
        f.set(
            "hook_patches_backup",
            if empty { self.keys.empty() } else { patches },
        );
        f.set(
            "cached_hook_patches",
            self.dictionary(vec![(
                self.token("hook-group"),
                self.strings(vec![("weight", self.token("cached:0"))]),
            )]),
        );
        f.set(
            "hook_backup",
            self.strings(vec![("weight", self.token("backup:0"))]),
        );
        f.set(
            "current_hooks",
            if empty {
                atom(V::None)
            } else {
                self.token("hook:0")
            },
        );
        f.set(
            "forced_hooks",
            if empty {
                list(vec![])
            } else {
                self.token("hook:1")
            },
        );
        f.set(
            "cached_patcher_init",
            if self.settings["cache"] == "none" {
                atom(V::None)
            } else {
                let mut v = vec![self.factory.clone(), tuple(vec![atom(V::Int(11))])];
                if self.settings["cache"] == "indexed" {
                    v.push(atom(V::Int(0)))
                }
                tuple(v)
            },
        );
        original
    }
}
impl InitializationBindings for Bindings {
    fn empty_dictionary(&self) -> Result<Options, Error> {
        Ok(self.keys.empty())
    }
    fn model_attribute(&self, model: &Options, name: &str) -> Result<Options, Error> {
        let Object::Model(_, fields) = object(model) else {
            unreachable!()
        };
        fields.get(name)?.ok_or_else(|| Error::ReferenceException {
            class: "AttributeError",
            message: name.into(),
        })
    }
    fn set_model_attribute(
        &self,
        model: &Options,
        name: &str,
        value: Options,
    ) -> Result<(), Error> {
        let Object::Model(_, fields) = object(model) else {
            unreachable!()
        };
        fields.set(name, value)
    }
    fn debug(&self, _: &str) -> Result<(), Error> {
        panic!("own model has device")
    }
    fn uuid4(&self) -> Result<Options, Error> {
        let id = self.uuid.get();
        self.uuid.set(id + 1);
        Ok(self.token(&format!("uuid:{id}")))
    }
    fn empty_pinned_set(&self) -> Result<Options, Error> {
        Ok(Options::Object(Rc::new(Object::Set)))
    }
    fn max_speed_hook_mode(&self) -> Result<Options, Error> {
        Ok(self.token("maxspeed"))
    }
}
impl RelationshipCloneBindings for Bindings {
    fn empty_dictionary(&self) -> Result<Options, Error> {
        Ok(self.keys.empty())
    }
    fn clone_attribute(&self, value: &Options) -> Result<Options, Error> {
        self.event("attachment-attribute", vec![("label", json!(label(value)))])?;
        if label(value).starts_with("attachment:") {
            Ok(self.token(&format!("method:{}", label(value))))
        } else {
            Err(Error::ReferenceException {
                class: "AttributeError",
                message: "own missing clone attribute".into(),
            })
        }
    }
    fn call_clone_attribute(&self, method: &Options) -> Result<Options, Error> {
        let name = label(method).strip_prefix("method:").unwrap();
        self.event("attachment-call", vec![("label", json!(name))])?;
        Ok(self.token(&format!("cloned:{name}")))
    }
    fn clone_model(&self, value: &Options) -> Result<Options, Error> {
        self.event("child-clone", vec![("label", json!(label(value)))])?;
        let id = self.child.get();
        self.child.set(id + 1);
        Ok(self.token(&format!("cloned-child:{id}")))
    }
}
impl DeepCopyBindings for Bindings {
    fn object_id(&self, value: &Options) -> Result<u64, Error> {
        Ok(match value {
            Options::Object(v) => Rc::as_ptr(v) as *const () as usize as u64,
            Options::Tuple(v) => Rc::as_ptr(v) as usize as u64,
            Options::List(v) => Rc::as_ptr(v) as usize as u64,
            Options::Dict(v) => Rc::as_ptr(v) as usize as u64,
            Options::Atom(V::None) => 1,
            Options::Atom(V::Bool(v)) => 2 + u64::from(*v),
            Options::Atom(V::Int(v)) => 100 + *v as u64,
            _ => panic!("own copied primitive"),
        })
    }
    fn key_object(&self, _: &Options, _: &str) -> Result<Options, Error> {
        panic!("whole-clone corpus uses typed general mappings")
    }
    fn enter_recursion(&self) -> Result<(), Error> {
        assert!(self.depth.get() < 4096, "own acyclic metadata bound");
        self.depth.set(self.depth.get() + 1);
        Ok(())
    }
    fn leave_recursion(&self) {
        self.depth.set(self.depth.get() - 1)
    }
}
impl CloneBindings for Bindings {
    fn state<'a>(&self, value: &'a Options) -> Result<&'a PatcherState, Error> {
        Ok(owner(value))
    }
    fn patcher_class(&self, value: &Options) -> Result<Options, Error> {
        Ok(self.token(if metadata(value).dynamic {
            "class:dynamic"
        } else {
            "class:base"
        }))
    }
    fn base_patcher_class(&self) -> Result<Options, Error> {
        Ok(self.token("class:base"))
    }
    fn is_dynamic(&self, value: &Options) -> Result<Options, Error> {
        self.event("dynamic", vec![("owner", json!(label(value)))])?;
        Ok(atom(V::Bool(metadata(value).dynamic)))
    }
    fn truth(&self, value: &Options) -> Result<bool, Error> {
        if let Some(v) = GeneralMapping::get(value) {
            return Ok(!v.dictionary.is_empty());
        }
        Ok(match value {
            Options::Atom(V::None) => false,
            Options::Atom(V::Bool(v)) => *v,
            Options::List(v) => !v.borrow().is_empty(),
            Options::Tuple(v) => !v.is_empty(),
            Options::Object(_) => true,
            _ => panic!("own truth"),
        })
    }
    fn index(&self, value: &Options, key: &Options) -> Result<Options, Error> {
        let Options::Atom(V::Int(index)) = key else {
            panic!("own integer index")
        };
        Ok(match value {
            Options::Tuple(v) => v[*index as usize].clone(),
            Options::List(v) => v.borrow()[*index as usize].clone(),
            _ => panic!("own indexed sequence"),
        })
    }
    fn length(&self, value: &Options) -> Result<usize, Error> {
        let Options::Tuple(v) = value else {
            panic!("own factory tuple")
        };
        Ok(v.len())
    }
    fn unpack_backups(&self, value: &Options) -> Result<[Options; 4], Error> {
        let Options::Tuple(v) = value else {
            panic!("own backup tuple")
        };
        Ok(v.as_ref().clone().try_into().ok().unwrap())
    }
    fn factory(&self, _: &Options, args: &Options, disable: bool) -> Result<Options, Error> {
        let Options::Tuple(args) = args else {
            unreachable!()
        };
        self.event(
            "factory",
            vec![
                (
                    "arguments",
                    json!(
                        args.iter()
                            .map(|v| {
                                let Options::Atom(V::Int(v)) = v else {
                                    unreachable!()
                                };
                                *v
                            })
                            .collect::<Vec<_>>()
                    ),
                ),
                ("disable", json!(disable)),
            ],
        )?;
        let result = self.construct(
            &self.token("class:base"),
            &Initialization {
                model: self.model("fresh"),
                load_device: self.token("load"),
                offload_device: self.token("offload"),
                size: atom(V::Int(0)),
                weight_inplace_update: atom(V::Bool(false)),
                fast_disk: atom(V::Bool(false)),
            },
        )?;
        Ok(if self.settings["cache"] == "indexed" {
            list(vec![result])
        } else {
            result
        })
    }
    fn construct(&self, class: &Options, args: &Initialization) -> Result<Options, Error> {
        fn number(v: &Options) -> i64 {
            let Options::Atom(V::Int(v)) = v else {
                unreachable!()
            };
            *v
        }
        fn boolean(v: &Options) -> bool {
            let Options::Atom(V::Bool(v)) = v else {
                unreachable!()
            };
            *v
        }
        let dynamic = label(class) == "class:dynamic";
        self.event(
            "construct",
            vec![
                ("class_", json!(if dynamic { "dynamic" } else { "base" })),
                ("model", json!(label(&args.model))),
                ("load", json!(label(&args.load_device))),
                ("offload", json!(label(&args.offload_device))),
                ("size", json!(number(&args.size))),
                ("weight", json!(boolean(&args.weight_inplace_update))),
                ("fast", json!(boolean(&args.fast_disk))),
            ],
        )?;
        self.create(dynamic, args)
    }
    fn model_size(&self, value: &Options) -> Result<Options, Error> {
        self.event("size", vec![("owner", json!(label(value)))])?;
        let f = &owner(value).fields;
        let size = f.get("size")?;
        let Options::Atom(V::Int(n)) = &size else {
            unreachable!()
        };
        if *n > 0 {
            return Ok(size);
        }
        self.event(
            "module-size",
            vec![("model", json!(label(&f.get("model")?)))],
        )?;
        f.set("size", atom(V::Int(27)));
        f.get("size")
    }
    fn model_override(&self, value: &Options) -> Result<Options, Error> {
        self.event("override", vec![("owner", json!(label(value)))])?;
        owner(value).get_clone_model_override()
    }
    fn slice_copy(&self, value: &Options) -> Result<Options, Error> {
        Ok(match value {
            Options::List(v) => list(v.borrow().clone()),
            Options::Tuple(_) => value.clone(),
            _ => panic!("own slice"),
        })
    }
    fn shallow_copy(&self, value: &Options) -> Result<Options, Error> {
        if let Options::List(v) = value {
            return Ok(list(v.borrow().clone()));
        }
        value.shallow_copy_dict()
    }
    fn hook_patches_clone(&self, value: &Options) -> Result<Options, Error> {
        let out = self.keys.empty();
        let refs = value.keys_iterator()?;
        while let Some(key) = refs.next_value()? {
            out.set_item(key.clone(), self.keys.empty())?;
            let keys = value.item(&key)?.keys_iterator()?;
            while let Some(k) = keys.next_value()? {
                let patches = self.slice_copy(&value.item(&key)?.item(&k)?)?;
                out.item(&key)?.set_item(k, patches)?;
            }
        }
        Ok(out)
    }
    fn hook_group_clone(&self, value: &Options) -> Result<Options, Error> {
        self.event("hook-clone", vec![("label", json!(label(value)))])?;
        Ok(self.token(&format!("cloned:{}", label(value))))
    }
    fn clone_callbacks(&self, value: &Options) -> Result<Options, Error> {
        self.event(
            "callbacks",
            vec![("owner", json!(label(value))), ("name", json!("on_clone"))],
        )?;
        let mut output = vec![];
        if let Some(groups) = owner(value).fields.get("callbacks")?.get("on_clone")? {
            let values = groups.values_iterator()?;
            while let Some(group) = values.next_value()? {
                let Options::List(group) = group else {
                    unreachable!()
                };
                output.extend(group.borrow().clone())
            }
        }
        Ok(list(output))
    }
    fn call_callback(
        &self,
        callback: &Options,
        source: &Options,
        target: &Options,
    ) -> Result<(), Error> {
        self.event(
            "callback",
            vec![
                ("label", json!(label(callback))),
                ("source", json!(label(source))),
                ("target", json!(label(target))),
            ],
        )?;
        if label(callback) == "callback:0" {
            let sf = &owner(source).fields;
            let tf = &owner(target).fields;
            match self.action.as_str() {
                None => (),
                Some("target-flag") => tf.set("force_cast_weights", self.token("callback-cast")),
                Some("source-options") => sf
                    .get("model_options")?
                    .set("changed", self.token("changed"))?,
                Some("append-callback" | "clear-callback") => {
                    let group = sf
                        .get("callbacks")?
                        .get("on_clone")?
                        .unwrap()
                        .item(&atom(V::None))?;
                    let Options::List(group) = group else {
                        unreachable!()
                    };
                    if self.action == "append-callback" {
                        group.borrow_mut().push(self.token("callback:2"))
                    } else {
                        group.borrow_mut().clear()
                    }
                }
                _ => unreachable!(),
            }
        }
        Ok(())
    }
}
#[test]
fn original_whole_clone_preserves_factories_dynamic_short_circuit_constructor_sharing_backups_hooks_callbacks_and_partial_failures()
 {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-whole-clone.json")).unwrap();
    assert_eq!(fixture["python_version"], json!([3, 12, 13]));
    assert_eq!(fixture["trained_model_weights"], 0);
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 160);
    for (i, case) in cases.iter().enumerate() {
        let bindings = Bindings {
            keys: super::own_mapping_keys::OwnMappingKeys::seeded(),
            owners: RefCell::new(vec![]),
            tokens: RefCell::new(vec![]),
            uuid: Cell::new(0),
            child: Cell::new(0),
            depth: Cell::new(0),
            settings: case["settings"].clone(),
            events: case["events"].as_array().unwrap().clone(),
            position: Cell::new(0),
            recording: Cell::new(false),
            action: case["action"].clone(),
            factory: Options::Object(Rc::new(Object::Factory)),
        };
        let source = bindings.setup();
        let model_override = if bindings.settings["override"] == true {
            tuple(vec![
                bindings.model("override"),
                tuple(vec![
                    bindings.strings(vec![("own", bindings.token("own-backup"))]),
                    bindings.keys.empty(),
                    bindings.keys.empty(),
                    Options::Object(Rc::new(Object::Set)),
                ]),
            ])
        } else {
            atom(V::None)
        };
        bindings.recording.set(true);
        let result = clone_patcher(
            &source,
            &CloneArguments {
                disable_dynamic: atom(V::Bool(bindings.settings["disable"].as_bool().unwrap())),
                force_deepcopy: atom(V::Bool(bindings.settings["force"].as_bool().unwrap())),
                model_override,
            },
            &bindings,
        );
        let outcome = match result {
            Ok(v) => json!({"ok":true,"output":label(&v)}),
            Err(Error::ReferenceException { class, message }) => {
                json!({"error":class,"message":message})
            }
            Err(e) => panic!("case {i}: {e:?}"),
        };
        let expected = if case.get("ok").is_some() {
            json!({"ok":true,"output":case["output"]})
        } else {
            json!({"error":case["error"],"message":case["message"]})
        };
        assert_eq!(outcome, expected, "outcome {i}");
        assert_eq!(
            bindings.position.get(),
            bindings.events.len(),
            "operations {i}"
        );
        assert_eq!(bindings.depth.get(), 0);
        assert_eq!(bindings.snapshot(), case["after"], "whole clone {i}");
    }
}
