use super::*;
use std::collections::BTreeSet;
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Target {
    pub key: String,
    pub offset: Option<[i64; 3]>,
}
pub struct Loaded {
    pub patches: Vec<(Target, PatchValue)>,
    pub loaded_keys: BTreeSet<String>,
    pub missing_keys: Vec<String>,
}
fn insert<T: PartialEq, V>(out: &mut Vec<(T, V)>, key: T, value: V) {
    if let Some((_, v)) = out.iter_mut().find(|(k, _)| *k == key) {
        *v = value;
    } else {
        out.push((key, value));
    }
}
/// Ordered QwenImage branches of model_lora_keys_unet, including generic aliases.
pub fn qwen_key_map(weights: &[(String, Vec<i64>)]) -> Vec<(String, Target)> {
    let mut out = vec![];
    for (key, _) in weights {
        if let Some(name) = key.strip_prefix("diffusion_model.") {
            let target = Target {
                key: key.clone(),
                offset: None,
            };
            if let Some(stem) = name.strip_suffix(".weight") {
                insert(
                    &mut out,
                    format!("lora_unet_{}", stem.replace('.', "_")),
                    target.clone(),
                );
                insert(
                    &mut out,
                    key.strip_suffix(".weight").unwrap().into(),
                    target,
                );
            } else {
                insert(&mut out, key.clone(), target);
            }
        }
    }
    for (key, shape) in weights {
        if let Some(name) = key
            .strip_prefix("diffusion_model.")
            .and_then(|s| s.strip_suffix(".weight"))
        {
            let targets = if name.ends_with(".img_mlp.gate_up") {
                let half = shape[0] / 2;
                vec![
                    (name.replace(".gate_up", ".gate_layer"), Some([0, 0, half])),
                    (name.replace(".gate_up", ".proj"), Some([0, half, half])),
                ]
            } else {
                vec![(name.to_owned(), None)]
            };
            for (name, offset) in targets {
                for alias in [
                    name.clone(),
                    format!("transformer.{name}"),
                    format!("lycoris_{}", name.replace('.', "_")),
                ] {
                    insert(
                        &mut out,
                        alias,
                        Target {
                            key: key.clone(),
                            offset,
                        },
                    );
                }
            }
        }
    }
    out
}
/// Generic and single-llama-prefix branches used by Qwen3-VL's source wrapper.
pub fn text_key_map(weights: &[String]) -> Vec<(String, Target)> {
    let mut out = vec![];
    let mut prefixes = BTreeSet::new();
    for key in weights {
        if let Some(stem) = key.strip_suffix(".weight") {
            let t = Target {
                key: key.clone(),
                offset: None,
            };
            insert(&mut out, format!("text_encoders.{stem}"), t.clone());
            if let Some(pos) = key
                .find(".transformer.")
                .filter(|p| *p > 0 && !key.starts_with("clip_"))
            {
                insert(&mut out, format!("text_encoders.{}", &stem[pos + 1..]), t);
            }
            prefixes.insert(key.split('.').next().unwrap());
        }
    }
    if prefixes.len() == 1 {
        let prefix = format!("{}.transformer.model.", prefixes.first().unwrap());
        for key in weights {
            if let Some(name) = key
                .strip_prefix(&prefix)
                .and_then(|s| s.strip_suffix(".weight"))
            {
                insert(
                    &mut out,
                    format!("lora_te_{}", name.replace('.', "_")),
                    Target {
                        key: key.clone(),
                        offset: None,
                    },
                );
            }
        }
    }
    out
}
pub fn load_lora(
    state: &BTreeMap<String, Value>,
    mapping: &[(String, Target)],
) -> Result<Loaded, Error> {
    let mut loaded = Loaded {
        patches: vec![],
        loaded_keys: BTreeSet::new(),
        missing_keys: vec![],
    };
    for (x, target) in mapping {
        let mut get = |suffix: &str| -> Option<Value> {
            let key = format!("{x}{suffix}");
            state.get(&key).map(|v| {
                loaded.loaded_keys.insert(key);
                v.clone()
            })
        };
        let alpha = if let Some(v) = get(".alpha") {
            Value::Float(v.as_tensor()?.f_double_value(&[])?)
        } else {
            Value::None
        };
        let dora = get(".dora_scale").unwrap_or(Value::None);
        let mut take = |suffix: &str, required: bool| -> Result<Value, Error> {
            let key = format!("{x}{suffix}");
            if let Some(v) = state.get(&key) {
                loaded.loaded_keys.insert(key);
                Ok(v.clone())
            } else if required {
                Err(exception("KeyError", format!("'{key}'")))
            } else {
                Ok(Value::None)
            }
        };
        let mut adapter = |name: &str, v: Vec<Value>| {
            insert(
                &mut loaded.patches,
                target.clone(),
                PatchValue::Adapter(Adapter {
                    name: name.into(),
                    weights: v,
                }),
            );
        };
        for (up, down, mid) in [
            (
                ".lora_up.weight",
                ".lora_down.weight",
                Some(".lora_mid.weight"),
            ),
            ("_lora.up.weight", "_lora.down.weight", None),
            (".lora_B.weight", ".lora_A.weight", None),
            (".lora.up.weight", ".lora.down.weight", None),
            (".lora_B", ".lora_A", None),
            (
                ".lora_linear_layer.up.weight",
                ".lora_linear_layer.down.weight",
                None,
            ),
            (".lora_B.default.weight", ".lora_A.default.weight", None),
        ] {
            if state.contains_key(&format!("{x}{up}")) {
                let mid = mid
                    .map(|s| take(s, false))
                    .transpose()?
                    .unwrap_or(Value::None);
                let reshape = if let Some(v) = state.get(&format!("{x}.reshape_weight")) {
                    let t = v.as_tensor()?;
                    if t.dim() == 1 {
                        let nums = (0..t.size()[0])
                            .map(|i| t.f_int64_value(&[i]).map(Value::Int))
                            .collect::<Result<Vec<_>, _>>()?;
                        let _ = take(".reshape_weight", true)?;
                        Value::List(nums)
                    } else {
                        Value::None
                    }
                } else {
                    Value::None
                };
                adapter(
                    "lora",
                    vec![
                        take(up, true)?,
                        take(down, true)?,
                        alpha.clone(),
                        mid,
                        dora.clone(),
                        reshape,
                    ],
                );
                break;
            }
        }
        if state.contains_key(&format!("{x}.hada_w1_a")) {
            let cp = state.contains_key(&format!("{x}.hada_t1"));
            adapter(
                "loha",
                vec![
                    take(".hada_w1_a", true)?,
                    take(".hada_w1_b", true)?,
                    alpha.clone(),
                    take(".hada_w2_a", true)?,
                    take(".hada_w2_b", true)?,
                    if cp {
                        take(".hada_t1", true)?
                    } else {
                        Value::None
                    },
                    if cp {
                        take(".hada_t2", true)?
                    } else {
                        Value::None
                    },
                    dora.clone(),
                ],
            );
        }
        let lokr = [
            ".lokr_w1",
            ".lokr_w2",
            ".lokr_w1_a",
            ".lokr_w1_b",
            ".lokr_w2_a",
            ".lokr_w2_b",
            ".lokr_t2",
        ]
        .iter()
        .map(|s| take(s, false))
        .collect::<Result<Vec<_>, _>>()?;
        if [0, 1, 2, 4]
            .iter()
            .any(|i| !matches!(lokr[*i], Value::None))
        {
            adapter(
                "lokr",
                vec![
                    lokr[0].clone(),
                    lokr[1].clone(),
                    alpha.clone(),
                    lokr[2].clone(),
                    lokr[3].clone(),
                    lokr[4].clone(),
                    lokr[5].clone(),
                    lokr[6].clone(),
                    dora.clone(),
                ],
            );
        }
        if state.contains_key(&format!("{x}.a1.weight")) {
            adapter(
                "glora",
                vec![
                    take(".a1.weight", true)?,
                    take(".a2.weight", true)?,
                    take(".b1.weight", true)?,
                    take(".b2.weight", true)?,
                    alpha.clone(),
                    dora.clone(),
                ],
            );
        }
        if let Some(v) = state.get(&format!("{x}.oft_blocks")) {
            let dim = v.as_tensor()?.dim();
            if dim == 3 || dim == 4 {
                adapter(
                    if dim == 3 { "oft" } else { "boft" },
                    vec![
                        take(".oft_blocks", true)?,
                        take(".rescale", false)?,
                        alpha,
                        dora,
                    ],
                );
            }
        }
        let bias = || -> Result<Target, Error> {
            if target.offset.is_some() {
                return Ok(Target {
                    key: "().bias".into(),
                    offset: None,
                });
            }
            let stem = target
                .key
                .get(..target.key.len().saturating_sub(7))
                .unwrap();
            Ok(Target {
                key: format!("{stem}.bias"),
                offset: None,
            })
        };
        for (suffix, kind, is_bias) in [
            (".w_norm", "diff", false),
            (".b_norm", "norm_bias", true),
            (".diff", "diff", false),
            (".diff_b", "diff", true),
            (".set_weight", "set", false),
        ] {
            if kind == "norm_bias" && !state.contains_key(&format!("{x}.w_norm")) {
                continue;
            }
            if let Some(v) = state.get(&format!("{x}{suffix}")) {
                loaded.loaded_keys.insert(format!("{x}{suffix}"));
                let t = if is_bias { bias()? } else { target.clone() };
                insert(
                    &mut loaded.patches,
                    t,
                    if kind == "set" {
                        PatchValue::Set(v.clone())
                    } else {
                        PatchValue::Diff {
                            weight: v.clone(),
                            pad: false,
                        }
                    },
                );
            }
        }
    }
    for (key, v) in state {
        if key.ends_with(".config") {
            loaded.loaded_keys.insert(key.clone());
            insert(
                &mut loaded.patches,
                Target {
                    key: key.clone(),
                    offset: None,
                },
                PatchValue::Config(v.clone()),
            );
        }
    }
    loaded.missing_keys = state
        .keys()
        .filter(|k| !loaded.loaded_keys.contains(*k))
        .cloned()
        .collect();
    Ok(loaded)
}
