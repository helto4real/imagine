use super::binding_tests::{check, reference_error, restore};
use super::*;
use imagine_inference::{
    execution::Runtime,
    execution_value::Value as V,
    kitchen_sage::{MaskAllocationObserver, PrequantizedInt8Attention},
    native::Error,
};
struct Prime {
    pattern: f64,
    pointer: usize,
    shape: [i64; 3],
    events: usize,
}
impl MaskAllocationObserver for Prime {
    fn before(&mut self, shape: [i64; 3], device: Device) -> Result<(), Error> {
        assert_eq!(shape, [1, 1, 1164]);
        assert_eq!(device, Device::Cuda(0));
        let mut own = Tensor::f_empty(shape, (Kind::Float, device))?;
        let _ = own.f_fill_(self.pattern)?;
        self.pointer = own.data_ptr() as usize;
        self.shape = shape;
        drop(own);
        Ok(())
    }
    fn after(&mut self, tensor: &Tensor) -> Result<(), Error> {
        if tensor.data_ptr() as usize != self.pointer
            || tensor.size() != self.shape
            || tensor.kind() != Kind::Float
        {
            return Err(Error::Native(
                "Own mask padding provenance failed before the original kernel".into(),
            ));
        }
        self.events += 1;
        Ok(())
    }
}
fn packed(p: &PrequantizedInt8Attention) -> V {
    V::Dict(vec![
        ("q".into(), V::tensor(p.q.shallow_clone())),
        ("k".into(), V::tensor(p.k.shallow_clone())),
        ("v".into(), V::tensor(p.v.shallow_clone())),
        ("q_scale".into(), V::tensor(p.q_scale.shallow_clone())),
        ("k_scale".into(), V::tensor(p.k_scale.shallow_clone())),
        ("v_scale".into(), V::tensor(p.v_scale.shallow_clone())),
        ("original_head_dim".into(), V::Int(p.original_head_dim)),
        ("input_dtype".into(), V::DType(p.input_dtype)),
        ("attention_scale".into(), V::Float(p.attention_scale)),
        ("cta_k".into(), V::Int(p.cta_k)),
        (
            "attn_mask".into(),
            p.attn_mask
                .as_ref()
                .map_or(V::None, |t| V::tensor(t.shallow_clone())),
        ),
    ])
}
#[test]
fn sage_attention_preserves_original_quantized_payloads_scales_masks_outputs_and_errors() {
    let rt = Runtime::frozen_partial(kitchen()).unwrap();
    for (i, c) in fixtures()["sage_attention"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        if c["action"] == "available" {
            let target = c["target"].as_str().map(device);
            check(
                &V::Bool(rt.int8_attention_available(target).unwrap()),
                &c["output"],
                Device::Cpu,
                "Sage availability",
            );
            continue;
        }
        let dev = device(c["device"].as_str().unwrap());
        let a = c["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| restore(v, dev))
            .collect::<Vec<_>>();
        let q = a[0].as_tensor().unwrap();
        let k = a[1].as_tensor().unwrap();
        let v = a[2].as_tensor().unwrap();
        let mask = restore(&c["mask"], dev);
        let mask = if matches!(mask, V::None) {
            None
        } else {
            Some(mask.as_tensor().unwrap())
        };
        let scale = if let Some(s) = c["scale"].as_str() {
            Some(match s {
                "nan" => f64::NAN,
                "inf" => f64::INFINITY,
                "-inf" => f64::NEG_INFINITY,
                _ => panic!("Unknown own scale"),
            })
        } else {
            c["scale"].as_f64()
        };
        let mut prime = c["pattern"].as_f64().map(|pattern| Prime {
            pattern,
            pointer: 0,
            shape: [0; 3],
            events: 0,
        });
        let note = format!("Sage {i} {} {}", c["action"], c["label"]);
        let result = (|| -> Result<V, Error> {
            if c["action"] == "regular" {
                // Prove the complete controlled range before any original mask consumer.
                return Ok(V::tensor(unsafe {
                    rt.int8_attention_observed(
                        &q,
                        &k,
                        &v,
                        scale,
                        mask.as_ref(),
                        prime.as_mut().map(|p| p as &mut dyn MaskAllocationObserver),
                    )?
                }));
            }
            let p = unsafe {
                rt.prequantize_int8_attention_observed(
                    &q,
                    &k,
                    &v,
                    scale,
                    mask.as_ref(),
                    prime.as_mut().map(|p| p as &mut dyn MaskAllocationObserver),
                )?
            };
            check(&packed(&p), &c["packed"], dev, &note);
            if let Some(prime) = &prime {
                assert_eq!(prime.events, 1, "{note} complete own-range guard");
            }
            let out = unsafe { rt.int8_attention_from_prequantized(&p)? };
            let repeat = unsafe { rt.int8_attention_from_prequantized(&p)? };
            check(&V::tensor(repeat), &c["repeat_output"], dev, &note);
            Ok(V::tensor(out))
        })();
        if let Some(prime) = &prime {
            assert_eq!(
                prime.events,
                c["priming_events"].as_array().unwrap().len(),
                "{note} provenance events"
            );
        }
        if c["error"].is_string() {
            match result {
                Err(e) => reference_error(e, c),
                Ok(_) => panic!("Expected original Sage error: {note}"),
            }
        } else {
            check(
                &result.unwrap_or_else(|e| panic!("{note}: {e}")),
                &c["output"],
                dev,
                &note,
            );
        }
        for (v, e) in a.iter().zip(c["post_args"].as_array().unwrap()) {
            check(v, e, dev, &format!("{note} post argument"));
        }
        check(
            &mask
                .as_ref()
                .map_or(V::None, |t| V::tensor(t.shallow_clone())),
            &c["post_mask"],
            dev,
            &format!("{note} post mask"),
        );
    }
}

#[test]
fn sage_prequantized_public_state_preserves_original_crop_dtype_validation_and_errors() {
    let rt = Runtime::frozen_partial(kitchen()).unwrap();
    for (i, c) in fixtures()["sage_prequantized_calls"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let dev = device(c["device"].as_str().unwrap());
        let V::Dict(fields) = restore(&c["packed"], dev) else {
            unreachable!()
        };
        let get = |name: &str| &fields.iter().find(|(n, _)| n == name).unwrap().1;
        let t = |name: &str| get(name).as_tensor().unwrap();
        let int = |name: &str| {
            if let V::Int(n) = get(name) {
                *n
            } else {
                panic!("Expected own integer metadata")
            }
        };
        let V::DType(dtype) = get("input_dtype") else {
            unreachable!()
        };
        let V::Float(scale) = get("attention_scale") else {
            unreachable!()
        };
        let p = PrequantizedInt8Attention {
            q: t("q"),
            k: t("k"),
            v: t("v"),
            q_scale: t("q_scale"),
            k_scale: t("k_scale"),
            v_scale: t("v_scale"),
            original_head_dim: int("original_head_dim"),
            input_dtype: *dtype,
            attention_scale: *scale,
            cta_k: int("cta_k"),
            attn_mask: if matches!(get("attn_mask"), V::None) {
                None
            } else {
                Some(t("attn_mask"))
            },
        };
        let result = unsafe { rt.int8_attention_from_prequantized(&p) };
        let note = format!("Sage packed {i} {}", c["label"]);
        if c["error"].is_string() {
            reference_error(result.expect_err("Expected original packed-state error"), c);
        } else {
            check(
                &V::tensor(result.unwrap_or_else(|e| panic!("{note}: {e}"))),
                &c["output"],
                dev,
                &note,
            );
        }
        check(
            &packed(&p),
            &c["post_packed"],
            dev,
            &format!("{note} post packed state"),
        );
    }
}
