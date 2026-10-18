use super::binding_tests::{check, kwargs, reference_error, restore};
use super::*;
use imagine_inference::{
    execution::Runtime,
    execution_value::Value as V,
    kitchen_direct::{DeltaArgs, flash_num_splits},
    native::Error,
};
fn integer(v: &V) -> i64 {
    if let V::Int(n) = v {
        *n
    } else {
        panic!("Expected own integer fixture")
    }
}
fn real(v: &V) -> f64 {
    if let V::Float(n) = v {
        *n
    } else {
        integer(v) as f64
    }
}
fn target(v: &V) -> Option<Device> {
    match v {
        V::None => None,
        V::Device(d) => Some(*d),
        _ => panic!("Expected own device fixture"),
    }
}
#[test]
fn public_flash_decode_and_deltanet_helpers_preserve_outputs_mutations_and_errors() {
    let rt = Runtime::frozen_partial(kitchen()).unwrap();
    for (i, c) in fixtures()["kitchen_direct"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let dev = device(c["device"].as_str().unwrap());
        let args = c["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| restore(v, dev))
            .collect::<Vec<_>>();
        let kw = kwargs(&c["kwargs"], dev);
        let t = |i: usize| args[i].as_tensor();
        let optional = |name: &str| -> Result<Option<Tensor>, Error> {
            kw.iter()
                .find(|(n, _)| n == name)
                .map(|(_, v)| v.as_tensor())
                .transpose()
        };
        let result = (|| -> Result<V, Error> {
            Ok(match c["operation"].as_str().unwrap() {
                "flash_available" => V::Bool(rt.flash_decode_available(target(&args[0]))?),
                "delta_available" => V::Bool(rt.gated_delta_available(
                    target(&args[0]),
                    integer(&args[1]),
                    integer(&args[2]),
                )?),
                "num_splits" => V::Int(flash_num_splits(
                    integer(&args[0]),
                    integer(&args[1]),
                    integer(&args[2]),
                    integer(&args[3]),
                )?),
                "flash_decode" => {
                    V::tensor(unsafe { rt.flash_attention_decode(&t(0)?, &t(1)?, &t(2)?, &t(3)?)? })
                }
                "conv_step" => V::tensor(unsafe {
                    rt.deltanet_conv_step(
                        &t(0)?,
                        &t(1)?,
                        &t(2)?,
                        optional("conv_b")?.as_ref(),
                        optional("snapshots")?.as_ref(),
                    )?
                }),
                "delta_decode" => V::tensor(unsafe {
                    rt.gated_delta_decode_fused(DeltaArgs {
                        mixed_qkv: &t(0)?,
                        x: &t(1)?,
                        w_a: &t(2)?,
                        w_b: &t(3)?,
                        dt_bias: &t(4)?,
                        g_decay: &t(5)?,
                        state: &t(6)?,
                        key_dim: integer(&args[7]),
                        num_key_heads: integer(&args[8]),
                        scale: real(&args[9]),
                        z: &t(10)?,
                        norm_weight: &t(11)?,
                        eps: real(&args[12]),
                        snapshots: optional("snapshots")?.as_ref(),
                    })?
                }),
                _ => panic!("Unknown own direct fixture"),
            })
        })();
        let note = format!("direct {i} {} {}", c["operation"], c["label"]);
        if c["error"].is_string() {
            match result {
                Err(e) => reference_error(e, c),
                Ok(_) => panic!("Expected original error: {note}"),
            }
        } else {
            check(
                &result.unwrap_or_else(|e| panic!("{note}: {e}")),
                &c["output"],
                dev,
                &note,
            );
        }
        for (v, expected) in args.iter().zip(c["post_args"].as_array().unwrap()) {
            check(v, expected, dev, &format!("{note} post argument"));
        }
        for ((n, v), expected) in kw.iter().zip(c["post_kwargs"].as_array().unwrap()) {
            assert_eq!(n, expected[0].as_str().unwrap());
            check(v, &expected[1], dev, &format!("{note} post keyword"));
        }
    }
}
