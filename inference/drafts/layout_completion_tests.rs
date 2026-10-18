use super::binding_tests::{check, kwargs, reference_error, restore};
use super::*;
use imagine_inference::{execution::Runtime, execution_value::Value as V};
use std::sync::{Arc, Mutex};

#[test]
fn layout_preparation_requantization_export_decode_and_dispatch_match_original() {
    let rt = Runtime::frozen_partial(kitchen()).unwrap();
    let trace = Arc::new(Mutex::new(Vec::<Vec<String>>::new()));
    let observed = trace.clone();
    rt.backends.observe_selection(Some(Arc::new(move |op, b| {
        observed.lock().unwrap().push(vec![op.into(), b.into()])
    })));
    for (i, c) in fixtures()["layout_completion"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        trace.lock().unwrap().clear();
        let dev = device(c["device"].as_str().unwrap());
        let args = c["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| restore(v, dev))
            .collect::<Vec<_>>();
        let kwargs = kwargs(&c["kwargs"], dev);
        let guard = c["override"]
            .as_str()
            .map(|name| rt.backends.policy.use_backend(name).unwrap());
        let action = c["action"].as_str().unwrap();
        let out = (|| -> Result<V, imagine_inference::native::Error> {
            Ok(match action {
                "from_float" => V::quantized(unsafe {
                    rt.from_float(
                        &args[0].as_tensor()?,
                        c["layout"].as_str().unwrap(),
                        &kwargs,
                    )?
                }),
                "requantize" => V::quantized(unsafe {
                    rt.requantize_from_float(
                        &args[0].as_quantized()?,
                        &args[1].as_tensor()?,
                        &kwargs,
                    )?
                }),
                "requantize_kwargs" => V::Dict(rt.requantize_kwargs(&args[0].as_quantized()?)?),
                "decode" => V::tensor(unsafe { rt.decode(&args[0].as_quantized()?)? }),
                "state_dict" => V::Dict(
                    rt.state_dict(&args[0].as_quantized()?, "layer.weight")?
                        .into_iter()
                        .map(|(k, t)| (k, V::tensor(t)))
                        .collect(),
                ),
                "op" => unsafe { rt.dispatch(c["operation"].as_str().unwrap(), &args, &kwargs)? },
                _ => panic!("Unknown own layout fixture action"),
            })
        })();
        drop(guard);
        let note = format!("layout {i} {} {} {}", c["layout"], action, c["label"]);
        if c["error"].is_string() {
            match out {
                Err(e) => reference_error(e, c),
                Ok(_) => panic!("Expected original rejection: {note}"),
            }
        } else {
            let out = out.unwrap_or_else(|e| panic!("{note}: {e}"));
            check(&out, &c["output"], dev, &note);
        }
        for (v, e) in args.iter().zip(c["post_args"].as_array().unwrap()) {
            check(v, e, dev, &format!("{note} post argument"));
        }
        for ((n, v), e) in kwargs.iter().zip(c["post_kwargs"].as_array().unwrap()) {
            assert_eq!(n, e[0].as_str().unwrap());
            check(v, &e[1], dev, &format!("{note} post keyword"));
        }
        assert_eq!(
            serde_json::json!(*trace.lock().unwrap()),
            c["backend_calls"],
            "{note} backend trace"
        );
    }
}
