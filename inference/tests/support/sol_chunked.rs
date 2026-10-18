use super::binding_tests::{check, reference_error, restore};
use super::*;
use imagine_inference::{
    execution::Runtime,
    execution_value::Value as V,
    kitchen_sol::{AllocationContext, allocation_context, set_allocation_context},
    native::Error,
};
use std::sync::{Arc, Mutex};
struct Context {
    label: &'static str,
    mode: String,
    events: Arc<Mutex<Vec<Value>>>,
}
impl AllocationContext for Context {
    fn enter(&self) -> Result<(), Error> {
        self.events
            .lock()
            .unwrap()
            .push(serde_json::json!([self.label, "enter", null]));
        if self.mode == "enter_error" {
            return Err(Error::ReferenceException {
                class: "ValueError",
                message: "own allocation context enter failure".into(),
            });
        }
        Ok(())
    }
    fn exit(&self, failure: Option<&Error>) -> Result<bool, Error> {
        let class = failure.map(|e| match e {
            Error::ReferenceException { class, .. } => *class,
            _ => "RuntimeError",
        });
        self.events
            .lock()
            .unwrap()
            .push(serde_json::json!([self.label, "exit", class]));
        if self.mode == "exit_error" {
            return Err(Error::ReferenceException {
                class: "RuntimeError",
                message: "own allocation context exit failure".into(),
            });
        }
        Ok(self.mode == "suppress")
    }
}
struct RestoreContext(Option<Arc<dyn AllocationContext>>);
impl Drop for RestoreContext {
    fn drop(&mut self) {
        set_allocation_context(self.0.take());
    }
}
#[test]
fn sol_rope_cache_preserves_tensor_identity_weak_lifetime_context_callbacks_and_quantized_inputs() {
    let rt = Runtime::frozen_partial(kitchen()).unwrap();
    let _restore = RestoreContext(allocation_context());
    for (i, c) in fixtures()["sol_fab_cache"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let dev = device(c["device"].as_str().unwrap());
        let input = restore(&c["input"], dev);
        if c["outputs"].is_array() {
            set_allocation_context(None);
            let copied = unsafe {
                rt.op("aten::clone.default", std::slice::from_ref(&input), &vec![])
                    .unwrap()
            };
            for (q, expected) in [&input, &input, &copied]
                .into_iter()
                .zip(c["outputs"].as_array().unwrap())
            {
                let out = unsafe { rt.sol_packed_rope_fab(q, 3).unwrap() };
                check(&V::tensor(out), expected, dev, "quantized Sol frequencies");
            }
            continue;
        }
        let events = Arc::new(Mutex::new(vec![]));
        let mode = c["mode"].as_str().unwrap().to_string();
        set_allocation_context(Some(Arc::new(Context {
            label: "A",
            mode,
            events: events.clone(),
        })));
        let mut first = None::<Tensor>;
        let mut alias = None::<V>;
        for step in c["steps"].as_array().unwrap() {
            let action = step["action"].as_str().unwrap();
            if action == "mutate" {
                let mut t = input.as_tensor().unwrap();
                let _ = t.f_add_scalar_(0.125).unwrap();
            }
            if action == "alias" {
                alias = Some(unsafe {
                    rt.op(
                        "aten::view_as.default",
                        &[input.clone(), input.clone()],
                        &vec![],
                    )
                    .unwrap()
                });
            }
            if action == "context_hot" {
                set_allocation_context(Some(Arc::new(Context {
                    label: "B",
                    mode: "normal".into(),
                    events: events.clone(),
                })));
            }
            let current = if action == "alias" {
                alias.as_ref().unwrap().clone()
            } else if action == "context_miss" {
                unsafe {
                    rt.op("aten::clone.default", std::slice::from_ref(&input), &vec![])
                        .unwrap()
                }
            } else {
                input.clone()
            };
            let result = unsafe { rt.sol_packed_rope_fab(&current, 3) };
            let note = format!("Sol fabric {i} {} {action}", c["mode"]);
            if step["error"].is_string() {
                match result {
                    Err(e) => reference_error(e, step),
                    Ok(_) => panic!("Expected context failure: {note}"),
                }
            } else {
                let out = result.unwrap_or_else(|e| panic!("{note}: {e}"));
                assert_eq!(
                    first
                        .as_ref()
                        .is_some_and(|first| rt.native.tensor_identity(first)
                            == rt.native.tensor_identity(&out)),
                    step["same_first"].as_bool().unwrap(),
                    "{note} cached object"
                );
                check(&V::tensor(out.shallow_clone()), &step["output"], dev, &note);
                if action == "first" {
                    first = Some(out);
                }
            }
            assert_eq!(
                serde_json::json!(*events.lock().unwrap()),
                step["events"],
                "{note} allocation context trace"
            );
        }
        drop(alias);
        drop(first);
        let V::Tensor(t) = input else { unreachable!() };
        let watched = Arc::downgrade(&t);
        drop(t);
        assert_eq!(
            watched.upgrade().is_none(),
            c["source_released"].as_bool().unwrap(),
            "Sol cache must not keep input tensor objects alive"
        );
    }
}

fn own_source_iterator(
    chunks: &[Tensor],
    events: Arc<Mutex<Vec<Value>>>,
    fail: bool,
) -> Box<dyn Iterator<Item = Result<Tensor, Error>>> {
    let chunks = chunks.iter().map(Tensor::shallow_clone).collect::<Vec<_>>();
    Box::new(chunks.into_iter().enumerate().map(move |(i, chunk)| {
        events.lock().unwrap().push(serde_json::json!(["yield", i]));
        if fail && i == 1 {
            Err(Error::ReferenceException {
                class: "ValueError",
                message: "own chunk iterator failure".into(),
            })
        } else {
            Ok(chunk)
        }
    }))
}
#[test]
fn sol_chunked_producer_preserves_bootstrap_streaming_statistics_routing_and_errors() {
    use imagine_inference::kitchen_sol::{ChunkSource, SolChunkOptions};
    let rt = Runtime::frozen_partial(kitchen()).unwrap();
    let _restore = RestoreContext(allocation_context());
    set_allocation_context(None);
    for (i, c) in fixtures()["sol_chunked"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let dev = Device::Cuda(0);
        let freq = restore(&c["frequencies"], dev);
        let V::List(weights) = restore(&c["weights"], dev) else {
            unreachable!()
        };
        let weights = weights
            .iter()
            .map(|v| v.as_tensor().unwrap())
            .collect::<Vec<_>>();
        let V::List(chunks) = restore(&c["chunks"], dev) else {
            unreachable!()
        };
        let chunks = chunks
            .iter()
            .map(|v| v.as_tensor().unwrap())
            .collect::<Vec<_>>();
        let kw = super::binding_tests::kwargs(&c["kwargs"], dev);
        let optional = |name: &str| {
            kw.iter()
                .find(|(n, _)| n == name)
                .map(|(_, v)| v.as_tensor().unwrap())
        };
        let km = optional("kmean");
        let vs = optional("vscale");
        let block = optional("block_len");
        let gate = optional("coarse_gate");
        let mut opt = SolChunkOptions {
            kmean: km.as_ref(),
            vscale: vs.as_ref(),
            block_len: block.as_ref(),
            coarse_gate: gate.as_ref(),
            ..Default::default()
        };
        for (name, v) in &kw {
            let float = || match v {
                V::Float(f) => *f,
                V::Int(i) => *i as f64,
                _ => panic!("Expected scalar"),
            };
            let pair = || {
                let V::List(v) = v else {
                    panic!("Expected pair")
                };
                let get = |i| {
                    let V::Int(n) = v[i] else {
                        panic!("Expected integer")
                    };
                    n
                };
                [get(0), get(1)]
            };
            match name.as_str() {
                "tau" => opt.tau = float(),
                "topk_ratio" => opt.topk_ratio = float(),
                "scale" => opt.scale = Some(float()),
                "rope_eps" => opt.rope_eps = float(),
                "sink_blocks" => opt.sink_blocks = Some(pair()),
                "sink_q" => opt.sink_q = Some(pair()),
                "tail" => {
                    let V::Bool(v) = v else { unreachable!() };
                    opt.tail = *v;
                }
                "token_aug" => {
                    let V::Int(v) = v else { unreachable!() };
                    opt.token_aug = *v;
                }
                _ => {}
            }
        }
        let events = Arc::new(Mutex::new(vec![]));
        let fail_yield = c["label"] == "yield_error";
        let source = match c["source_kind"].as_str().unwrap() {
            "iterator" => ChunkSource::Iterator(Some(own_source_iterator(
                &chunks,
                events.clone(),
                fail_yield,
            ))),
            "replay" => ChunkSource::Replay(chunks.iter().map(Tensor::shallow_clone).collect()),
            "factory" => {
                let events = events.clone();
                let fail = c["label"] == "factory_error";
                ChunkSource::Factory(Box::new(move || {
                    events.lock().unwrap().push(serde_json::json!(["factory"]));
                    if fail {
                        return Err(Error::ReferenceException {
                            class: "RuntimeError",
                            message: "own chunk factory failure".into(),
                        });
                    }
                    Ok(own_source_iterator(&chunks, events.clone(), fail_yield))
                }))
            }
            _ => panic!("Unknown own source"),
        };
        let result = unsafe {
            rt.sol_attn_chunked(
                source,
                c["t"].as_i64().unwrap(),
                c["h"].as_i64().unwrap(),
                &freq,
                (&weights[0], &weights[1]),
                &opt,
            )
        };
        let note = format!(
            "Sol chunked {i} {} {} {}",
            c["label"], c["stats"], c["source_kind"]
        );
        if c["error"].is_string() {
            match result {
                Err(e) => reference_error(e, c),
                Ok(_) => panic!("Expected error: {note}"),
            }
        } else {
            let (out, km, vs) = result.unwrap_or_else(|e| panic!("{note}: {e}"));
            check(
                &V::Tuple(vec![V::tensor(out), V::tensor(km), V::tensor(vs)]),
                &c["output"],
                dev,
                &note,
            );
        }
        assert_eq!(
            serde_json::json!(*events.lock().unwrap()),
            c["events"],
            "{note} source callback order"
        );
    }
}
