use super::{
    binding_tests::{check, reference_error, restore},
    device, kitchen,
};
use imagine_inference::{
    checkpoint::state_dict, execution::Runtime, execution_value::Value as V,
    patching::options::Options,
};
use serde_json::{Value, json};

fn entries(value: &Options) -> Vec<Options> {
    match value {
        Options::Dict(v) => v.borrow().iter().map(|(_, v)| v.clone()).collect(),
        _ => vec![],
    }
}
fn aliases(values: &[Options], old: &[Options]) -> Value {
    let owned = |v: &Options| {
        matches!(
            v,
            Options::Dict(_)
                | Options::List(_)
                | Options::Tuple(_)
                | Options::Object(_)
                | Options::Atom(V::Tensor(_) | V::Quantized(_) | V::None)
        )
    };
    json!(
        values
            .iter()
            .map(|v| old
                .iter()
                .map(|o| if owned(v) && owned(o) {
                    Some(
                        v.same_object(o)
                            || matches!((v, o), (Options::Atom(V::None), Options::Atom(V::None))),
                    )
                } else {
                    None
                })
                .collect::<Vec<_>>())
            .collect::<Vec<_>>()
    )
}
#[test]
fn quantization_metadata_preserves_legacy_mutations_json_bytes_order_aliases_and_original_errors() {
    let rt = Runtime::frozen_partial(kitchen()).unwrap();
    let _guard = tch::no_grad_guard();
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-quant-metadata.json")).unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert!(cases.len() > 2500);
    for (index, case) in cases.iter().enumerate() {
        let dev = device(case["device"].as_str().unwrap());
        let sd = Options::from_value(restore(&case["initial"], dev));
        let metadata = Options::from_value(restore(&case["metadata"], dev));
        let old = entries(&sd);
        let note = format!("quant metadata {index} {}", case["label"]);
        let out = unsafe {
            state_dict::convert_old_quants(&rt, &sd, case["prefix"].as_str().unwrap(), &metadata)
        };
        let output = match out {
            Err(e) => {
                assert!(case["error"].is_string(), "{note}: {e}");
                reference_error(e, case);
                None
            }
            Ok((out, meta)) => {
                assert!(
                    !case["error"].is_string(),
                    "{note}: expected {} {}",
                    case["error"],
                    case["message"]
                );
                check(&out.snapshot().unwrap(), &case["output"], dev, &note);
                check(&meta.snapshot().unwrap(), &case["metadata_out"], dev, &note);
                assert_eq!(
                    meta.same_object(&metadata),
                    case["metadata_same"].as_bool().unwrap(),
                    "metadata identity {note}"
                );
                Some(out)
            }
        };
        check(&sd.snapshot().unwrap(), &case["post"], dev, &note);
        check(
            &metadata.snapshot().unwrap(),
            &case["metadata_post"],
            dev,
            &note,
        );
        assert_eq!(
            output.as_ref().is_some_and(|out| out.same_object(&sd)),
            case["root_same"].as_bool().unwrap(),
            "root identity {note}"
        );
        assert_eq!(
            aliases(&output.as_ref().map(entries).unwrap_or_default(), &old),
            case["aliases"],
            "output aliases {note}"
        );
        assert_eq!(
            aliases(&entries(&sd), &old),
            case["post_aliases"],
            "post aliases {note}"
        );
    }
}
