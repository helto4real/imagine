use imagine_inference::{
    execution_value::Value as V,
    patching::{
        options::{Options, SourceText},
        pinned_key_set::PinnedKeySet,
    },
    python_hash::HashState,
};
use serde_json::Value;

fn order(set: &PinnedKeySet) -> Vec<Vec<u32>> {
    let Options::List(list) = set.snapshot().unwrap() else {
        panic!("list");
    };
    list.borrow()
        .iter()
        .map(|v| SourceText::get(v).unwrap().codepoints().to_vec())
        .collect()
}
#[test]
fn original_mutable_unicode_set_preserves_dummy_reuse_resize_thresholds_hash_seed_order_and_key_identity()
 {
    let data: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-pinned-key-set.json")).unwrap();
    for case in data["cases"].as_array().unwrap() {
        let set = PinnedKeySet::new(HashState::from_seed(case["seed"].as_u64().unwrap() as u32));
        let keys: Vec<_> = case["keys"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| {
                SourceText::from_codepoints(
                    v.as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_u64().unwrap() as u32)
                        .collect(),
                )
                .unwrap()
            })
            .collect();
        let commands = case["commands"].as_array().unwrap();
        let mut executed = 0;
        let mut result = None;
        for observation in case["observations"].as_array().unwrap() {
            let until = observation["after"].as_u64().unwrap() as usize;
            while executed < until {
                let command = &commands[executed];
                let key = &keys[command[1].as_u64().unwrap() as usize];
                result = match command[0].as_str().unwrap() {
                    "add" => {
                        set.add(key).unwrap();
                        None
                    }
                    "discard" => Some(set.discard(key).unwrap()),
                    "contains" => Some(set.contains(key).unwrap()),
                    _ => unreachable!(),
                };
                executed += 1;
            }
            if let Some(expected) = observation.get("result") {
                assert_eq!(serde_json::to_value(result).unwrap(), *expected);
            }
            if let Some(expected) = observation.get("order") {
                let expected: Vec<_> = expected
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|i| {
                        SourceText::get(&keys[i.as_u64().unwrap() as usize])
                            .unwrap()
                            .codepoints()
                            .to_vec()
                    })
                    .collect();
                assert_eq!(
                    order(&set),
                    expected,
                    "seed={} after={executed}",
                    case["seed"]
                );
            }
        }
        if let Some(bulk) = case["bulk"].as_array() {
            let set = PinnedKeySet::new(HashState::from_seed(0));
            for phase in bulk {
                for i in phase["start"].as_u64().unwrap()..phase["stop"].as_u64().unwrap() {
                    let key = SourceText::from_utf8(&format!("bulk.weight.{i}"));
                    if phase["op"] == "add" {
                        set.add(&key).unwrap();
                    } else {
                        assert!(set.discard(&key).unwrap());
                    }
                }
                let expected: Vec<Vec<u32>> = phase["order"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| {
                        format!("bulk.weight.{}", v.as_u64().unwrap())
                            .chars()
                            .map(u32::from)
                            .collect()
                    })
                    .collect();
                assert_eq!(order(&set), expected);
            }
        }
    }
    let set = PinnedKeySet::new(HashState::from_seed(42));
    let original = SourceText::from_utf8("own.weight");
    set.add(&original).unwrap();
    set.add(&SourceText::from_utf8("own.weight")).unwrap();
    let Options::List(snapshot) = set.snapshot().unwrap() else {
        panic!("list")
    };
    assert!(snapshot.borrow()[0].same_object(&original));
    assert!(set.discard(&original).unwrap());
    assert!(snapshot.borrow()[0].same_object(&original));
    assert!(matches!(
        set.add(&Options::Atom(V::None)),
        Err(imagine_inference::native::Error::BindingGap { .. })
    ));
}
