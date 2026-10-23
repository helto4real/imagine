use imagine_inference::{
    patching::options::{GeneralDictionary, NumericKeyBindings, Options, SourceNumber as N},
    python_hash,
};
use serde_json::{Value, json};
fn hex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}
fn float(s: &str) -> f64 {
    f64::from_le_bytes(hex(s).try_into().unwrap())
}
#[test]
fn native_numeric_objects_match_cpython_cache_payload_truth_exact_integer_and_key_hashes() {
    let f: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-numeric-objects.json")).unwrap();
    let values = objects(&f);
    for (i, (d, v)) in f["definitions"]
        .as_array()
        .unwrap()
        .iter()
        .zip(&values)
        .enumerate()
    {
        let n = N::get(v).unwrap();
        assert_eq!(n.is_exact_integer(), d["exact_int"].as_bool().unwrap());
        assert_eq!(n.truth(), d["truth"].as_bool().unwrap());
        let owner = d["owner_class"].as_u64().unwrap() as usize;
        assert!(v.same_object(&values[owner]), "owner cache {i}/{owner}");
        let expected = match d["kind"].as_str().unwrap() {
            "float" => {
                let x = float(d["bits"].as_str().unwrap());
                assert_eq!(n.float_bits(), Some(x.to_bits()));
                if x.is_nan() {
                    python_hash::identity_hash_address(n.identity_token() as usize)
                } else {
                    d["hash"].as_i64().unwrap()
                }
            }
            "complex" => {
                let r = float(d["real"].as_str().unwrap());
                let im = float(d["imaginary"].as_str().unwrap());
                assert_eq!(n.complex_bits(), Some((r.to_bits(), im.to_bits())));
                if r.is_nan() || im.is_nan() {
                    python_hash::hash_complex(
                        r,
                        im,
                        python_hash::identity_hash_address(n.identity_token() as usize),
                    )
                } else {
                    d["hash"].as_i64().unwrap()
                }
            }
            _ => d["hash"].as_i64().unwrap(),
        };
        assert_eq!(n.hash(), expected, "numeric key hash {i}");
    }
    for c in f["comparisons"].as_array().unwrap() {
        let a = c[0].as_u64().unwrap() as usize;
        let b = c[1].as_u64().unwrap() as usize;
        assert_eq!(
            N::get(&values[a])
                .unwrap()
                .equal(N::get(&values[b]).unwrap()),
            c[2].as_bool().unwrap(),
            "numeric comparison {a}/{b}"
        );
    }
}
fn objects(f: &Value) -> Vec<Options> {
    f["definitions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| match d["kind"].as_str().unwrap() {
            "int" => N::integer(&hex(d["bytes"].as_str().unwrap())),
            "bool" => N::boolean(d["value"].as_bool().unwrap()),
            "float" => N::float(float(d["bits"].as_str().unwrap())),
            "complex" => N::complex_parts(
                float(d["real"].as_str().unwrap()),
                float(d["imaginary"].as_str().unwrap()),
            ),
            _ => panic!("fixture kind"),
        })
        .collect()
}
#[test]
fn native_numeric_dictionary_retains_original_first_key_owners_cross_type_collisions_and_distinct_nan_keys()
 {
    let f: Value =
        serde_json::from_str(include_str!("../../fixtures/shared-numeric-objects.json")).unwrap();
    let values = objects(&f);
    let bindings = NumericKeyBindings;
    let map = GeneralDictionary::new();
    let mut index = 0;
    for step in f["finite_history"].as_array().unwrap() {
        let last = step["index"].as_u64().unwrap() as usize;
        while index <= last {
            let n = N::get(&values[index]).unwrap();
            let is_nan = n.float_bits().is_some_and(|b| f64::from_bits(b).is_nan())
                || n.complex_bits()
                    .is_some_and(|(r, i)| f64::from_bits(r).is_nan() || f64::from_bits(i).is_nan());
            if !is_nan {
                map.set(values[index].clone(), N::from_i64(index as i64), &bindings)
                    .unwrap();
            }
            index += 1;
        }
        let entries = map
            .entries()
            .iter()
            .map(|(key, value)| {
                let owner = values.iter().position(|v| v.same_object(key)).unwrap();
                let group = f["definitions"][owner]["owner_class"].as_u64().unwrap();
                let v = value.snapshot().unwrap();
                let imagine_inference::execution_value::Value::Int(v) = v else {
                    panic!("value")
                };
                json!([group, v])
            })
            .collect::<Vec<_>>();
        assert_eq!(json!(entries), step["entries"]);
    }
    let a = N::float(f64::NAN);
    let b = N::float(f64::NAN);
    let nans = GeneralDictionary::new();
    nans.set(a.clone(), N::from_i64(11), &bindings).unwrap();
    nans.set(b.clone(), N::from_i64(22), &bindings).unwrap();
    assert_eq!(
        nans.len(),
        f["nan_identity"]["length"].as_u64().unwrap() as usize
    );
    for (key, expected) in [(&a, 11), (&b, 22)] {
        let value = nans.get(key, &bindings).unwrap().unwrap();
        assert!(
            N::get(&value)
                .unwrap()
                .equal(N::get(&N::from_i64(expected)).unwrap())
        );
    }
    assert_eq!(
        N::get(&a).unwrap().equal(N::get(&a).unwrap()),
        f["nan_identity"]["equality"].as_bool().unwrap()
    );
}
