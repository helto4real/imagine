//! Python's JSON wire conventions, retained when the reference rewrites a body.
use serde_json::Value;

pub fn dumps(value: &Value, ascii: bool, compact: bool) -> String {
    let comma = if compact { "," } else { ", " };
    let colon = if compact { ":" } else { ": " };
    match value {
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(|v| dumps(v, ascii, compact))
                .collect::<Vec<_>>()
                .join(comma)
        ),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(k, v)| format!(
                    "{}{colon}{}",
                    dumps(&Value::String(k.clone()), ascii, compact),
                    dumps(v, ascii, compact)
                ))
                .collect::<Vec<_>>()
                .join(comma)
        ),
        Value::String(s) if ascii => {
            let mut result = String::new();
            for c in serde_json::to_string(s).unwrap().chars() {
                if c.is_ascii() && c != '\u{7f}' {
                    result.push(c);
                } else {
                    for code in c.encode_utf16(&mut [0; 2]) {
                        result.push_str(&format!("\\u{code:04x}"));
                    }
                }
            }
            result
        }
        _ => value.to_string(),
    }
}

/// Model-profile JSON rejects duplicate keys at every depth, as the reference does.
pub fn unique(bytes: &[u8]) -> anyhow::Result<Value> {
    use serde::{
        Deserialize, Deserializer,
        de::{self, MapAccess, SeqAccess, Visitor},
    };
    struct Unique(Value);
    impl<'de> Deserialize<'de> for Unique {
        fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
            struct V;
            impl<'de> Visitor<'de> for V {
                type Value = Unique;
                fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                    f.write_str("a JSON value")
                }
                fn visit_bool<E: de::Error>(self, v: bool) -> Result<Unique, E> {
                    Ok(Unique(v.into()))
                }
                fn visit_i64<E: de::Error>(self, v: i64) -> Result<Unique, E> {
                    Ok(Unique(v.into()))
                }
                fn visit_u64<E: de::Error>(self, v: u64) -> Result<Unique, E> {
                    Ok(Unique(v.into()))
                }
                fn visit_f64<E: de::Error>(self, v: f64) -> Result<Unique, E> {
                    Ok(Unique(serde_json::json!(v)))
                }
                fn visit_str<E: de::Error>(self, v: &str) -> Result<Unique, E> {
                    Ok(Unique(v.into()))
                }
                fn visit_none<E: de::Error>(self) -> Result<Unique, E> {
                    Ok(Unique(Value::Null))
                }
                fn visit_unit<E: de::Error>(self) -> Result<Unique, E> {
                    Ok(Unique(Value::Null))
                }
                fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<Unique, A::Error> {
                    let mut v = Vec::new();
                    while let Some(Unique(x)) = a.next_element()? {
                        v.push(x)
                    }
                    Ok(Unique(Value::Array(v)))
                }
                fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> Result<Unique, A::Error> {
                    let mut v = serde_json::Map::new();
                    while let Some(k) = a.next_key::<String>()? {
                        if v.contains_key(&k) {
                            return Err(de::Error::custom(format!("duplicate key {k}")));
                        }
                        let Unique(x) = a.next_value()?;
                        v.insert(k, x);
                    }
                    Ok(Unique(Value::Object(v)))
                }
            }
            d.deserialize_any(V)
        }
    }
    serde_json::from_slice::<Unique>(bytes)
        .map(|v| v.0)
        .map_err(|e| {
            let message = e.to_string();
            anyhow::anyhow!(if message.starts_with("duplicate key ") {
                message
                    .rsplit_once(" at line ")
                    .map_or(message.clone(), |(m, _)| m.to_string())
            } else {
                message
            })
        })
}
