//! Source-contract probe: input and output are synthetic test JSON, never runtime data.
use imagine_video_contract::prompt_catalog::{self as pc, Segment};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    io::{self, Read},
};
fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let request: Value = serde_json::from_str(&input).unwrap();
    let sources: BTreeMap<String, String> =
        serde_json::from_value(request["sources"].clone()).unwrap();
    // Canonical editor YAML for synthetic documents, independent of the sources.
    let serialized = request["serialize"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|v| json!({"text": pc::source_text(v).unwrap(), "form": pc::editor::form_save(v)}))
        .collect::<Vec<_>>();
    match pc::load(&sources) {
        Err(e) => println!(
            "{}",
            json!({"ok":false,"issue":e,"summary":e.to_string(),"serialized":serialized})
        ),
        Ok(catalog) => {
            fn normalized(nodes: &[pc::Container]) -> Value {
                Value::Array(nodes.iter().map(|n|{
                    let mut v=json!({"key":n.key,"label":n.label});
                    if !n.description.is_empty(){v["description"]=json!(n.description);}
                    if !n.groups.is_empty(){v["groups"]=normalized(&n.groups);}
                    if !n.choices.is_empty(){v["choices"]=Value::Array(n.choices.iter().map(|c|{
                        let segments=c.segments.iter().map(|s|match s {Segment::Text{text}=>json!({"text":text}),Segment::Placeholder{placeholder:p}=>json!({"placeholder":p})}).collect::<Vec<_>>();
                        let mut v=json!({"key":c.key,"aliases":c.aliases,"label":c.label,"append":c.append,"segments":segments});
                        if !c.description.is_empty(){v["description"]=json!(c.description);}
                        v
                    }).collect());}
                    v
                }).collect())
            }
            let queries=request["queries"].as_array().into_iter().flatten().filter_map(Value::as_str).map(|text|{
                let entries=pc::context(text,text.chars().count(),text.chars().count()).map(|c|catalog.completions(&c)).unwrap_or_default();
                json!({"text":text,"items":entries.iter().map(|e|json!({"path":e.path,"label":e.label,"detail":e.detail,"expansion":e.choice.as_ref().map(|c|c.expand(&BTreeMap::new()))})).collect::<Vec<_>>()})
            }).collect::<Vec<_>>();
            let composer = request["composer"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|r| {
                    let parsed = pc::composer::parse(r["text"].as_str().unwrap(), &r["previous"]);
                    json!({"parsed":parsed,"text":pc::composer::text(&parsed)})
                })
                .collect::<Vec<_>>();
            let duplicates = request["duplicates"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|c| {
                    let mut c = c.clone();
                    let index = pc::composer::duplicate(&mut c, 0).unwrap();
                    let v = &c["choices"][index];
                    json!({"key":v["key"],"label":v["label"],"aliases":v["aliases"]})
                })
                .collect::<Vec<_>>();
            println!(
                "{}",
                json!({"ok":true,"catalog":{"schema_version":catalog.schema_version,"namespaces":normalized(&catalog.namespaces)},"queries":queries,"composer":composer,"duplicates":duplicates,"serialized":serialized})
            );
        }
    }
}
