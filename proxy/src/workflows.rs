//! Operator-owned workflow contracts. Graphs remain the source of generation settings.
use anyhow::{Result, anyhow, bail};
use serde_json::{Map, Value, json};
use std::{
    collections::{BTreeMap, HashSet},
    path::{Path, PathBuf},
};
const SCHEMA: &[&str] = &[
    "type",
    "description",
    "default",
    "enum",
    "minimum",
    "maximum",
    "minLength",
    "maxLength",
    "multipleOf",
];
#[derive(Clone)]
pub struct Workflow {
    pub manifest: Value,
    pub graph: Value,
    pub guidance: Option<String>,
    pub modes: Option<Value>,
}
#[derive(Clone, Default)]
pub struct Catalog {
    pub workflows: BTreeMap<String, Workflow>,
}
fn inputs<'a>(graph: &'a Value, node: &str) -> Result<&'a Map<String, Value>> {
    graph[node]["inputs"]
        .as_object()
        .ok_or_else(|| anyhow!("configured node or its inputs are missing"))
}
fn inside(dir: &Path, name: &str, limit: usize) -> Result<Vec<u8>> {
    let root = dir.canonicalize()?;
    let path = dir.join(name).canonicalize()?;
    if !path.starts_with(root) || path.metadata()?.len() > limit as u64 {
        bail!("workflow path or size is invalid")
    };
    Ok(std::fs::read(path)?)
}
pub fn check(spec: &Value, value: &Value) -> Result<()> {
    let valid = match spec["type"].as_str() {
        Some("string") => value.is_string(),
        Some("boolean") => value.is_boolean(),
        Some("integer") => value.is_i64() || value.is_u64(),
        Some("number") => value.is_number(),
        _ => false,
    };
    if !valid {
        bail!("parameter has the wrong type")
    }
    if let Some(n) = value.as_f64() {
        if spec["minimum"].as_f64().is_some_and(|v| n < v) {
            bail!("parameter is below its minimum")
        }
        if spec["maximum"].as_f64().is_some_and(|v| n > v) {
            bail!("parameter exceeds its maximum")
        }
        if spec["multipleOf"].as_f64().is_some_and(|v| n % v != 0.) {
            bail!("parameter must be a multiple of its configured step")
        }
    }
    if let Some(s) = value.as_str() {
        let n = s.chars().count() as u64;
        if n < spec["minLength"].as_u64().unwrap_or(0)
            || n > spec["maxLength"].as_u64().unwrap_or(65536)
        {
            bail!("parameter has an invalid length")
        }
    }
    if let Some(choices) = spec["enum"].as_array()
        && !choices.contains(value)
    {
        bail!("parameter is not an allowed choice")
    }
    Ok(())
}
impl Workflow {
    pub fn id(&self) -> &str {
        self.manifest["id"].as_str().unwrap()
    }
    pub fn summary(&self) -> Value {
        json!({"id":self.manifest["id"],"name":self.manifest["name"],"description":self.manifest["description"]})
    }
    pub fn parameters(&self) -> Map<String, Value> {
        let mut p = self.manifest["parameters"].as_object().unwrap().clone();
        if let Some(m) = &self.modes {
            let modes = m["modes"].as_object().unwrap();
            let desc = modes
                .iter()
                .map(|(name, v)| format!("{name}: {}", v["description"].as_str().unwrap()))
                .collect::<Vec<_>>()
                .join(" ");
            p.insert("mode".into(),json!({"type":"string","default":m["default"],"enum":modes.keys().collect::<Vec<_>>(),"description":format!("Generation mode. {desc}")}));
        }
        if self.manifest.get("nsfw_loras").is_some() {
            p.insert("nsfw".into(),json!({"type":"boolean","default":false,"description":include_str!("../reference/nsfw-description.txt")}));
        }
        p
    }
    pub fn describe(&self) -> Value {
        let params = self.parameters();
        let properties: Map<String, Value> = params
            .iter()
            .map(|(n, s)| {
                (
                    n.clone(),
                    Value::Object(
                        s.as_object()
                            .unwrap()
                            .iter()
                            .filter(|(k, _)| SCHEMA.contains(&k.as_str()))
                            .map(|(k, v)| (k.clone(), v.clone()))
                            .collect(),
                    ),
                )
            })
            .collect();
        let mut r = self.summary();
        r["parameters"] = json!({"type":"object","properties":properties,"additionalProperties":false,"required":params.iter().filter(|(_,v)|v["required"]==true).map(|(k,_)|k).collect::<Vec<_>>()});
        r["result_type"] = json!("local_image_files");
        if let Some(g) = &self.guidance {
            r["prompt_guidance"] = json!(g)
        }
        if let Some(s) = self.manifest.get("references") {
            let min = s["minItems"].as_u64().unwrap_or(0);
            r["references"] = json!({"minItems":min,"maxItems":s["inputs"].as_array().unwrap().len(),"description":format!("Register images first; ordered image IDs map to <image1>, <image2>, etc. {}",if min==0{"An empty list generates without references."}else{"This workflow requires reference images."})});
        }
        r
    }
    pub fn normalize(&self, p: &Value) -> Result<Value> {
        let specs = self.parameters();
        let p = p
            .as_object()
            .filter(|p| p.keys().all(|k| specs.contains_key(k)))
            .ok_or_else(|| anyhow!("unknown parameter or invalid parameter object"))?;
        let mut result = Map::new();
        for (n, s) in specs {
            let v = if let Some(v) = p.get(&n) {
                v.clone()
            } else if s["required"] == true {
                bail!("missing required parameter: {n}")
            } else if s["random_default"] == true {
                continue;
            } else {
                s["default"].clone()
            };
            check(&s, &v).map_err(|e| anyhow!("{n}: {e}"))?;
            result.insert(n, v);
        }
        Ok(Value::Object(result))
    }
    pub fn references(&self, count: usize) -> Result<()> {
        let s = &self.manifest["references"];
        let min = s["minItems"].as_u64().unwrap_or(0) as usize;
        let max = s["inputs"].as_array().map_or(0, Vec::len);
        if count < min {
            bail!("workflow requires at least {min} reference image(s)")
        }
        if count > max {
            bail!("workflow does not support this number of references")
        }
        Ok(())
    }
    pub fn seed(&self, p: &mut Value) {
        let s = &self.manifest["parameters"]["seed"];
        if s["random_default"] == true && p.get("seed").is_none() {
            let min = s["minimum"].as_u64().unwrap();
            let span = s["maximum"].as_u64().unwrap() - min + 1;
            let limit = u64::MAX - u64::MAX % span;
            let n = loop {
                let n = {
                    let uuid = uuid::Uuid::new_v4();
                    let b = uuid.as_bytes();
                    u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[9], b[10]])
                };
                if n < limit {
                    break n % span;
                }
            };
            p["seed"] = json!(min + n)
        }
    }
    pub fn build(&self, p: &Value) -> Result<Value> {
        let mut g = self.graph.clone();
        if let Some(m) = &self.modes {
            let name = p
                .get("mode")
                .unwrap_or(&m["default"])
                .as_str()
                .ok_or_else(|| anyhow!("unknown generation mode"))?;
            let mode = &m["modes"][name];
            if mode.is_null() {
                bail!("unknown generation mode")
            };
            for (op, patch) in mode.as_object().unwrap() {
                if op == "set" {
                    for (n, values) in patch.as_object().unwrap() {
                        for (k, v) in values.as_object().unwrap() {
                            g[n]["inputs"][k] = v.clone()
                        }
                    }
                } else if op == "remove" {
                    for (n, values) in patch.as_object().unwrap() {
                        for k in values.as_array().unwrap() {
                            g[n]["inputs"]
                                .as_object_mut()
                                .unwrap()
                                .remove(k.as_str().unwrap());
                        }
                    }
                }
            }
        }
        if let Some(s) = self.manifest.get("nsfw_loras") {
            let node = s["node"].as_str().unwrap();
            for name in s["inputs"].as_array().unwrap() {
                g[node]["inputs"][name.as_str().unwrap()]["on"] =
                    json!(p["nsfw"].as_bool().unwrap_or(false));
            }
        }
        for (n, v) in p.as_object().unwrap() {
            if (n == "mode" && self.modes.is_some())
                || (n == "nsfw" && self.manifest.get("nsfw_loras").is_some())
            {
                continue;
            }
            let t = &self.manifest["parameters"][n]["target"];
            g[t["node"].as_str().unwrap()]["inputs"][t["input"].as_str().unwrap()] = v.clone();
        }
        Ok(g)
    }
    pub fn load(manifest: Value, dir: &Path, max: usize) -> Result<Self> {
        for k in ["id", "name", "description", "workflow"] {
            if manifest[k].as_str().is_none_or(|s| s.trim().is_empty()) {
                bail!("manifest needs id, name, description and workflow")
            }
        }
        let graph: Value =
            serde_json::from_slice(&inside(dir, manifest["workflow"].as_str().unwrap(), max)?)?;
        if !graph.is_object() {
            bail!("API workflow must be an object")
        }
        let p = manifest["parameters"]
            .as_object()
            .ok_or_else(|| anyhow!("manifest needs parameters and outputs"))?;
        let outputs = manifest["outputs"]
            .as_array()
            .filter(|v| !v.is_empty())
            .ok_or_else(|| anyhow!("manifest needs parameters and outputs"))?;
        let mut mapped = HashSet::new();
        let mut controlled = HashSet::new();
        if let Some(s) = manifest.get("nsfw_loras") {
            if p.contains_key("nsfw") {
                bail!("nsfw is reserved when NSFW LoRAs are configured")
            }
            let node = s["node"]
                .as_str()
                .ok_or_else(|| anyhow!("invalid NSFW LoRA targets"))?;
            let names = s["inputs"]
                .as_array()
                .filter(|v| !v.is_empty())
                .ok_or_else(|| anyhow!("invalid NSFW LoRA targets"))?;
            if s.as_object().map_or(0, Map::len) != 2 {
                bail!("invalid NSFW LoRA targets")
            }
            let i = inputs(&graph, node)?;
            if graph[node]["class_type"] != "AIOLoraConfiguration" {
                bail!("NSFW LoRAs require AIOLoraConfiguration")
            }
            for n in names {
                let n = n
                    .as_str()
                    .filter(|n| {
                        n.strip_prefix("lora_")
                            .is_some_and(|v| !v.is_empty() && v.chars().all(|c| c.is_ascii_digit()))
                    })
                    .ok_or_else(|| anyhow!("invalid NSFW LoRA targets"))?;
                if !controlled.insert((node.to_string(), n.to_string())) {
                    bail!("invalid NSFW LoRA targets")
                };
                let v = &i[n];
                if !v["on"].is_boolean()
                    || !v["lora"]
                        .as_str()
                        .is_some_and(|s| !s.trim().is_empty() && s != "None")
                {
                    bail!("configured NSFW LoRA row is missing or invalid")
                }
            }
        }
        for (n, s) in p {
            let obj = s
                .as_object()
                .ok_or_else(|| anyhow!("invalid parameter definition"))?;
            if n.is_empty()
                || obj.keys().any(|k| {
                    !SCHEMA.contains(&k.as_str())
                        && !["target", "required", "private", "random_default"]
                            .contains(&k.as_str())
                })
                || !["string", "integer", "number", "boolean"]
                    .contains(&s["type"].as_str().unwrap_or(""))
            {
                bail!("invalid parameter definition")
            }
            for k in ["required", "private", "random_default"] {
                if s.get(k).is_some_and(|v| !v.is_boolean()) {
                    bail!("required/private/random_default must be booleans")
                }
            }
            for k in ["minimum", "maximum", "minLength", "maxLength"] {
                if s.get(k).is_some_and(|v| !v.is_number()) {
                    bail!("invalid parameter constraint")
                }
            }
            for k in ["minLength", "maxLength"] {
                if s.get(k)
                    .is_some_and(|v| v.as_u64().is_none() || s["type"] != "string")
                {
                    bail!("invalid string constraint")
                }
            }
            if (s.get("minimum").is_some() || s.get("maximum").is_some())
                && s["type"] != "number"
                && s["type"] != "integer"
            {
                bail!("numeric constraints require a number")
            }
            if s.get("multipleOf")
                .is_some_and(|v| v.as_u64().is_none_or(|n| n == 0) || s["type"] != "integer")
            {
                bail!("multipleOf requires an integer parameter and positive step")
            }
            for (lo, hi) in [("minimum", "maximum"), ("minLength", "maxLength")] {
                if let (Some(a), Some(b)) = (s[lo].as_f64(), s[hi].as_f64())
                    && a > b
                {
                    bail!("inverted parameter constraints")
                }
            }
            if let Some(v) = s.get("enum") {
                let values = v
                    .as_array()
                    .filter(|v| !v.is_empty())
                    .ok_or_else(|| anyhow!("enum must contain choices"))?;
                for v in values {
                    check(s, v)?
                }
            }
            if let Some(v) = s.get("default") {
                check(s, v)?;
                if s["private"] == true {
                    bail!("private parameters must be required, without defaults")
                }
            }
            if s["random_default"] == true
                && (n != "seed"
                    || s["type"] != "integer"
                    || s.get("default").is_some()
                    || s["required"] == true
                    || s["private"] == true
                    || s["minimum"].as_u64().is_none()
                    || s["maximum"]
                        .as_u64()
                        .is_none_or(|v| v > 9_007_199_254_740_991)
                    || s.get("enum").is_some()
                    || s.get("multipleOf").is_some())
            {
                bail!("random_default requires an optional public bounded seed")
            }
            if s["required"] != true && s.get("default").is_none() && s["random_default"] != true {
                bail!("optional parameters need a default")
            }
            let t = &s["target"];
            let node = t["node"]
                .as_str()
                .ok_or_else(|| anyhow!("invalid parameter target"))?;
            let field = t["input"]
                .as_str()
                .ok_or_else(|| anyhow!("invalid parameter target"))?;
            if !inputs(&graph, node)?.contains_key(field) {
                bail!("configured input is missing")
            };
            let pair = (node.to_string(), field.to_string());
            if controlled.contains(&pair) {
                bail!("parameter conflicts with NSFW LoRA controls")
            }
            if !mapped.insert(pair) {
                bail!("parameters cannot target the same input")
            }
        }
        let mut protected = HashSet::new();
        for o in outputs {
            let n = o["node"]
                .as_str()
                .ok_or_else(|| anyhow!("invalid output definition"))?;
            if !["comfy_images", "helto_saved_images"]
                .contains(&o["adapter"].as_str().unwrap_or(""))
            {
                bail!("invalid output definition")
            };
            if !protected.insert(n.to_string()) || !graph[n].is_object() {
                bail!("output node is missing or duplicated")
            }
            if o["adapter"] == "helto_saved_images" {
                if graph[n]["class_type"] != "HeltoSaveImageAdvanced" {
                    bail!("Helto adapter requires HeltoSaveImageAdvanced")
                };
                if mapped
                    .iter()
                    .chain(controlled.iter())
                    .any(|(node, _)| node == n)
                {
                    bail!("Helto output controls cannot be agent parameters")
                };
                let i = &graph[n]["inputs"];
                for key in [
                    "reference_only",
                    "privacy_mode",
                    "save_image",
                    "pause_mode",
                    "use_alternative_folder",
                    "use_date_folder",
                ] {
                    if !i[key].is_boolean() {
                        bail!("Helto output requires boolean save/privacy/reference controls")
                    }
                }
                if i["save_image"] != true || i["pause_mode"] != false {
                    bail!("Helto output requires save_image=true and pause_mode=false")
                };
                let folder = if i["use_alternative_folder"] == true {
                    "alternative_folder"
                } else {
                    "folder"
                };
                if !i[folder]
                    .as_str()
                    .is_some_and(|s| Path::new(s).is_absolute())
                {
                    bail!("Helto output requires an absolute destination folder")
                };
                if i.get("subfolder").is_some_and(|v| {
                    v.as_str().is_none_or(|s| {
                        Path::new(s).is_absolute()
                            || Path::new(s)
                                .components()
                                .any(|p| p == std::path::Component::ParentDir)
                    })
                }) {
                    bail!("Helto subfolder must remain inside the destination folder")
                }
            }
        }
        if let Some(r) = manifest.get("references").filter(|r| !r.is_null()) {
            let generator = r["generator"]
                .as_str()
                .ok_or_else(|| anyhow!("invalid reference mapping"))?;
            let loader = r["loader"]
                .as_str()
                .ok_or_else(|| anyhow!("invalid reference mapping"))?;
            let slots = r["inputs"]
                .as_array()
                .ok_or_else(|| anyhow!("invalid reference mapping"))?;
            if r["adapter"] != "helto_root_images" {
                bail!("invalid reference mapping")
            }
            let names = slots
                .iter()
                .filter_map(Value::as_str)
                .collect::<HashSet<_>>();
            if slots.is_empty()
                || slots.len() > 10
                || names.len() != slots.len()
                || names.contains("")
                || generator == loader
            {
                bail!("invalid reference inputs")
            }
            if r.get("minItems")
                .is_some_and(|v| v.as_u64().is_none_or(|v| v > slots.len() as u64))
            {
                bail!("invalid minimum reference count")
            };
            inputs(&graph, generator)?;
            let i = inputs(&graph, loader)?;
            if graph[loader]["class_type"] != "HeltoLoadImageFromRoot"
                || ["root", "subfolder", "filename"]
                    .iter()
                    .any(|k| !i.get(*k).is_some_and(Value::is_string))
            {
                bail!("reference adapter requires a HeltoLoadImageFromRoot template")
            }
            if mapped
                .iter()
                .chain(controlled.iter())
                .any(|(n, k)| n == loader || (n == generator && names.contains(k.as_str())))
            {
                bail!("reference controls cannot also be parameter targets")
            };
            protected.insert(loader.to_string());
            for k in names {
                mapped.insert((generator.to_string(), k.to_string()));
            }
        }
        let guidance = if let Some(name) = manifest.get("prompt_guidance") {
            Some(String::from_utf8(inside(
                dir,
                name.as_str()
                    .ok_or_else(|| anyhow!("prompt_guidance must name a local text file"))?,
                65536,
            )?)?)
        } else {
            None
        };
        let modes = if let Some(name) = manifest.get("modes") {
            if p.contains_key("mode") {
                bail!("mode is reserved when a modes file is configured")
            };
            let bytes = inside(
                dir,
                name.as_str()
                    .ok_or_else(|| anyhow!("modes must name a local YAML file"))?,
                max,
            )?;
            let m: Value =
                serde_yaml::from_slice(&bytes).map_err(|_| anyhow!("invalid mode YAML"))?;
            if m.as_object().map_or(0, Map::len) != 3
                || m["version"].as_u64() != Some(1)
                || !m["default"].is_string()
                || m["modes"].as_object().is_none_or(|v| {
                    v.is_empty() || !v.contains_key(m["default"].as_str().unwrap_or(""))
                })
            {
                bail!("invalid mode version, default or modes")
            }
            for mode in m["modes"].as_object().unwrap().values() {
                if mode.as_object().is_none_or(|v| {
                    v.keys()
                        .any(|k| !["description", "set", "remove"].contains(&k.as_str()))
                }) || mode["description"]
                    .as_str()
                    .is_none_or(|s| s.trim().is_empty())
                {
                    bail!("mode needs a description and optional set/remove patches")
                }
                for op in ["set", "remove"] {
                    if let Some(patch) = mode.get(op) {
                        let patch = patch
                            .as_object()
                            .ok_or_else(|| anyhow!("mode patches must map node IDs to inputs"))?;
                        for (n, v) in patch {
                            inputs(&graph, n).map_err(|_| {
                                anyhow!("configured mode node or its inputs are missing")
                            })?;
                            let keys: Vec<&str> = if op == "set" {
                                v.as_object()
                                    .ok_or_else(|| {
                                        anyhow!("mode set must map input names to values")
                                    })?
                                    .keys()
                                    .map(String::as_str)
                                    .collect()
                            } else {
                                v.as_array()
                                    .ok_or_else(|| {
                                        anyhow!("mode remove must list unique input names")
                                    })?
                                    .iter()
                                    .map(|v| {
                                        v.as_str().filter(|s| !s.is_empty()).ok_or_else(|| {
                                            anyhow!("mode remove must list unique input names")
                                        })
                                    })
                                    .collect::<Result<_>>()?
                            };
                            if keys.iter().collect::<HashSet<_>>().len() != keys.len() {
                                bail!("mode remove must list unique input names")
                            };
                            if protected.contains(n)
                                || keys.iter().any(|k| {
                                    mapped.contains(&(n.clone(), k.to_string()))
                                        || controlled.contains(&(n.clone(), k.to_string()))
                                })
                            {
                                bail!("mode conflicts with parameter or adapter controls")
                            };
                            if op == "remove"
                                && keys.iter().any(|k| mode["set"][n].get(*k).is_some())
                            {
                                bail!("mode cannot set and remove the same input")
                            }
                        }
                    }
                }
            }
            Some(m)
        } else {
            None
        };
        Ok(Self {
            manifest,
            graph,
            guidance,
            modes,
        })
    }
}
impl Catalog {
    pub fn load(dir: &str, max: usize) -> Result<Self> {
        let dir = Path::new(dir);
        if !dir.is_dir() {
            bail!("MCP workflow directory does not exist")
        };
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .ends_with(".manifest.json")
            })
            .collect();
        paths.sort();
        let mut seen = HashSet::new();
        let mut result = Self::default();
        for path in paths {
            let load = (|| {
                if path.metadata()?.len() > max as u64 {
                    bail!("manifest is too large")
                };
                let m: Value = serde_json::from_slice(&std::fs::read(&path)?)?;
                if let Some(id) = m["id"].as_str()
                    && !seen.insert(id.to_string())
                {
                    result.workflows.remove(id);
                    bail!("duplicate workflow id")
                }
                Workflow::load(m, dir, max)
            })();
            match load {
                Ok(w) => {
                    result.workflows.insert(w.id().into(), w);
                }
                Err(e) => crate::logging::log(
                    40,
                    "st_proxy.workflows",
                    format!(
                        "MCP workflow unavailable: manifest={} reason={e}",
                        path.file_name().unwrap().to_string_lossy()
                    ),
                ),
            }
        }
        Ok(result)
    }
    pub fn get(&self, id: &str) -> Result<&Workflow> {
        self.workflows
            .get(id)
            .ok_or_else(|| anyhow!("unknown or unavailable workflow"))
    }
}
