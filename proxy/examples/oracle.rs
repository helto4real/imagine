use serde_json::{Value, json};
use std::io::{self, BufRead};
fn main() {
    for line in io::stdin().lock().lines() {
        let v: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let result = match v["op"].as_str().unwrap() {
            "tabby" => {
                let body = st_proxy::requests::tabby(v["body"].as_str().unwrap().as_bytes());
                json!(String::from_utf8(body).unwrap())
            }
            "prism" => {
                let mut o =
                    st_proxy::diagnostics::Prism::new(v["body"].as_str().unwrap().as_bytes())
                        .unwrap();
                o.start(
                    v["status"].as_u64().unwrap() as u16,
                    v["content_type"].as_str().unwrap(),
                );
                for chunk in v["chunks"].as_array().unwrap() {
                    o.feed(chunk.as_str().unwrap().as_bytes());
                }
                json!(o.finish(
                    v["complete"].as_bool().unwrap(),
                    v["connected"].as_bool().unwrap()
                ))
            }
            "profiles" => {
                let backends = st_proxy::registry::parse(st_proxy::registry::BUILTIN).unwrap();
                let engine = v["engine"].as_str().unwrap();
                // Engines without a profile-capable adapter keep their payload.
                let adapter = backends
                    .iter()
                    .find(|s| s.id == engine)
                    .map_or(st_proxy::registry::Adapter::Koboldcpp, |s| s.adapter);
                match st_proxy::profiles::Profiles::parse(v["profiles"].clone(), &backends) {
                    Ok(p) => p.apply(&v["payload"], engine, adapter, v["model"].as_str()),
                    Err(e) => json!({"error":e.to_string()}),
                }
            }
            "workflow" => {
                let c =
                    st_proxy::workflows::Catalog::load(v["directory"].as_str().unwrap(), 67108864)
                        .unwrap();
                let w = c.get(v["id"].as_str().unwrap()).unwrap();
                if let Some(params) = v.get("parameters") {
                    match w.normalize(params) {
                        Ok(p) => p,
                        Err(e) => json!({"error":e.to_string()}),
                    }
                } else {
                    w.describe()
                }
            }
            "route" => {
                json!({"kind":st_proxy::routes::classify(v["method"].as_str().unwrap(),v["path"].as_str().unwrap()).as_str(),"path":st_proxy::routes::upstream_path(v["path"].as_str().unwrap())})
            }
            "log" => json!(st_proxy::logging::format(
                v["timestamp"].as_str().unwrap(),
                v["level"].as_i64().unwrap() as i32,
                v["name"].as_str().unwrap(),
                v["message"].as_str().unwrap(),
                v["color"].as_bool().unwrap()
            )),
            "strata" => json!(st_proxy::logging::strata_line(v["line"].as_str().unwrap())),
            "colors" => json!(st_proxy::logging::colors_enabled(
                v["tty"].as_bool().unwrap(),
                v["no_color"].as_str(),
                v["term"].as_str()
            )),
            "config" => {
                let env = serde_json::from_value(v["env"].clone()).unwrap();
                let args: Vec<String> = serde_json::from_value(v["args"].clone()).unwrap();
                match st_proxy::config::Config::parse(args, &env) {
                    Ok((c, _)) => serde_json::to_value(c).unwrap(),
                    Err(e) => json!({"error":e.to_string()}),
                }
            }
            _ => panic!("unknown oracle operation"),
        };
        println!("{}", result);
    }
}
