use std::io::{self, BufRead};
fn main() {
    for line in io::stdin().lock().lines() {
        let v: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
        let source = std::fs::read(v["source"].as_str().unwrap()).unwrap();
        let reference = std::fs::read(v["reference"].as_str().unwrap()).unwrap();
        let bytes = imagine_server::color_match::apply(
            &source,
            &reference,
            v["color"].as_f64().unwrap() as f32,
            v["brightness"].as_f64().unwrap() as f32,
            v["contrast"].as_f64().unwrap() as f32,
        )
        .unwrap();
        std::fs::write(v["output"].as_str().unwrap(), bytes).unwrap();
        println!("ok");
    }
}
