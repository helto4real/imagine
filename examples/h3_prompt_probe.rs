//! Pure source compiler adapter. No models, providers, runtime data or media IO.
use imagine_server::video::adapters::h3::prompt;
fn main() {
    let input: serde_json::Value = serde_json::from_reader(std::io::stdin()).unwrap();
    let result = if let Some(cases) = input.as_array() {
        serde_json::Value::Array(cases.iter().map(evaluate).collect())
    } else {
        evaluate(&input)
    };
    println!("{}", result);
}
fn evaluate(input: &serde_json::Value) -> serde_json::Value {
    match input["operation"].as_str().unwrap_or("i2v") {
        "soundtrack_projection" => {
            let p: imagine_server::video::contract::VideoProject =
                serde_json::from_value(input["project"].clone()).unwrap();
            serde_json::to_value(imagine_video_contract::audio::references::soundtrack_clips(
                &p,
            ))
            .unwrap()
        }
        "protection" => {
            let p = imagine_server::video::authoring::references::Protected::new(
                input["source"].as_str().unwrap(),
            );
            let r = p.restore(input["response"].as_str().unwrap());
            serde_json::json!({"protected":p.text,"references":p.references,"restored":r.as_ref().ok().map(|x|&x.0),"introduced":r.as_ref().ok().map(|x|x.1)})
        }
        "enhancement_user" => {
            serde_json::json!({"prompt":imagine_server::video::authoring::references::enhancement_user_prompt(input["section"].as_bool().unwrap(),input["context"].as_str().unwrap(),input["instruction"].as_str().unwrap(),input["source"].as_str().unwrap(),input["images"].as_bool().unwrap(),input["opening"].as_bool().unwrap_or(false))})
        }
        "projection" => {
            let p: imagine_server::video::contract::VideoProject =
                serde_json::from_value(input["project"].clone()).unwrap();
            serde_json::to_value(
                imagine_server::video::adapters::h3::bindings::compile(&p, &p.scenes[0]).unwrap(),
            )
            .unwrap()
        }
        "dialogue" => {
            let maps: prompt::dialogue::Maps =
                serde_json::from_value(input["maps"].clone()).unwrap();
            let text = input["text"].as_str().unwrap();
            serde_json::json!({"prompt":prompt::dialogue::compile(text,&maps),"cues":prompt::dialogue::parse(text)})
        }
        "reference" => serde_json::to_value(prompt::compile_reference(
            &serde_json::from_value(input["input"].clone()).unwrap(),
        ))
        .unwrap(),
        _ => serde_json::to_value(prompt::compile_i2v(
            &serde_json::from_value(input["input"].clone()).unwrap(),
        ))
        .unwrap(),
    }
}
