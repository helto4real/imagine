//! Synthetic loopback-only LLM for opt-in tests; never wired into product services.
use axum::{Json, Router, extract::State, routing::post};
use serde_json::{Value, json};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
#[derive(Default)]
pub struct Fixture {
    pub delay_ms: AtomicU64,
    pub requests: Mutex<Vec<Value>>,
    pub fail_second_once: AtomicBool,
    pub fail_inventory_once: AtomicBool,
    pub drop_reference_once: AtomicBool,
    pub introduce_reference_once: AtomicBool,
    pub fail_animate_once: AtomicBool,
    /// Answers wrapped in Markdown code fences, as local models often write them.
    pub fence_answers: AtomicBool,
}
pub async fn spawn() -> (String, tokio::task::JoinHandle<()>, Arc<Fixture>) {
    let state = Arc::new(Fixture::default());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/v1", listener.local_addr().unwrap());
    let app = Router::new()
        .route("/v1/chat/completions", post(reply))
        .route("/v1/api/generate", post(reply))
        .with_state(state.clone());
    let worker = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (url, worker, state)
}
async fn reply(State(state): State<Arc<Fixture>>, Json(body): Json<Value>) -> Json<Value> {
    state.requests.lock().unwrap().push(body.clone());
    let delay = state.delay_ms.load(Ordering::SeqCst);
    if delay > 0 {
        tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
    }
    let content = &body["messages"][1]["content"];
    let user = content
        .as_str()
        .or_else(|| {
            content
                .as_array()
                .and_then(|parts| parts.iter().find_map(|p| p["text"].as_str()))
        })
        .or_else(|| body["prompt"].as_str())
        .unwrap_or("");
    if user.contains("fixture_wait") {
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
    }
    let tag = |name: &str| {
        user.split(&format!("<{name}>\n"))
            .nth(1)
            .and_then(|s| s.split(&format!("\n</{name}>")).next())
            .and_then(|s| serde_json::from_str::<Value>(s).ok())
            .unwrap_or(Value::Null)
    };
    // One labelled row of a Director message: `Label:\n<json>`.
    let row = |name: &str| {
        user.split(&format!("\n\n{name}:\n"))
            .nth(1)
            .and_then(|s| s.split("\n\n").next())
            .and_then(|s| serde_json::from_str::<Value>(s).ok())
            .unwrap_or(Value::Null)
    };
    let text = if user.starts_with("<scene_context>\n") {
        if state.fail_animate_once.swap(false, Ordering::SeqCst) {
            json!({"description":"Own incomplete answer"}).to_string()
        } else {
            json!({"description":"Own animated harbour","prompt":"","ambient_sound":"Own gulls"})
                .to_string()
        }
    } else if user.contains("\n\nStable image ID:\n") {
        if state.fail_inventory_once.swap(false, Ordering::SeqCst) {
            String::new()
        } else {
            "Own factual inventory: a lighthouse and wet rocks under a blue sky.".into()
        }
    } else if let Some(text) = user.split("<scene_json>\n").nth(1) {
        let mut scene: Value =
            serde_json::from_str(text.split("\n</scene_json>").next().unwrap()).unwrap();
        if matches!(scene["id"].as_str(), Some("story_scene_1" | "scene_002"))
            && state.fail_second_once.swap(false, Ordering::SeqCst)
        {
            "{invalid fixture JSON".into()
        } else {
            scene["description"] = json!(format!(
                "Expanded {}",
                scene["description"].as_str().unwrap()
            ));
            for s in scene["sections"].as_array_mut().unwrap() {
                s["prompt"] = json!(format!("Expanded {}", s["prompt"].as_str().unwrap()));
                if let Some(participants) = s["dialogue_intent"]["participants"].as_array() {
                    let a = participants[0].as_str().unwrap();
                    let b = participants[1].as_str().unwrap();
                    s["prompt"] = json!(format!(
                        "{} {a} says [Swedish] \"Hej!\" {b} replies [Swedish] \"Hej igen!\"",
                        s["prompt"].as_str().unwrap()
                    ));
                }
            }
            scene.to_string()
        }
    } else if user.contains("\n\nMinimal JSON shape:\n") {
        let mut board = json!({"scenes":[{"title":"Arrival","summary":"Approach the lighthouse","description":"The keeper walks across the rocks.","ambient_sound":"Waves","non_diegetic_music":"Own score","weight":1,"boundary_from_previous":"hard_cut","setting_reference_matches":{},"sections":[{"prompt":"The keeper approaches the lighthouse.","weight":1}]},{"title":"Signal","summary":"Light the beacon","description":"The keeper lights the beacon.","ambient_sound":"Wind","non_diegetic_music":"","weight":2,"boundary_from_previous":"continuous_shot","setting_reference_matches":{},"sections":[{"prompt":"The keeper switches on the beacon.","weight":1}]}],"visual_placements":[]});
        let images = row("Image inventory");
        for scene in board["scenes"].as_array_mut().unwrap() {
            for image in images
                .as_array()
                .into_iter()
                .flatten()
                .filter(|i| i["usage"] == "story_reference" && i["reuse"] == true)
            {
                scene["setting_reference_matches"][image["id"].as_str().unwrap()] = json!(true);
            }
        }
        board["visual_placements"]=json!(row("Required image order and visual beats").as_array().into_iter().flatten().map(|b|json!({"beat_key":b["beat_key"],"scene_ordinal":1,"section_ordinal":1,"semantic_role":if b["type"]=="required_moment"{"depicted_moment"}else{"state_transition"}})).collect::<Vec<_>>());
        let opening = tag("opening_appearance_context");
        if let Some(characters) = opening["characters"].as_array() {
            board["opening_appearance"] = json!({"schema_version":"1.0","characters":characters.iter().map(|c|json!({"character_id":c["character_id"],"reference_tag":c["reference_tag"],"source":"latest_moment","exposure":"clothed","garments":["Own fixture coat"],"worn_items":[],"temporary_traits":[],"visible_body_state":[]})).collect::<Vec<_>>()});
        }
        let cast = tag("selected_cast");
        let system = body["messages"][0]["content"]
            .as_str()
            .or_else(|| body["system"].as_str())
            .unwrap_or("");
        if system.contains("visual-story and dialogue planner")
            && cast.as_array().is_some_and(|c| c.len() >= 2)
        {
            board["scenes"][0]["sections"][0]["dialogue_intent"] = json!({"participants":[cast[0]["reference_tag"],cast[1]["reference_tag"]],"purpose":"A brief greeting"});
        }
        board.to_string()
    } else if let Some(source) = user.split("Source text:\n").nth(1) {
        if state.drop_reference_once.swap(false, Ordering::SeqCst) {
            "Own fixture omitted the marker.".into()
        } else {
            format!(
                "Expanded {source}{}",
                if state.introduce_reference_once.swap(false, Ordering::SeqCst) {
                    " @new_reference"
                } else {
                    ""
                }
            )
        }
    } else {
        "The lighthouse keeper walks across wet rocks and waves toward the camera.".into()
    };
    let text = if state.fence_answers.load(Ordering::SeqCst) {
        let language = if text.starts_with('{') {
            "json"
        } else {
            "text"
        };
        format!("\n```{language}\n{text}\n```\n")
    } else {
        text
    };
    if body["messages"].is_array() {
        Json(json!({"choices":[{"finish_reason":"stop","message":{"content":text}}]}))
    } else {
        Json(json!({"done":true,"done_reason":"stop","response":text}))
    }
}
