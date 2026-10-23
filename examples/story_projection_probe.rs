//! Pure source-contract oracle endpoint, stdin/stdout only. No runtime or media.
use std::io::Read;
fn main() {
    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text).unwrap();
    let value: serde_json::Value = serde_json::from_str(&text).unwrap();
    let images: Vec<imagine_video_contract::authoring::StoryImage> =
        serde_json::from_value(value["images"].clone()).unwrap();
    let result = imagine_server::video::authoring::engine::outline_with_images(
        &value["response"].to_string(),
        &images,
        value["dialogue"].as_bool().unwrap_or(false),
    );
    println!(
        "{}",
        match result {
            Ok(board) => serde_json::json!({"board":board}),
            Err(code) => serde_json::json!({"error":code}),
        }
    );
}
