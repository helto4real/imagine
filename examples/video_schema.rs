//! Emit video DTO schemas without starting the server or loading a model.
fn main() {
    println!(
        "{}",
        serde_json::to_string_pretty(&imagine_server::video::planning::schemas()).unwrap()
    );
}
