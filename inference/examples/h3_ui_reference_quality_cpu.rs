//! Actual Reference geometry for synthetic dimensions only; no tensor allocation.
use imagine_inference::h3_product::{protocol::Shape, references::geometry};
use serde_json::json;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cases = vec![];
    for (width, height) in [
        (33, 17),
        (4000, 3000),
        (8192, 512),
        (256, 4097),
        (320, 224),
        (1, 1),
    ] {
        for quality in ["Standard", "High", "Extra High"] {
            let (w, h, pw, ph) = geometry(width, height, quality, Shape::default())?;
            cases.push(json!({"input_width":width,"input_height":height,"quality":quality,"content_width":w,"content_height":h,"width":pw,"height":ph}));
        }
    }
    println!(
        "{}",
        json!({"cases":cases,"device":"cpu","model_runtime_initialized":false})
    );
    Ok(())
}
