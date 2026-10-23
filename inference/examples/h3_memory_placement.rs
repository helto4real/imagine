//! Allocation-free static-placement check using the real H3 module inventory.
use imagine_inference::{
    h3_product::protocol,
    memory::policy::MemoryPolicy,
    models::qwen_image21::{OffloadModule, StaticPlacement},
};
#[path = "../src/h3_product/memory_budget.rs"]
mod memory_budget;

fn main() {
    let inventory: serde_json::Value =
        serde_json::from_str(include_str!("../resources/h3-component-modules.json")).unwrap();
    let modules: Vec<OffloadModule> =
        serde_json::from_value(inventory["models"]["dit"]["modules"].clone()).unwrap();
    let p = MemoryPolicy::default();
    let mut rows = Vec::new();
    for shape in [
        protocol::Shape::default(),
        protocol::Shape {
            width: 768,
            height: 512,
            frames: 124,
            fps: 24,
        },
    ] {
        let (requested, minimum) = memory_budget::dit(shape);
        for free_gib in [0., 2., 4., 8., 12., 16.] {
            let free = free_gib * 1073741824.;
            let placements = [None, Some(minimum)].map(|min| {
                let (_, minimum) = p.requirements(requested, min);
                StaticPlacement::load(&modules, p.lowvram_budget(free, 0., minimum), false)
            });
            let bytes = placements.each_ref().map(|placement| {
                modules
                    .iter()
                    .filter(|m| placement.resident.contains(&m.name))
                    .map(|m| m.module_bytes)
                    .sum::<u64>()
            });
            assert!(bytes[1] >= bytes[0]);
            if shape == protocol::Shape::default() {
                assert_eq!(placements[0].resident, placements[1].resident);
                assert_eq!(placements[0].buffer_bytes, placements[1].buffer_bytes);
            }
            if shape.frames == 124 && free_gib == 16. {
                assert!(bytes[1] > bytes[0]);
                assert!(placements[1].buffer_bytes > 0);
            }
            rows.push(serde_json::json!({"shape":shape,"synthetic_free_gib":free_gib,
                "resident_module_bytes":bytes,"cast_buffer_bytes":placements.each_ref().map(|p|p.buffer_bytes)}));
        }
    }
    println!(
        "{}",
        serde_json::json!({"result":"passed","scope":"synthetic placement; no model tensors or GPU execution","cases":rows})
    );
}
