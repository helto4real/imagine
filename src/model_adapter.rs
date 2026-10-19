//! Explicit operation support and Qwen-specific prompt/reference mapping.
use crate::{
    character_sheet::CharacterSheet,
    contract::{BF16, JobRequest, MAX_SEED, MODEL},
};
use serde_json::{Value, json};

pub const SHEET_REVISION: &str = "qwen-character-sheet-v1";
const PORTRAIT: &str = include_str!("prompts/qwen-character-sheet-v1/portrait.txt");
const TURNAROUND: &str = include_str!("prompts/qwen-character-sheet-v1/turnaround.txt");

#[derive(Clone, Copy)]
pub enum SheetStrategy {
    QwenTwoPass,
}
pub fn sheet_strategy(model: &str) -> Result<SheetStrategy, &'static str> {
    match model {
        MODEL | BF16 => Ok(SheetStrategy::QwenTwoPass),
        _ => Err("unsupported_character_sheet_model"),
    }
}
pub fn sheet_capabilities() -> Value {
    json!({"strategy":SHEET_REVISION,"max_identity_references":8,"pose_guide":true,
        "layout":{"aspect_ratio":"16:9","output_width":{"min":512,"max":4096,"multiple":512},"portrait_fraction":0.45},
        "output_roles":["character_sheet","portrait","turnaround"],"live_validated":false})
}
pub fn next_seed(seed: u64) -> u64 {
    (seed + 1) & MAX_SEED
}
pub fn sheet_plan(request: &JobRequest, sheet: &CharacterSheet) -> Value {
    let sizes = sheet.layout.pass_sizes().expect("validated sheet");
    json!({"strategy":SHEET_REVISION,"prompt_revision":SHEET_REVISION,"composition_revision":"sheet-compose-v1",
        "pose_guide_revision":"sheet-pose-v1","output_width":sheet.layout.output_width,
        "output_height":sheet.layout.dimensions().1,"portrait_fraction":sheet.layout.portrait_fraction,
        "passes":[{"stage":"portrait","width":sizes[0].0,"height":sizes[0].1,"seed":request.seed,
            "reference_roles":roles(request.inputs.reference_images.len(),false)},
            {"stage":"turnaround","width":sizes[1].0,"height":sizes[1].1,"seed":request.seed.map(next_seed),
            "reference_roles":roles(request.inputs.reference_images.len(),true)}]})
}
fn roles(count: usize, turnaround: bool) -> Vec<&'static str> {
    let mut roles = vec!["identity reference of the same character"; count];
    if turnaround {
        roles.extend([
            "generated facial identity portrait",
            "pose and direction guide only",
        ]);
    }
    roles
}
pub fn sheet_prompt(sheet: &CharacterSheet, count: usize, turnaround: bool) -> String {
    let roles = roles(count, turnaround);
    let mut parts = vec![
        if turnaround { TURNAROUND } else { PORTRAIT }
            .trim()
            .to_owned(),
    ];
    if !roles.is_empty() {
        parts.push(
            roles
                .iter()
                .enumerate()
                .map(|(i, role)| format!("<image{}>: {role}.", i + 1))
                .collect::<Vec<_>>()
                .join(" "),
        );
        parts.push("All identity references show the same individual, not different people. Preserve their facial anatomy, hairstyle, skin, distinguishing features and natural body proportions.".into());
    }
    if !sheet.character_description.trim().is_empty() {
        parts.push(format!(
            "Character description: {}.",
            sheet.character_description.trim()
        ));
    }
    if turnaround {
        parts.push(if sheet.clothing_description.trim().is_empty() {
            "Preserve the outfit in the identity references if supplied; otherwise choose one coherent outfit.".into()
        } else { format!("Clothing and accessories: {}. Replace reference outfits with this outfit without changing physical identity.",sheet.clothing_description.trim()) });
    } else if count == 0 {
        parts.push("Create the person's facial identity from the description above. Body proportions are context only, not a request to show the body. Final composition: one face filling the image, complete head from crown to chin occupying 90-95% of image height, eyes looking straight into the camera, head upright. The bottom edge is just below the chin. Shoulders, chest, torso and waist remain outside the image even when described above.".into());
    }
    parts.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prompts_keep_clothing_out_of_portrait_and_reserve_internal_slots() {
        let sheet = CharacterSheet {
            character_description: "SYNTHETIC_IDENTITY".into(),
            clothing_description: "SYNTHETIC_CLOTHES".into(),
            ..Default::default()
        };
        let portrait = sheet_prompt(&sheet, 8, false);
        let body = sheet_prompt(&sheet, 8, true);
        assert!(portrait.contains("SYNTHETIC_IDENTITY"));
        assert!(!portrait.contains("SYNTHETIC_CLOTHES"));
        assert!(!portrait.contains("<image9>"));
        assert!(body.contains("<image9>: generated facial identity portrait"));
        assert!(body.contains("<image10>: pose and direction guide only"));
        assert!(body.contains("SYNTHETIC_CLOTHES"));
        assert!(sheet_prompt(&sheet, 0, false).contains("Body proportions are context only"));
        assert_eq!(next_seed(MAX_SEED), 0);
        assert_eq!(next_seed(42), 43);
        assert!(sheet_strategy(crate::contract::TURBO).is_err());
    }
}
