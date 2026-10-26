//! Explicit operation support and model-specific prompt/reference mapping.
use crate::{
    character_sheet::CharacterSheet,
    contract::{BF16, JobRequest, MAX_SEED, MODEL, MOSS, MOSS_BF16},
};
use serde_json::{Value, json};

pub const SHEET_REVISION: &str = "qwen-character-sheet-v1";
pub const MOSS_SHEET_REVISION: &str = "moss-character-sheet-v1";
const PORTRAIT: &str = "image/character-sheet/portrait.txt";
const TURNAROUND: &str = "image/character-sheet/turnaround.txt";
const MOSS_SHEET: &str = "image/character-sheet/moss.txt";

#[derive(Clone, Copy)]
pub enum SheetStrategy {
    /// Portrait, then a reference-guided turnaround, composed on the CPU.
    QwenTwoPass,
    /// The whole sheet in one text-to-image generation.
    MossSinglePass,
}
impl SheetStrategy {
    pub fn output_roles(self) -> &'static [&'static str] {
        match self {
            Self::QwenTwoPass => &["character_sheet", "portrait", "turnaround"],
            Self::MossSinglePass => &["character_sheet"],
        }
    }
    pub fn max_identity_references(self) -> usize {
        match self {
            Self::QwenTwoPass => 8,
            Self::MossSinglePass => 0,
        }
    }
}
pub fn sheet_strategy(model: &str) -> Result<SheetStrategy, &'static str> {
    match model {
        MODEL | BF16 => Ok(SheetStrategy::QwenTwoPass),
        MOSS | MOSS_BF16 => Ok(SheetStrategy::MossSinglePass),
        _ => Err("unsupported_character_sheet_model"),
    }
}
/// Images a job publishes: every role of its sheet strategy, otherwise one.
pub fn output_count(request: &JobRequest) -> usize {
    if request.operation == crate::contract::Operation::CharacterSheet {
        sheet_strategy(&request.model).map_or(1, |s| s.output_roles().len())
    } else {
        1
    }
}
pub fn sheet_capabilities(strategy: SheetStrategy) -> Value {
    let layout = json!({"aspect_ratio":"16:9","output_width":{"min":512,"max":4096,"multiple":512},"portrait_fraction":0.45});
    let (revision, pose_guide) = match strategy {
        SheetStrategy::QwenTwoPass => (SHEET_REVISION, true),
        SheetStrategy::MossSinglePass => (MOSS_SHEET_REVISION, false),
    };
    json!({"strategy":revision,"max_identity_references":strategy.max_identity_references(),"pose_guide":pose_guide,
        "layout":layout,"output_roles":strategy.output_roles(),"live_validated":false})
}
pub fn next_seed(seed: u64) -> u64 {
    (seed + 1) & MAX_SEED
}
pub fn sheet_plan(request: &JobRequest, sheet: &CharacterSheet) -> Value {
    let (width, height) = sheet.layout.dimensions();
    match sheet_strategy(&request.model).expect("validated sheet") {
        SheetStrategy::QwenTwoPass => {
            let sizes = sheet.layout.pass_sizes().expect("validated sheet");
            json!({"strategy":SHEET_REVISION,"prompt_revision":SHEET_REVISION,"composition_revision":"sheet-compose-v1",
                "pose_guide_revision":"sheet-pose-v1","output_width":width,
                "output_height":height,"portrait_fraction":sheet.layout.portrait_fraction,
                "output_roles":SheetStrategy::QwenTwoPass.output_roles(),
                "passes":[{"stage":"portrait","width":sizes[0].0,"height":sizes[0].1,"seed":request.seed,
                    "reference_roles":roles(request.inputs.reference_images.len(),false)},
                    {"stage":"turnaround","width":sizes[1].0,"height":sizes[1].1,"seed":request.seed.map(next_seed),
                    "reference_roles":roles(request.inputs.reference_images.len(),true)}]})
        }
        SheetStrategy::MossSinglePass => {
            json!({"strategy":MOSS_SHEET_REVISION,"prompt_revision":MOSS_SHEET_REVISION,"output_width":width,
                "output_height":height,"portrait_fraction":sheet.layout.portrait_fraction,
                "output_roles":SheetStrategy::MossSinglePass.output_roles(),
                "passes":[{"stage":"sheet","width":width,"height":height,"seed":request.seed,"reference_roles":[]}]})
        }
    }
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
        crate::config_files::required(if turnaround { TURNAROUND } else { PORTRAIT })
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

/// The single-pass MOSS sheet prompt: the server's template with the
/// character, the outfit and the layout's split filled in.
pub fn moss_sheet_prompt(sheet: &CharacterSheet) -> String {
    let text = |s: &str| s.trim().trim_end_matches('.').trim_end().to_owned();
    let clothing = text(&sheet.clothing_description);
    let outfit = if clothing.is_empty() {
        "In all three body views the character wears one coherent everyday outfit that suits them, from head to shoes.".into()
    } else {
        format!("Clothing and accessories in all three body views: {clothing}.")
    };
    let portrait = (sheet.layout.portrait_fraction * 100.).round() as u32;
    crate::config_files::required(MOSS_SHEET)
        .trim()
        .replace(
            "{character_description}",
            &text(&sheet.character_description),
        )
        .replace("{portrait_percent}", &portrait.to_string())
        .replace("{views_percent}", &(100 - portrait).to_string())
        .replace("{outfit}", &outfit)
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
    #[test]
    fn moss_sheet_prompt_fills_the_template_from_the_sheet() {
        let mut sheet = CharacterSheet {
            character_description: " SYNTHETIC_IDENTITY. ".into(),
            clothing_description: "SYNTHETIC_CLOTHES.".into(),
            ..Default::default()
        };
        let prompt = moss_sheet_prompt(&sheet);
        assert!(prompt.contains("The character: SYNTHETIC_IDENTITY. The sheet"));
        assert!(prompt.contains("in all three body views: SYNTHETIC_CLOTHES. The face"));
        assert!(prompt.contains("exactly the left 45 percent"));
        assert!(prompt.contains("the remaining 55 percent"));
        assert!(!prompt.contains(['{', '}']), "{prompt}");
        sheet.clothing_description = " ".into();
        sheet.layout.portrait_fraction = 0.5;
        let prompt = moss_sheet_prompt(&sheet);
        assert!(prompt.contains("wears one coherent everyday outfit"));
        assert!(prompt.contains("left 50 percent") && prompt.contains("remaining 50 percent"));
        assert!(!prompt.contains("Clothing and accessories"));
        for model in [crate::contract::MOSS, crate::contract::MOSS_BF16] {
            let strategy = sheet_strategy(model).unwrap();
            assert_eq!(strategy.output_roles(), ["character_sheet"]);
            assert_eq!(strategy.max_identity_references(), 0);
            assert_eq!(
                sheet_capabilities(strategy)["strategy"],
                MOSS_SHEET_REVISION
            );
        }
        assert_eq!(
            sheet_strategy(MODEL).unwrap().output_roles(),
            ["character_sheet", "portrait", "turnaround"]
        );
    }
}
