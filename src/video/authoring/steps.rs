//! Story messages and answer handling as pure functions, shared by the local
//! provider and an external agent's context bundle and submission.
use super::{engine::*, grounding::Grounding, profiles, visuals};
use crate::video::contract::*;
use serde_json::{Value, json};
type Result<T> = std::result::Result<T, &'static str>;

/// Authoring by an agent outside Imagine: no provider task, one submission.
pub const EXTERNAL: &str = "external";
pub const INVENTORY: &str = "story_image_inventory";
/// The longest user message the provider path sends, in characters.
pub const CONTEXT_LIMIT: usize = 100000;
const NO_IMAGES: &str = "No images are supplied.";
const SELECTED_IMAGES: &str = "The selected source images are supplied. Ground visible details in those images; do not invent unseen facts.";
/// Fields the server fills in an expansion from the projected outline Scene.
pub const CODE_OWNED: [&str; 6] = [
    "id",
    "image_ids",
    "sections[].id",
    "sections[].image_ids",
    "sections[].first_frame_id",
    "sections[].last_frame_id",
];
/// A per-Scene value in a bundle message, filled by the agent.
pub fn slot(name: &str) -> String {
    format!("{{{{SLOT:{name}}}}}")
}
pub fn outline_key(dialogue: bool) -> &'static str {
    if dialogue {
        "story_outline_dialogue"
    } else {
        "story_outline"
    }
}
pub fn scene_key(dialogue: bool) -> &'static str {
    if dialogue {
        "story_scene_dialogue"
    } else {
        "story_scene"
    }
}
pub(crate) fn last_scene(base: &VideoProject) -> Option<&Scene> {
    base.scenes.iter().max_by(|a, b| {
        let end = |s: &Scene| {
            s.start.num as f64 / s.start.den as f64 + s.duration.num as f64 / s.duration.den as f64
        };
        end(a).total_cmp(&end(b))
    })
}
/// The existing Scene a continuation follows, as the provider sees it.
fn continuation(base: &VideoProject) -> Option<Value> {
    last_scene(base).map(|s| json!({"description":s.description,"sections":s.sections}))
}
pub fn inventory_message(selection: &StoryImage, instruction: &str) -> String {
    format!(
        "Task:\nInspect this one image for visual-story planning.\n\n<image_id>\n{}\n</image_id>\n\n<label>\n{}\n</label>\n\n<additional_instruction>\n{}\n</additional_instruction>",
        selection.id, selection.label, instruction
    )
}
/// The context blocks every Story message ends with.
fn grounded(user: String, draft: &Draft, inventory: &str) -> String {
    let mut user = user.replace(
        "<selected_cast>\n[]\n</selected_cast>",
        &format!("<selected_cast>\n{}\n</selected_cast>", draft.selected_cast),
    );
    user.push_str(&format!(
        "\n\n<image_sources>\n{}\n</image_sources>",
        json!(draft.request.images)
    ));
    user.push_str(&format!(
        "\n\n<image_inventory>\n{}\n</image_inventory>\n\n<visual_beats>\n{}\n</visual_beats>",
        inventory,
        json!(
            visuals::beats(&draft.request.images)
                .unwrap()
                .iter()
                .map(|b| json!({"beat_key":b.key,"image_ids":b.ids,"semantic_role":b.role}))
                .collect::<Vec<_>>()
        )
    ));
    user
}
/// `inventory` is the notes' JSON, or a slot.
pub fn outline_message(draft: &Draft, base: &VideoProject, inventory: &str) -> String {
    let context = if draft.request.destination == Destination::Continue {
        continuation(base).into_iter().collect::<Vec<_>>()
    } else {
        vec![]
    };
    grounded(
        format!(
            "Task:\nPlan an ordered Story Outline. Return only JSON matching the supplied schema. Every Scene must fit {} seconds and contain 1–12 sections. Requested duration: {} seconds.\n\n<story>\n{}\n</story>\n\n<latest_moment>\n{}\n</latest_moment>\n\n<earlier_continuity>\n{}\n</earlier_continuity>\n\n<selected_cast>\n[]\n</selected_cast>\n\n<additional_instruction>\n{}\n</additional_instruction>",
            MAX_SCENE_SECONDS,
            draft.request.duration_seconds,
            draft.request.brief,
            draft.request.brief,
            json!(context),
            draft.request.instruction
        ),
        draft,
        inventory,
    )
}
/// The Scene before `index`: the previous expansion, or for the first Scene
/// the continued one.
pub fn previous_context(draft: &Draft, base: &VideoProject, index: usize) -> Value {
    if index > 0 {
        json!(draft.scenes[index - 1])
    } else if draft.request.destination == Destination::Continue {
        continuation(base).unwrap_or(json!(null))
    } else {
        json!(null)
    }
}
pub fn next_summary(outline: &Storyboard, index: usize) -> &str {
    outline
        .scenes
        .get(index + 1)
        .map(|s| s.summary.as_str())
        .unwrap_or("")
}
/// Every story image an outline Scene refers to.
pub fn scene_image_ids(source: &StoryScene) -> Vec<String> {
    source
        .image_ids
        .iter()
        .chain(source.sections.iter().flat_map(|s| {
            s.image_ids
                .iter()
                .chain([&s.first_frame_id, &s.last_frame_id])
        }))
        .cloned()
        .collect()
}
/// One Scene's expansion message; `images` when that Scene's images are supplied.
pub fn expansion_message(
    draft: &Draft,
    previous: &str,
    next_summary: &str,
    scene_json: &str,
    inventory: &str,
    images: bool,
) -> String {
    let user = grounded(
        format!(
            "Task:\nExpand this one Scene and its Sections. Return one JSON object matching the supplied Scene shape and no markdown.\n\nVisual context rule:\n{NO_IMAGES} Provide detailed environment, spatial layout, appearance, lighting, and atmosphere.\n\n<previous_scene_context>\n{previous}\n</previous_scene_context>\n\n<selected_cast>\n[]\n</selected_cast>\n\n<continuity_cast>\n[]\n</continuity_cast>\n\n<next_scene_summary>\n{next_summary}\n</next_scene_summary>\n\n<additional_instruction>\n{}\n</additional_instruction>\n\n<scene_json>\n{scene_json}\n</scene_json>",
            draft.request.instruction,
        ),
        draft,
        inventory,
    );
    if images {
        user.replace(NO_IMAGES, SELECTED_IMAGES)
    } else {
        user
    }
}
pub fn accept_inventory(draft: &mut Draft, id: &str, note: String) {
    draft.inventory.push(InventoryNote {
        id: id.into(),
        note,
    });
}
pub fn accept_outline(draft: &mut Draft, text: &str) -> Result<()> {
    let request = &draft.request;
    let mut board = outline_with_images(text, &request.images, request.add_dialogue)?;
    if !request.add_music {
        for scene in &mut board.scenes {
            scene.non_diegetic_music.clear();
        }
    }
    visuals::validate_cast(&board, &draft.selected_cast, request.add_dialogue)?;
    draft.planned_durations = durations(
        &board.scenes.iter().map(|s| s.weight).collect::<Vec<_>>(),
        request.duration_seconds,
        MAX_SCENE_SECONDS,
    )?;
    draft.scenes = vec![None; board.scenes.len()];
    draft.outline = Some(board);
    Ok(())
}
pub fn accept_expansion(draft: &mut Draft, index: usize, text: &str) -> Result<()> {
    let source = draft
        .outline
        .as_ref()
        .and_then(|o| o.scenes.get(index))
        .ok_or("story_outline_required")?;
    let mut scene = expansion(text, source)?;
    if !draft.request.add_music {
        scene.non_diegetic_music.clear();
    }
    visuals::validate_cast(
        &Storyboard {
            scenes: vec![scene.clone()],
        },
        &draft.selected_cast,
        draft.request.add_dialogue,
    )?;
    draft.scenes[index] = Some(scene);
    Ok(())
}
/// The planning limits `GET /v1/video/capabilities` reports, with continuity
/// methods per target variant.
pub fn limits() -> Value {
    let continuity = crate::video::adapters::h3::variants()
        .into_iter()
        .map(|(model, capabilities)| {
            (
                model.variant.clone(),
                json!({"methods":capabilities.continuity,"auto":auto_continuity(&model.variant)}),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    json!({"max_scene_seconds":MAX_SCENE_SECONDS,"max_scenes":MAX_SCENES,"max_sections_per_scene":MAX_SECTIONS_PER_SCENE,"supported_boundaries":BOUNDARIES,"continuity":continuity})
}
/// The `StoryScene` shape an expansion returns; code-owned fields are optional.
pub fn scene_schema(dialogue: bool) -> Value {
    let ids = json!({"type":"array","items":{"type":"string"}});
    let mut section = json!({"type":"object","required":["prompt","weight"],"additionalProperties":false,"properties":{"id":{"type":"string"},"prompt":{"type":"string","minLength":1},"weight":{"type":"number","exclusiveMinimum":0},"image_ids":ids,"first_frame_id":{"type":"string"},"last_frame_id":{"type":"string"}}});
    if dialogue {
        section["properties"]["dialogue_intent"] = json!({"type":"object","required":["participants","purpose"],"additionalProperties":false,"properties":{"participants":{"type":"array","minItems":2,"maxItems":2,"items":{"type":"string"}},"purpose":{"type":"string","minLength":1},"max_turns":{"const":2}}});
    }
    json!({"type":"object","required":["title","summary","description","ambient_sound","non_diegetic_music","weight","boundary_from_previous","sections"],"additionalProperties":false,"properties":{"id":{"type":"string"},"title":{"type":"string"},"summary":{"type":"string"},"description":{"type":"string"},"ambient_sound":{"type":"string"},"non_diegetic_music":{"type":"string"},"weight":{"type":"number","exclusiveMinimum":0},"boundary_from_previous":{"enum":BOUNDARIES},"sections":{"type":"array","minItems":1,"maxItems":MAX_SECTIONS_PER_SCENE,"items":section},"image_ids":ids}})
}
/// A cast member's enabled identity image, normalized as story images are.
pub(crate) struct Identity {
    pub image_id: String,
    pub label: String,
    pub image: crate::enhancement::ProviderImage,
}
/// Everything the provider would be given, for an agent writing the whole
/// Story at once. Per-Scene values are slots; `{"profile_id":…}` metadata only.
pub(crate) fn bundle(
    draft: &Draft,
    base: &VideoProject,
    profile: &profiles::Snapshot,
    grounding: &Grounding,
    identities: &[Identity],
) -> Result<Value> {
    let request = &draft.request;
    let images = !request.images.is_empty();
    let inventory = if images {
        slot("image_inventory")
    } else {
        "[]".into()
    };
    let mut outline_slots = vec![];
    let mut scene_slots = vec!["previous_scene_context", "next_scene_summary", "scene_json"];
    if images {
        outline_slots.push("image_inventory");
        scene_slots.extend(["image_inventory", "image_context"]);
    }
    let mut scene = expansion_message(
        draft,
        &slot("previous_scene_context"),
        &slot("next_scene_summary"),
        &slot("scene_json"),
        &inventory,
        false,
    );
    if images {
        scene = scene.replace(NO_IMAGES, &slot("image_context"));
    }
    let first = previous_context(draft, base, 0).to_string();
    let story = grounding.images.iter().map(|(selection, image)| {
        json!({"image_id":selection.id,"role":"story","usage":selection.usage,"label":selection.label,"mime":image.mime,"base64":image.base64})
    });
    let cast = identities.iter().map(|i| {
        json!({"image_id":i.image_id,"role":"cast","usage":"identity_reference","label":i.label,"mime":i.image.mime,"base64":i.image.base64})
    });
    Ok(json!({
        "profile":{"id":profile.resolved_id,"revision":profile.profile.revision,"prompt_files":profile.profile.prompt_files},
        "limits":limits(),
        "inventory":{"system":system(profile, INVENTORY)?,"messages":request.images.iter().map(|s|json!({"image_id":s.id,"user":inventory_message(s,&request.instruction)})).collect::<Vec<_>>()},
        "outline":{"system":system_with_images(profile,outline_key(request.add_dialogue),&request.images,request.add_dialogue)?,"user":outline_message(draft,base,&inventory),"slots":outline_slots,"output_schema":visuals::schema(&request.images,request.add_dialogue)?},
        "expansion":{"system":system_with_images(profile,scene_key(request.add_dialogue),&request.images,request.add_dialogue)?,"user":scene,"slots":scene_slots,"slot_values":{
            "previous_scene_context":{"first_scene":first,"later_scenes":"the previous Scene's expansion as compact JSON"},
            "next_scene_summary":"the next outline Scene's summary; empty for the last Scene",
            "scene_json":"this Scene from the projected outline as compact JSON, with its code-owned fields",
            "image_inventory":"the submitted inventory notes as a JSON array of {id, note}, in story image order",
            "image_context":{"with_images":SELECTED_IMAGES,"without_images":NO_IMAGES}},
            "output_schema":scene_schema(request.add_dialogue)},
        "code_owned_fields":CODE_OWNED,
        "images":story.chain(cast).collect::<Vec<_>>(),
    }))
}
/// A submission error: a stable code and, when it concerns one part of the
/// submission, where it was found.
#[derive(Debug)]
pub struct Rejection {
    pub status: u16,
    pub code: &'static str,
    pub at: Option<String>,
    pub detail: Option<String>,
}
impl Rejection {
    pub fn new(code: &'static str, at: impl Into<String>) -> Self {
        Self {
            status: 422,
            code,
            at: Some(at.into()),
            detail: None,
        }
    }
}
impl From<crate::scheduler::Fault> for Rejection {
    fn from(fault: crate::scheduler::Fault) -> Self {
        Self {
            status: fault.status,
            code: fault.code,
            at: None,
            detail: None,
        }
    }
}
/// Where a production error is reported.
impl From<Misplaced> for Rejection {
    fn from(m: Misplaced) -> Self {
        let (at, detail) = match (m.scene, m.code) {
            (None, "story_continuity_source_missing") => {
                (Some("outline".into()), Some("scenes[0]".into()))
            }
            (Some(i), _) => (
                Some(format!("production.scenes[{i}]")),
                m.section.map(|j| format!("sections[{j}]")),
            ),
            (None, "production_scene_count_mismatch") => (Some("production.scenes".into()), None),
            _ => (None, None),
        };
        Self {
            status: 422,
            code: m.code,
            at,
            detail,
        }
    }
}
/// Parses a complete submission into `draft` through the provider path's
/// steps, in order; the caller keeps the previous Draft on error.
pub(crate) fn submit(
    draft: &mut Draft,
    base: &VideoProject,
    grounding: &Grounding,
    submission: &Submission,
) -> std::result::Result<(), Rejection> {
    let request = draft.request.clone();
    draft.inventory.clear();
    draft.outline = None;
    draft.scenes.clear();
    draft.planned_durations.clear();
    draft.failure = None;
    draft.failed_index = None;
    let inventory = |code| Rejection::new(code, "inventory");
    if submission.inventory.len() != request.images.len() {
        return Err(inventory("authoring_inventory_mismatch"));
    }
    for selection in &request.images {
        let mut notes = submission.inventory.iter().filter(|n| n.id == selection.id);
        let (Some(note), None) = (notes.next(), notes.next()) else {
            return Err(inventory("authoring_inventory_mismatch"));
        };
        accept_inventory(draft, &selection.id, note.note.clone());
    }
    let notes = json!(draft.inventory).to_string();
    if outline_message(draft, base, &notes).chars().count() > CONTEXT_LIMIT {
        return Err(Rejection::new("story_context_limit", "outline"));
    }
    accept_outline(draft, &submission.outline).map_err(|code| {
        let mut rejection = Rejection::new(code, "outline");
        if code == "story_scene_duration_capacity" {
            let minimum = (request.duration_seconds / MAX_SCENE_SECONDS).ceil();
            rejection.detail = Some(format!("minimum_scenes={minimum}"));
        }
        rejection
    })?;
    let outline = draft.outline.clone().unwrap();
    if submission.expansions.len() != outline.scenes.len() {
        return Err(Rejection::new(
            "story_expansion_count_mismatch",
            "expansions",
        ));
    }
    for (index, (text, source)) in submission
        .expansions
        .iter()
        .zip(&outline.scenes)
        .enumerate()
    {
        let at = || format!("expansions[{index}]");
        let message = expansion_message(
            draft,
            &previous_context(draft, base, index).to_string(),
            next_summary(&outline, index),
            &serde_json::to_string(source).unwrap(),
            &notes,
            !grounding.selected(&scene_image_ids(source)).is_empty(),
        );
        if message.chars().count() > CONTEXT_LIMIT {
            return Err(Rejection::new("story_context_limit", at()));
        }
        let text = owned(text, source).map_err(|c| Rejection::new(c, at()))?;
        accept_expansion(draft, index, &text).map_err(|c| Rejection::new(c, at()))?;
    }
    draft.production = submission.production.clone();
    draft.phase = Phase::Review;
    Ok(())
}
/// Sets an expansion's code-owned fields from its projected outline Scene.
fn owned(text: &str, source: &StoryScene) -> Result<String> {
    let mut scene: Value = serde_json::from_str(text).map_err(|_| "authoring_invalid_json")?;
    let object = scene.as_object_mut().ok_or("authoring_invalid_json")?;
    object.insert("id".into(), json!(source.id));
    object.insert("image_ids".into(), json!(source.image_ids));
    let sections = object
        .get_mut("sections")
        .and_then(Value::as_array_mut)
        .ok_or("authoring_invalid_json")?;
    if sections.len() != source.sections.len() {
        return Err("story_expansion_changed_structure");
    }
    for (section, source) in sections.iter_mut().zip(&source.sections) {
        let section = section.as_object_mut().ok_or("authoring_invalid_json")?;
        section.insert("id".into(), json!(source.id));
        section.insert("image_ids".into(), json!(source.image_ids));
        section.insert("first_frame_id".into(), json!(source.first_frame_id));
        section.insert("last_frame_id".into(), json!(source.last_frame_id));
    }
    Ok(scene.to_string())
}
