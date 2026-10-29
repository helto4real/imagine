//! Story messages and answer handling as pure functions, shared by the local
//! provider and an external agent's context bundle and submission.
use super::{
    director::{self, Py, js_trim},
    engine::*,
    grounding::Grounding,
    profiles, references, visuals,
};
use crate::video::{contract::*, text::PythonText};
use serde_json::{Value, json};
type Result<T> = std::result::Result<T, &'static str>;

/// Authoring by an agent outside Imagine: no provider task, one submission.
pub const EXTERNAL: &str = "external";
pub const INVENTORY: &str = "story_image_inventory";
/// The longest user message the provider path sends, in characters.
pub const CONTEXT_LIMIT: usize = 100000;
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
fn seconds(t: Time) -> f64 {
    t.num as f64 / t.den as f64
}
/// Timeline items in Director's order: start, end, then id.
fn timeline_order(a: (Time, Time, &str), b: (Time, Time, &str)) -> std::cmp::Ordering {
    let start = |t: (Time, Time, &str)| seconds(t.0);
    let end = |t: (Time, Time, &str)| seconds(t.0) + seconds(t.1);
    start(a)
        .total_cmp(&start(b))
        .then(end(a).total_cmp(&end(b)))
        .then_with(|| director::locale_cmp(a.2, b.2))
}
/// The browser's `storyScenePreviousContext` shape of a Scene.
fn previous_shape(
    title: &str,
    summary: &str,
    description: &str,
    ambient_sound: &str,
    non_diegetic_music: &str,
    prompts: &[&str],
) -> Py {
    Py::obj([
        ("title", Py::str(js_trim(title))),
        ("summary", Py::str(js_trim(summary))),
        ("description", Py::str(js_trim(description))),
        ("ambient_sound", Py::str(js_trim(ambient_sound))),
        ("non_diegetic_music", Py::str(js_trim(non_diegetic_music))),
        (
            "sections",
            Py::List(
                prompts
                    .iter()
                    .map(|p| js_trim(p))
                    .filter(|p| !p.is_empty())
                    .map(|p| Py::obj([("prompt", Py::str(p))]))
                    .collect(),
            ),
        ),
    ])
}
/// The Story a continuation follows, as the browser's
/// `storyContinuationAvailability` offers it.
pub(crate) fn continuation(base: &VideoProject) -> Result<Py> {
    let mut scenes = base.scenes.iter().collect::<Vec<_>>();
    scenes.sort_by(|a, b| {
        timeline_order(
            (a.start, a.duration, &a.id.0),
            (b.start, b.duration, &b.id.0),
        )
    });
    let source = *scenes.last().ok_or("story_continuation_source_missing")?;
    let prompt = |scene: &Scene| {
        base.creative
            .scene_prompts
            .get(&scene.id)
            .cloned()
            .unwrap_or_default()
    };
    let named = |title: String, fallback: &str| {
        if title.is_empty() {
            fallback.to_owned()
        } else {
            title
        }
    };
    let source_prompt = prompt(source);
    let mut sections = source.sections.iter().collect::<Vec<_>>();
    sections.sort_by(|a, b| {
        timeline_order(
            (a.start, a.duration, &a.id.0),
            (b.start, b.duration, &b.id.0),
        )
    });
    let prompts = sections
        .iter()
        .map(|s| s.prompt.as_str())
        .collect::<Vec<_>>();
    let previous = previous_shape(
        &named(source_prompt.title.clone(), "Final Scene"),
        &source_prompt.summary,
        &source.description,
        &source_prompt.ambient_sound,
        &source_prompt.non_diegetic_music,
        &prompts,
    );
    let meaningful = [
        "summary",
        "description",
        "ambient_sound",
        "non_diegetic_music",
    ]
    .iter()
    .any(|k| !previous.at(k).text().is_empty())
        || !previous.at("sections").as_list().unwrap().is_empty();
    if !meaningful {
        return Err("story_continuation_text_missing");
    }
    let recent = scenes[scenes.len().saturating_sub(3)..scenes.len() - 1]
        .iter()
        .map(|scene| {
            let source = prompt(scene);
            Py::obj([
                ("title", Py::str(js_trim(&named(source.title, "Scene")))),
                ("summary", Py::str(js_trim(&source.summary))),
            ])
        })
        .filter(|row| !row.at("summary").text().is_empty())
        .collect::<Vec<_>>();
    let mut text = recent
        .iter()
        .flat_map(|r| [r.at("title").text(), r.at("summary").text()])
        .collect::<Vec<_>>();
    for key in [
        "title",
        "summary",
        "description",
        "ambient_sound",
        "non_diegetic_music",
    ] {
        text.push(previous.at(key).text());
    }
    text.extend(
        previous
            .at("sections")
            .as_list()
            .unwrap()
            .iter()
            .map(|s| s.at("prompt").text()),
    );
    let aliases = imagine_video_contract::references::aliases(&text.join("\n"));
    let mut cast = if base.creative.visual_references_enabled == Some(false) {
        vec![]
    } else {
        base.creative
            .references
            .iter()
            .filter(|r| {
                r.enabled
                    && r.kind.eq_ignore_ascii_case("character")
                    && aliases.contains(&js_trim(&r.alias).to_lowercase())
            })
            .map(|r| {
                Py::obj([
                    (
                        "reference_tag",
                        Py::Str(format!(
                            "@{}",
                            imagine_video_contract::creative::reference_alias(&r.alias)
                        )),
                    ),
                    ("description", Py::str(js_trim(&r.description))),
                    ("required", Py::Bool(false)),
                ])
            })
            .collect::<Vec<_>>()
    };
    cast.sort_by(|a, b| {
        director::locale_cmp(&a.at("reference_tag").text(), &b.at("reference_tag").text())
    });
    let stable = Py::obj([
        ("schema_version", Py::Int(1)),
        ("source_scene_id", Py::str(&source.id.0)),
        ("recent_scene_summaries", Py::List(recent.clone())),
        ("previous_scene", previous.clone()),
        ("context_cast", Py::List(cast.clone())),
    ]);
    Ok(Py::obj([
        ("schema_version", Py::Int(1)),
        ("source_scene_id", Py::str(&source.id.0)),
        ("recent_scene_summaries", Py::List(recent)),
        ("previous_scene", previous),
        ("context_cast", Py::List(cast)),
        (
            "source_revision",
            Py::Str(director::stable_hash(&stable.compact())),
        ),
    ]))
}
/// The selected cast as the browser offers it.
fn cast_records(draft: &Draft, base: &VideoProject) -> Py {
    let members = draft
        .request
        .cast
        .iter()
        .filter_map(|id| base.creative.references.iter().find(|r| &r.id == id))
        .filter(|r| r.enabled)
        .map(|r| director::CastMember {
            title: &r.name,
            alias: &r.alias,
            description: &r.description,
            language: r.voice.as_ref().map_or("", |v| v.language.as_str()),
            voice: r.voice.as_ref().map_or("", |v| v.description.as_str()),
        })
        .collect::<Vec<_>>();
    director::cast_records(&members)
}
/// The inventory note of each story image, or a slot an agent fills.
pub enum Notes<'a> {
    Known(&'a [InventoryNote]),
    Slots,
}
pub fn note_slot(id: &str) -> String {
    slot(&format!("inventory_note:{id}"))
}
/// The browser's visual request descriptor of a story image.
pub fn visual(selection: &StoryImage) -> Py {
    Py::obj([
        ("id", Py::str(&selection.id)),
        ("label", Py::str(&selection.label)),
        ("note", Py::str("")),
        ("usage", Py::str(visuals::usage(selection.usage))),
    ])
}
pub fn inventory_message(selection: &StoryImage, instruction: &str) -> String {
    director::build_inventory(&visual(selection), instruction)
}
/// The browser's `story_outline` request for this Draft.
pub fn outline_request(draft: &Draft, base: &VideoProject, notes: Notes) -> Result<Py> {
    let request = &draft.request;
    let contract = visuals::contract(&request.images, request.add_dialogue)?;
    let note = |id: &str| match &notes {
        Notes::Known(notes) => notes
            .iter()
            .find(|n| n.id == id)
            .map(|n| n.note.clone())
            .unwrap_or_default(),
        Notes::Slots => note_slot(id),
    };
    Ok(Py::obj([
        ("operation", Py::str("story_outline")),
        ("user_instruction", Py::str(&request.instruction)),
        ("story", Py::str(&request.brief)),
        ("cast", cast_records(draft, base)),
        (
            "continuation_context",
            if request.destination == Destination::Continue {
                continuation(base)?
            } else {
                Py::Null
            },
        ),
        ("duration_seconds", Py::js(request.duration_seconds)),
        ("add_dialogue", Py::Bool(request.add_dialogue)),
        ("include_non_diegetic_music", Py::Bool(request.add_music)),
        (
            "image_inventory",
            visuals::inventory_rows(&request.images, &note),
        ),
        ("visual_beats", visuals::beat_records(&contract.beats)),
    ]))
}
/// The outline message: Director's `_build_story_outline` of the browser's request.
pub fn outline_message(
    draft: &Draft,
    base: &VideoProject,
    notes: Notes,
) -> Result<director::Built> {
    director::build_outline(&outline_request(draft, base, notes)?)
}
/// A Scene in Director's normalized layout, as the browser sends it back.
pub fn director_scene(scene: &StoryScene) -> Py {
    let sections = scene.sections.iter().map(|s| {
        let mut row = Py::obj([
            ("id", Py::str(&s.id)),
            ("summary", Py::str(&s.summary)),
            ("prompt", Py::str(&s.prompt)),
            ("weight", Py::js(s.weight)),
            ("image_ids", Py::strs(&s.image_ids)),
            ("first_frame_id", Py::str(&s.first_frame_id)),
            ("last_frame_id", Py::str(&s.last_frame_id)),
        ]);
        if let Some(intent) = &s.dialogue_intent {
            row.push(
                "dialogue_intent",
                Py::obj([
                    ("participants", Py::strs(&intent.participants)),
                    ("purpose", Py::str(&intent.purpose)),
                    ("max_turns", Py::Int(intent.max_turns.into())),
                ]),
            );
        }
        row
    });
    Py::obj([
        ("id", Py::str(&scene.id)),
        ("title", Py::str(&scene.title)),
        ("summary", Py::str(&scene.summary)),
        ("description", Py::str(&scene.description)),
        ("ambient_sound", Py::str(&scene.ambient_sound)),
        ("non_diegetic_music", Py::str(&scene.non_diegetic_music)),
        ("weight", Py::js(scene.weight)),
        ("image_ids", Py::strs(&scene.image_ids)),
        (
            "boundary_from_previous",
            Py::str(&scene.boundary_from_previous),
        ),
        ("sections", Py::List(sections.collect())),
    ])
}
/// The browser's `story_scene` request for Scene `index`, without its images.
pub fn scene_request(draft: &Draft, base: &VideoProject, index: usize) -> Result<Py> {
    let request = &draft.request;
    let outline = draft.outline.as_ref().ok_or("story_outline_required")?;
    let source = outline.scenes.get(index).ok_or("story_outline_required")?;
    let scene = director_scene(source);
    let continuation = if index == 0 && request.destination == Destination::Continue {
        Some(continuation(base)?)
    } else {
        None
    };
    let previous = match (index, &continuation) {
        (0, Some(c)) => previous_shape(
            &c.at("previous_scene").at("title").text(),
            &c.at("previous_scene").at("summary").text(),
            &c.at("previous_scene").at("description").text(),
            &c.at("previous_scene").at("ambient_sound").text(),
            &c.at("previous_scene").at("non_diegetic_music").text(),
            &c.at("previous_scene")
                .at("sections")
                .as_list()
                .unwrap_or_default()
                .iter()
                .map(|s| s.at("prompt").text())
                .collect::<Vec<_>>()
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
        ),
        (0, None) => Py::Null,
        _ => match draft.scenes.get(index - 1).and_then(Option::as_ref) {
            Some(s) => previous_shape(
                &s.title,
                &s.summary,
                &s.description,
                &s.ambient_sound,
                &s.non_diegetic_music,
                &s.sections
                    .iter()
                    .map(|s| s.prompt.as_str())
                    .collect::<Vec<_>>(),
            ),
            None => Py::Null,
        },
    };
    let summary = |i: Option<usize>| {
        i.and_then(|i| outline.scenes.get(i))
            .map(|s| s.summary.clone())
            .unwrap_or_default()
    };
    let mut out = Py::obj([
        ("operation", Py::str("story_scene")),
        ("user_instruction", Py::str(&request.instruction)),
        ("add_dialogue", Py::Bool(request.add_dialogue)),
        (
            "cast",
            director::scene_cast(&cast_records(draft, base), &scene),
        ),
        ("scene", scene),
        ("previous_summary", Py::Str(summary(index.checked_sub(1)))),
        ("previous_scene_context", previous),
        (
            "context_cast",
            continuation
                .as_ref()
                .map_or(Py::List(vec![]), |c| c.at("context_cast").clone()),
        ),
        ("next_summary", Py::Str(summary(Some(index + 1)))),
    ]);
    if request.add_dialogue {
        let timings = director::dialogue_timings(
            &outline
                .scenes
                .iter()
                .map(|s| {
                    (
                        s.weight,
                        s.sections
                            .iter()
                            .map(|section| (section.id.clone(), section.weight))
                            .collect(),
                    )
                })
                .collect::<Vec<_>>(),
            request.duration_seconds,
        );
        out.push("dialogue_timing", timings[index].clone());
    }
    Ok(out)
}
/// Scene `index`'s expansion message; `images` when its images are supplied.
pub fn expansion_message(
    draft: &Draft,
    base: &VideoProject,
    index: usize,
    images: bool,
) -> Result<director::Built> {
    director::build_scene(&scene_request(draft, base, index)?, images)
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
pub fn accept_inventory(draft: &mut Draft, id: &str, note: String) {
    draft.inventory.push(InventoryNote {
        id: id.into(),
        note,
    });
}
/// Accepts an outline answer; `references` restores the message's markers.
pub fn accept_outline(
    draft: &mut Draft,
    text: &str,
    references: &[(String, String)],
) -> Result<()> {
    let request = &draft.request;
    let text = references::restore_all(text, references);
    let mut board = outline_with_images(&text, &request.images, request.add_dialogue)?;
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
pub fn accept_expansion(
    draft: &mut Draft,
    index: usize,
    text: &str,
    references: &[(String, String)],
) -> Result<()> {
    let source = draft
        .outline
        .as_ref()
        .and_then(|o| o.scenes.get(index))
        .ok_or("story_outline_required")?;
    let mut scene = expansion(&references::restore_all(text, references), source)?;
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
    let mut section = json!({"type":"object","required":["prompt","weight"],"additionalProperties":false,"properties":{"id":{"type":"string"},"summary":{"type":"string"},"prompt":{"type":"string","minLength":1},"weight":{"type":"number","exclusiveMinimum":0},"image_ids":ids,"first_frame_id":{"type":"string"},"last_frame_id":{"type":"string"}}});
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
    let outline_slots = request
        .images
        .iter()
        .map(|i| format!("inventory_note:{}", i.id))
        .collect::<Vec<_>>();
    let mut scene_slots = vec![
        "previous_scene_summary",
        "previous_scene_context",
        "next_scene_summary",
        "selected_cast",
        "continuity_cast",
    ];
    if request.add_dialogue {
        scene_slots.push("dialogue_timing");
    }
    scene_slots.push("scene_json");
    if images {
        scene_slots.push("image_context");
    }
    let instruction = request.instruction.pytrim();
    let timing = slot("dialogue_timing");
    let scene = director::scene_user(&director::Expansion {
        scene_json: &slot("scene_json"),
        rule: &if images {
            slot("image_context")
        } else {
            director::SCENE_NO_IMAGES.into()
        },
        instruction,
        cast: &slot("selected_cast"),
        previous_summary: &slot("previous_scene_summary"),
        previous: &slot("previous_scene_context"),
        next_summary: &slot("next_scene_summary"),
        context_cast: &slot("continuity_cast"),
        timing: request.add_dialogue.then_some(timing.as_str()),
    });
    let continued = request.destination == Destination::Continue;
    let (first, first_cast) = if continued {
        let context = director::normalize_continuation(&continuation(base)?)?;
        (
            context.at("previous_scene").dumps(),
            context.at("context_cast").dumps(),
        )
    } else {
        ("null".into(), "[]".into())
    };

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
        "outline":{"system":system_with_images(profile,outline_key(request.add_dialogue),&request.images,request.add_dialogue)?,"user":outline_message(draft,base,Notes::Slots)?.user,"slots":outline_slots,"slot_values":{
            "inventory_note":"each story image's submitted inventory note, escaped as the body of a JSON string"},
            "output_schema":visuals::schema(&request.images,request.add_dialogue)?},
        "expansion":{"system":system_with_images(profile,scene_key(request.add_dialogue),&request.images,request.add_dialogue)?,"user":scene,"slots":scene_slots,"slot_values":{
            "previous_scene_summary":"the previous outline Scene's summary; empty for the first Scene",
            "previous_scene_context":{"first_scene":first,"later_scenes":"the previous Scene's expansion as JSON {title, summary, description, ambient_sound, non_diegetic_music, sections: [{prompt}]} with trimmed text and empty Section prompts left out"},
            "next_scene_summary":"the next outline Scene's summary; empty for the last Scene",
            "selected_cast":"the outline message's selected_cast entries whose reference tag appears in scene_json, as JSON",
            "continuity_cast":{"first_scene":first_cast,"later_scenes":"[]"},
            "dialogue_timing":"this Scene's {scene_duration_seconds, sections: [{id, duration_seconds}]} from the outline weights and the requested duration, as JSON",
            "scene_json":"this Scene from the projected outline as JSON in Director's layout {id, title, summary, description, ambient_sound, non_diegetic_music, weight, image_ids, boundary_from_previous, sections: [{id, summary, prompt, weight, image_ids, first_frame_id, last_frame_id[, dialogue_intent]}]}",
            "image_context":{"with_images":director::SCENE_IMAGES,"without_images":director::SCENE_NO_IMAGES},
            "references":"the provider path replaces each @reference in the outline's story, cast and continuation and in each Scene's scene_json, selected_cast, previous_scene_context and continuity_cast with [[DIRECTOR_REF_nnnn]] markers numbered in order of first appearance, and restores them in the answer; an agent may write the references directly"},
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
    let message = outline_message(draft, base, Notes::Known(&draft.inventory.clone()))
        .map_err(|code| Rejection::new(code, "outline"))?;
    if message.user.chars().count() > CONTEXT_LIMIT {
        return Err(Rejection::new("story_context_limit", "outline"));
    }
    accept_outline(draft, &submission.outline, &message.references).map_err(|code| {
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
            base,
            index,
            !grounding.selected(&scene_image_ids(source)).is_empty(),
        )
        .map_err(|c| Rejection::new(c, at()))?;
        if message.user.chars().count() > CONTEXT_LIMIT {
            return Err(Rejection::new("story_context_limit", at()));
        }
        let text = owned(text, source).map_err(|c| Rejection::new(c, at()))?;
        accept_expansion(draft, index, &text, &message.references)
            .map_err(|c| Rejection::new(c, at()))?;
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

#[cfg(test)]
mod tests {
    use super::*;
    /// Director's browser helpers on the same inputs
    /// (`tools/director-story-browser-oracle.mjs`).
    #[test]
    fn browser_requests_match_the_director_source_oracle() {
        let fixture = Py::parse(include_str!(
            "../../../tests/fixtures/director-story-browser.json"
        ))
        .unwrap();
        let time = |seconds: &Py| json!({"num": (seconds.number().unwrap() * 4.) as i64, "den": 4});
        for case in fixture.at("continuation").as_list().unwrap() {
            let mut project = VideoProject::create(
                serde_json::from_str(include_str!(
                    "../../../design/video-contract/create-project.json"
                ))
                .unwrap(),
            )
            .unwrap();
            project.creative.visual_references_enabled =
                Some(*case.at("references_enabled") == Py::Bool(true));
            for (i, r) in case.at("references").as_list().unwrap().iter().enumerate() {
                project.creative.references.push(
                    serde_json::from_value(json!({"id":format!("own_ref_{i}"),"name":r.at("alias").text(),"alias":r.at("alias").text(),"kind":r.at("kind").text(),"description":r.at("description").text(),"enabled":*r.at("enabled") == Py::Bool(true),"strength":1.,"retention":"","images":[],"voice":null,"origin":null})).unwrap(),
                );
            }
            for scene in case.at("scenes").as_list().unwrap() {
                let span = |s: &Py| {
                    let start = s.at("start").number().unwrap();
                    (
                        time(s.at("start")),
                        time(&Py::Float(s.at("end").number().unwrap() - start)),
                    )
                };
                let (start, duration) = span(scene);
                let sections = scene
                    .at("sections")
                    .as_list()
                    .unwrap()
                    .iter()
                    .map(|s| {
                        let (start, duration) = span(s);
                        json!({"id":s.at("id").text(),"start":start,"duration":duration,"prompt":s.at("prompt").text()})
                    })
                    .collect::<Vec<_>>();
                let id = scene.at("id").text();
                project.scenes.push(serde_json::from_value(json!({"id":id,"start":start,"duration":duration,"description":scene.at("description").text(),"model":"project_default","settings":[],"sections":sections,"bindings":[],"continuity":null,"accepted_take":null})).unwrap());
                project.creative.scene_prompts.insert(
                    SceneId(id),
                    serde_json::from_value(json!({"title":scene.at("title").text(),"summary":scene.at("summary").text(),"ambient_sound":scene.at("ambient_sound").text(),"non_diegetic_music":scene.at("non_diegetic_music").text()})).unwrap(),
                );
            }
            let name = case.at("name").text();
            match case.at("context") {
                Py::Null => assert_eq!(
                    continuation(&project).unwrap_err(),
                    "story_continuation_text_missing",
                    "{name}"
                ),
                context => assert_eq!(&continuation(&project).unwrap(), context, "{name}"),
            }
        }
        for case in fixture.at("timing").as_list().unwrap() {
            let scenes = case
                .at("scenes")
                .as_list()
                .unwrap()
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    let s = s.as_list().unwrap();
                    (
                        s[0].number().unwrap(),
                        s[1].as_list()
                            .unwrap()
                            .iter()
                            .enumerate()
                            .map(|(j, w)| {
                                (
                                    format!("scene_00{}_section_00{}", i + 1, j + 1),
                                    w.number().unwrap(),
                                )
                            })
                            .collect(),
                    )
                })
                .collect::<Vec<_>>();
            assert_eq!(
                Py::List(director::dialogue_timings(
                    &scenes,
                    case.at("duration").number().unwrap()
                )),
                *case.at("timing"),
                "{}",
                case.at("duration").dumps()
            );
        }
        for case in fixture.at("beats").as_list().unwrap() {
            let images: Vec<StoryImage> = serde_json::from_str(
                &Py::List(
                    case.at("images")
                        .as_list()
                        .unwrap()
                        .iter()
                        .map(|i| {
                            Py::obj([
                                ("id", i.at("id").clone()),
                                ("asset", Py::str("own_asset")),
                                ("label", Py::str("")),
                                ("usage", i.at("usage").clone()),
                            ])
                        })
                        .collect(),
                )
                .compact(),
            )
            .unwrap();
            assert_eq!(
                visuals::beat_records(&visuals::beats(&images).unwrap()),
                *case.at("records")
            );
            assert_eq!(
                visuals::inventory_rows(&images, &|_| String::new())
                    .as_list()
                    .unwrap()
                    .iter()
                    .map(|r| (r.at("id").clone(), r.at("usage").clone()))
                    .collect::<Vec<_>>(),
                case.at("visuals")
                    .as_list()
                    .unwrap()
                    .iter()
                    .map(|r| (r.at("id").clone(), r.at("usage").clone()))
                    .collect::<Vec<_>>()
            );
        }
        let cast = fixture.at("cast");
        let members = cast.at("members").as_list().unwrap();
        let members = members
            .iter()
            .map(|m| director::CastMember {
                title: m.at("title").as_str().unwrap(),
                alias: m.at("alias").as_str().unwrap(),
                description: m.at("description").as_str().unwrap(),
                language: m.at("language").as_str().unwrap(),
                voice: m.at("voice").as_str().unwrap(),
            })
            .collect::<Vec<_>>();
        let records = director::cast_records(&members);
        assert_eq!(records, *cast.at("records"));
        assert_eq!(
            director::scene_cast(&records, cast.at("scene")),
            *cast.at("scene_records")
        );
        let previous = fixture.at("previous");
        let scene = previous.at("scene");
        let prompts = scene
            .at("sections")
            .as_list()
            .unwrap()
            .iter()
            .map(|s| s.at("prompt").text())
            .collect::<Vec<_>>();
        assert_eq!(
            previous_shape(
                &scene.at("title").text(),
                &scene.at("summary").text(),
                &scene.at("description").text(),
                &scene.at("ambient_sound").text(),
                &scene.at("non_diegetic_music").text(),
                &prompts.iter().map(String::as_str).collect::<Vec<_>>(),
            ),
            *previous.at("context")
        );
    }
}
