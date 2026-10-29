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
/// The longest rendered conversation Director's Story bridge accepts.
const MAX_STORY_CONTEXT_CHARS: usize = 200000;
/// A Story context as Director's Story bridge receives it.
pub struct Bridge {
    /// The rendered conversation, Director's `story`.
    pub story: String,
    /// `grounding_context`, or null without cast state and protected facts.
    pub grounding: Py,
    /// `opening_appearance_context`, or null without cast state.
    pub opening: Py,
}
fn context_error(at: impl Into<String>, reason: &str) -> Rejection {
    located("story_context_invalid", at, reason)
}
fn located(code: &'static str, at: impl Into<String>, reason: &str) -> Rejection {
    let mut rejection = Rejection::new(code, at);
    rejection.detail = Some(reason.into());
    rejection
}
/// Messages as `[name] text` lines, the role standing in for an empty name.
fn transcript(messages: &[ContextMessage]) -> String {
    messages
        .iter()
        .map(|m| {
            let name = m.name.pytrim();
            format!(
                "[{}] {}",
                if name.is_empty() { &m.role } else { name },
                m.text
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}
fn check_messages(messages: &[ContextMessage], at: &str) -> std::result::Result<(), Rejection> {
    for (i, message) in messages.iter().enumerate() {
        if !["user", "assistant"].contains(&message.role.as_str()) {
            return Err(context_error(
                format!("{at}[{i}].role"),
                "must be user or assistant",
            ));
        }
        if message.text.pytrim().is_empty() {
            return Err(context_error(
                format!("{at}[{i}].text"),
                "must not be empty",
            ));
        }
    }
    Ok(())
}
/// Director's grounding path as the request's: `$.characters[i]` is
/// `context.cast_state[i]`, whose reference stands for the ID and tag.
fn context_path(invalid: &director::Invalid) -> String {
    let path = invalid.path.replacen('$', "context", 1);
    let path = path.replacen("context.characters", "context.cast_state", 1);
    path.replace(".character_id", ".reference_id")
        .replace(".reference_tag", ".reference_id")
}
/// The request's Story context as Director's bridge receives it, validated
/// by Director's own rules; errors name where in the request.
pub fn bridge(
    request: &Request,
    base: &VideoProject,
) -> std::result::Result<Option<Bridge>, Rejection> {
    let Some(context) = &request.context else {
        return Ok(None);
    };
    if !request.brief.is_empty() {
        return Err(context_error("brief", "must be empty with a context"));
    }
    if context.latest_moment.is_empty() {
        return Err(context_error("context.latest_moment", "must not be empty"));
    }
    check_messages(&context.earlier_continuity, "context.earlier_continuity")?;
    check_messages(&context.latest_moment, "context.latest_moment")?;
    let latest = format!(
        "<latest_moment>\n{}\n</latest_moment>",
        transcript(&context.latest_moment)
    );
    let story = if context.earlier_continuity.is_empty() {
        latest
    } else {
        format!(
            "<earlier_continuity>\n{}\n</earlier_continuity>\n\n{latest}",
            transcript(&context.earlier_continuity)
        )
    };
    if story.chars().count() > MAX_STORY_CONTEXT_CHARS {
        return Err(context_error("context", "exceeds the story context limit"));
    }
    let mut characters = vec![];
    let mut roster = vec![];
    for (i, state) in context.cast_state.iter().enumerate() {
        let at = format!("context.cast_state[{i}].reference_id");
        let reference = base
            .creative
            .references
            .iter()
            .find(|r| r.id == state.reference_id && r.enabled)
            .filter(|_| request.cast.contains(&state.reference_id))
            .ok_or_else(|| context_error(&at, "must be an enabled member of cast"))?;
        let tag = format!(
            "@{}",
            imagine_video_contract::creative::reference_alias(&reference.alias)
        );
        let appearance = &state.current_appearance;
        characters.push(Py::obj([
            ("character_id", Py::str(&state.reference_id)),
            ("reference_tag", Py::str(&tag)),
            (
                "current_appearance",
                Py::obj([
                    ("exposure", Py::str(&appearance.exposure)),
                    ("garments", Py::strs(&appearance.garments)),
                    ("worn_items", Py::strs(&appearance.worn_items)),
                    ("temporary_traits", Py::strs(&appearance.temporary_traits)),
                    (
                        "visible_body_state",
                        Py::strs(&appearance.visible_body_state),
                    ),
                ]),
            ),
        ]));
        roster.push(Py::obj([
            ("character_id", Py::str(&state.reference_id)),
            ("reference_tag", Py::Str(tag)),
            ("fallback_wardrobe", Py::str(&state.fallback_wardrobe)),
        ]));
    }
    if characters.is_empty() && context.protected_facts.is_empty() {
        return Ok(Some(Bridge {
            story,
            grounding: Py::Null,
            opening: Py::Null,
        }));
    }
    use sha2::Digest;
    let digest = Py::Str(format!(
        "sha256:{:x}",
        sha2::Sha256::digest(serde_json::to_vec(context).unwrap())
    ));
    let grounding = director::normalize_grounding(&Py::obj([
        ("schema_version", Py::str("1.0")),
        ("source_digest", digest.clone()),
        ("characters", Py::List(characters)),
        ("protected_facts", Py::strs(&context.protected_facts)),
    ]))
    .map_err(|e| context_error(context_path(&e), e.reason))?;
    let opening = if roster.is_empty() {
        Py::Null
    } else {
        director::normalize_opening(
            &Py::obj([
                ("schema_version", Py::str("1.0")),
                ("source_digest", digest),
                ("characters", Py::List(roster)),
            ]),
            &grounding,
        )
        .map_err(|e| context_error(context_path(&e), e.reason))?
    };
    Ok(Some(Bridge {
        story,
        grounding,
        opening,
    }))
}
fn bridged(request: &Request, base: &VideoProject) -> Result<Option<Bridge>> {
    bridge(request, base).map_err(|r| r.code)
}
/// The longest Animate scene context, as Director's `animate_image` contract.
const MAX_ANIMATE_MESSAGES: usize = 10;
fn animate_error(at: impl Into<String>, reason: &str) -> Rejection {
    located("animate_context_invalid", at, reason)
}
/// Animate's target Scene and its First frame.
fn animate_scene<'a>(
    request: &Request,
    base: &'a VideoProject,
) -> std::result::Result<&'a Scene, Rejection> {
    let scene = request
        .scene_id
        .as_ref()
        .and_then(|id| base.scene(id).ok())
        .ok_or_else(|| Rejection::new("scene_not_found", "scene_id"))?;
    if scene.sections.first().is_none_or(|s| s.start.num != 0) {
        return Err(located(
            "animate_scene_invalid",
            "scene_id",
            "the first Section must start at 0",
        ));
    }
    let frames = scene
        .bindings
        .iter()
        .filter(|b| {
            b.role == InputRole::FirstFrame
                && base
                    .assets
                    .get(&b.asset)
                    .is_some_and(|a| a.kind == MediaKind::Image)
        })
        .collect::<Vec<_>>();
    if frames.is_empty() {
        return Err(located(
            "animate_first_frame_missing",
            "scene_id",
            "the Scene has no First frame image",
        ));
    }
    if !frames.iter().any(|b| b.enabled) {
        return Err(located(
            "animate_first_frame_disabled",
            "scene_id",
            "the Scene's First frame is disabled",
        ));
    }
    Ok(scene)
}
/// The enabled cast reference `id` of the request.
fn cast_reference<'a>(
    request: &Request,
    base: &'a VideoProject,
    id: &str,
) -> Option<&'a imagine_video_contract::creative::VisualReference> {
    base.creative
        .references
        .iter()
        .find(|r| r.id == id && r.enabled)
        .filter(|_| request.cast.iter().any(|c| c == id))
}
/// Each cast state's appearance as Director's grounding normalizes it.
fn animate_states(
    request: &Request,
    base: &VideoProject,
) -> std::result::Result<Vec<(String, Py, String)>, Rejection> {
    let Some(context) = &request.context else {
        return Ok(vec![]);
    };
    let mut characters = vec![];
    let mut roster = vec![];
    for (i, state) in context.cast_state.iter().enumerate() {
        let reference = cast_reference(request, base, &state.reference_id).ok_or_else(|| {
            animate_error(
                format!("context.cast_state[{i}].reference_id"),
                "must be an enabled member of cast",
            )
        })?;
        let tag = format!(
            "@{}",
            imagine_video_contract::creative::reference_alias(&reference.alias)
        );
        let appearance = &state.current_appearance;
        characters.push(Py::obj([
            ("character_id", Py::str(&state.reference_id)),
            ("reference_tag", Py::str(&tag)),
            (
                "current_appearance",
                Py::obj([
                    ("exposure", Py::str(&appearance.exposure)),
                    ("garments", Py::strs(&appearance.garments)),
                    ("worn_items", Py::strs(&appearance.worn_items)),
                    ("temporary_traits", Py::strs(&appearance.temporary_traits)),
                    (
                        "visible_body_state",
                        Py::strs(&appearance.visible_body_state),
                    ),
                ]),
            ),
        ]));
        roster.push(Py::obj([
            ("character_id", Py::str(&state.reference_id)),
            ("reference_tag", Py::Str(tag)),
            ("fallback_wardrobe", Py::str(&state.fallback_wardrobe)),
        ]));
    }
    let digest = Py::Str(format!("sha256:{}", "0".repeat(64)));
    let grounding = director::normalize_grounding(&Py::obj([
        ("schema_version", Py::str("1.0")),
        ("source_digest", digest.clone()),
        ("characters", Py::List(characters)),
        ("protected_facts", Py::List(vec![])),
    ]))
    .map_err(|e| animate_error(context_path(&e), e.reason))?;
    let opening = director::normalize_opening(
        &Py::obj([
            ("schema_version", Py::str("1.0")),
            ("source_digest", digest),
            ("characters", Py::List(roster)),
        ]),
        &grounding,
    )
    .map_err(|e| animate_error(context_path(&e), e.reason))?;
    Ok(grounding
        .at("characters")
        .as_list()
        .unwrap()
        .iter()
        .zip(opening.at("characters").as_list().unwrap())
        .map(|(c, o)| {
            (
                c.at("character_id").text(),
                c.at("current_appearance").clone(),
                o.at("fallback_wardrobe").text(),
            )
        })
        .collect())
}
/// An Animate request's Scene, First frame, context, cast and mentions,
/// checked before anything starts; errors name where.
pub fn animate_check(request: &Request, base: &VideoProject) -> std::result::Result<(), Rejection> {
    if request.provider == EXTERNAL {
        return Err(located(
            "animate_external_unsupported",
            "provider",
            "Animate runs on the provider path only",
        ));
    }
    animate_scene(request, base)?;
    if !(1.0..=15.0).contains(&request.duration_seconds) {
        return Err(animate_error("duration_seconds", "must be from 1 to 15"));
    }
    for (field, unused) in [
        ("brief", !request.brief.is_empty()),
        ("images", !request.images.is_empty()),
        ("section_id", request.section_id.is_some()),
    ] {
        if unused {
            return Err(animate_error(field, "is not used by Animate"));
        }
    }
    for (i, id) in request.cast.iter().enumerate() {
        if cast_reference(request, base, id).is_none() {
            return Err(animate_error(
                format!("cast[{i}]"),
                "must be an enabled reference",
            ));
        }
    }
    for (i, mention) in request.mentions.iter().enumerate() {
        if !request.cast.contains(&mention.reference_id) {
            return Err(animate_error(
                format!("mentions[{i}].reference_id"),
                "must be a member of cast",
            ));
        }
    }
    if let Some(context) = &request.context {
        if !context.earlier_continuity.is_empty() {
            return Err(animate_error(
                "context.earlier_continuity",
                "is not used by Animate",
            ));
        }
        if !context.protected_facts.is_empty() {
            return Err(animate_error(
                "context.protected_facts",
                "is not used by Animate",
            ));
        }
        if context.latest_moment.len() > MAX_ANIMATE_MESSAGES {
            return Err(animate_error(
                "context.latest_moment",
                "must contain at most 10 messages",
            ));
        }
        for (i, message) in context.latest_moment.iter().enumerate() {
            if !["user", "assistant"].contains(&message.role.as_str()) {
                return Err(animate_error(
                    format!("context.latest_moment[{i}].role"),
                    "must be user or assistant",
                ));
            }
        }
    }
    animate_states(request, base)?;
    Ok(())
}
/// A cast member's appearance as one text: its description, then the
/// context's current state, one labelled line each.
fn appearance_text(description: &str, state: Option<&(String, Py, String)>) -> String {
    let mut lines = vec![];
    if !description.pytrim().is_empty() {
        lines.push(description.pytrim().to_owned());
    }
    if let Some((_, appearance, fallback)) = state {
        lines.push(format!("Exposure: {}", appearance.at("exposure").text()));
        for (field, label) in [
            ("garments", "Garments"),
            ("worn_items", "Worn items"),
            ("temporary_traits", "Temporary traits"),
            ("visible_body_state", "Visible body state"),
        ] {
            let items = appearance
                .at(field)
                .as_list()
                .unwrap_or_default()
                .iter()
                .map(Py::text)
                .collect::<Vec<_>>();
            if !items.is_empty() {
                lines.push(format!("{label}: {}", items.join("; ")));
            }
        }
        if !fallback.is_empty() {
            lines.push(format!("Fallback wardrobe: {fallback}"));
        }
    }
    lines.join("\n")
}
fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a.max(1) } else { gcd(b, a % b) }
}
/// Director's `animate_image` request for this Draft, without its image.
pub fn animate_request(draft: &Draft, base: &VideoProject) -> Result<Py> {
    let request = &draft.request;
    let states = animate_states(request, base).map_err(|r| r.code)?;
    let messages = request
        .context
        .as_ref()
        .map(|c| c.latest_moment.as_slice())
        .unwrap_or_default();
    let characters = request
        .cast
        .iter()
        .filter_map(|id| cast_reference(request, base, id))
        .map(|r| {
            Py::obj([
                ("id", Py::str(&r.id)),
                ("name", Py::str(&r.name)),
                (
                    "appearance",
                    Py::Str(appearance_text(
                        &r.description,
                        states.iter().find(|s| s.0 == r.id),
                    )),
                ),
            ])
        })
        .collect();
    let (width, height) = (base.delivery.width, base.delivery.height);
    let divisor = gcd(width, height);
    Ok(Py::obj([
        ("schema_version", Py::Int(1)),
        ("operation", Py::str("animate_image")),
        (
            "scene_context",
            Py::obj([(
                "messages",
                Py::List(
                    messages
                        .iter()
                        .map(|m| {
                            Py::obj([
                                ("role", Py::str(&m.role)),
                                ("name", Py::str(&m.name)),
                                ("text", Py::str(&m.text)),
                            ])
                        })
                        .collect(),
                ),
            )]),
        ),
        ("active_characters", Py::List(characters)),
        (
            "user_instruction",
            Py::obj([
                ("text", Py::str(&request.instruction)),
                (
                    "mentions",
                    Py::List(
                        request
                            .mentions
                            .iter()
                            .map(|m| {
                                Py::obj([
                                    ("text", Py::str(&m.text)),
                                    ("character_id", Py::str(&m.reference_id)),
                                ])
                            })
                            .collect(),
                    ),
                ),
            ]),
        ),
        (
            "audio_request",
            Py::obj([
                ("generate_dialogue", Py::Bool(request.add_dialogue)),
                (
                    "generate_background_sound",
                    Py::Bool(request.add_ambient_sound),
                ),
            ]),
        ),
        (
            "generation_context",
            Py::obj([
                ("duration_seconds", Py::js(request.duration_seconds)),
                (
                    "aspect_ratio",
                    Py::Str(format!("{}:{}", width / divisor, height / divisor)),
                ),
                (
                    "orientation",
                    Py::str(if width >= height {
                        "Landscape"
                    } else {
                        "Portrait"
                    }),
                ),
            ]),
        ),
    ]))
}
pub fn animate_message(draft: &Draft, base: &VideoProject) -> Result<String> {
    Ok(director::build_animate(&animate_request(draft, base)?))
}
/// Accepts an Animate answer: exactly `description`, `prompt` and
/// `ambient_sound`, each a string; nothing is repaired.
pub fn accept_animate(draft: &mut Draft, text: &str) -> Result<()> {
    let value = Py::parse(text).ok_or("animate_response_invalid")?;
    let [description, prompt, ambient_sound] =
        director::animate_content(&value).map_err(|_| "animate_response_invalid")?;
    if [&description, &prompt, &ambient_sound]
        .iter()
        .any(|t| t.len() > 65536)
    {
        return Err("story_text_limit");
    }
    draft.animate = Some(AnimateProposal {
        description,
        prompt,
        ambient_sound,
    });
    Ok(())
}
/// A request's operation-specific inputs, checked before anything starts.
pub fn check(request: &Request, base: &VideoProject) -> std::result::Result<(), Rejection> {
    match request.operation {
        Operation::Animate => animate_check(request, base),
        Operation::Outline => bridge(request, base).map(|_| ()),
        _ if request.context.is_some() => Err(context_error(
            "context",
            "is used by Outline and Animate only",
        )),
        _ => Ok(()),
    }
}
/// The grounding an expansion starts from: each character's current
/// appearance replaced by its opening appearance, as Director's Story bridge
/// client applies it; fallback wardrobe, facts and digest stay as captured.
pub fn effective_grounding(grounding: &Py, opening: Option<&Value>) -> Result<Py> {
    let Some(opening) = opening.and_then(|o| Py::parse(&o.to_string())) else {
        return Ok(grounding.clone());
    };
    let mut grounding = grounding.clone();
    if let Some(Py::List(characters)) = grounding.get_mut("characters") {
        for character in characters {
            let id = character.at("character_id").text();
            let Some(resolved) = opening
                .at("characters")
                .as_list()
                .unwrap_or_default()
                .iter()
                .find(|c| c.at("character_id").text() == id)
            else {
                continue;
            };
            *character.get_mut("current_appearance").unwrap() = Py::obj([
                ("exposure", resolved.at("exposure").clone()),
                ("garments", resolved.at("garments").clone()),
                ("worn_items", resolved.at("worn_items").clone()),
                ("temporary_traits", resolved.at("temporary_traits").clone()),
                (
                    "visible_body_state",
                    resolved.at("visible_body_state").clone(),
                ),
            ]);
        }
    }
    director::normalize_grounding(&grounding).map_err(|_| "story_opening_appearance_invalid")
}
/// A reviewed opening appearance, validated as the outline's answer is:
/// every cast character once, in order, within Director's limits.
pub fn review_opening(draft: &Draft, value: &Value) -> std::result::Result<(), Rejection> {
    let invalid = |at: &str, reason: &str| located("story_opening_appearance_invalid", at, reason);
    let current = draft
        .opening_appearance
        .as_ref()
        .and_then(|o| Py::parse(&o.to_string()))
        .ok_or_else(|| invalid("opening_appearance", "the Draft has no opening appearance"))?;
    let states = draft
        .request
        .context
        .as_ref()
        .map(|c| c.cast_state.as_slice())
        .unwrap_or_default();
    let roster = current
        .at("characters")
        .as_list()
        .unwrap_or_default()
        .iter()
        .map(|c| director::OpeningCharacter {
            character_id: c.at("character_id").text(),
            reference_tag: c.at("reference_tag").text(),
            fallback_wardrobe: states
                .iter()
                .find(|s| s.reference_id == c.at("character_id").text())
                .map(|s| s.fallback_wardrobe.pytrim().to_owned())
                .unwrap_or_default(),
        })
        .collect::<Vec<_>>();
    let value = Py::parse(&value.to_string()).unwrap_or(Py::Null);
    director::validate_opening(&value, &roster).map_err(|e| {
        invalid(
            &e.path
                .replacen("$.opening_appearance", "opening_appearance", 1),
            e.reason,
        )
    })
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
    let bridge = bridged(request, base)?;
    let note = |id: &str| match &notes {
        Notes::Known(notes) => notes
            .iter()
            .find(|n| n.id == id)
            .map(|n| n.note.clone())
            .unwrap_or_default(),
        Notes::Slots => note_slot(id),
    };
    let mut out = Py::obj([
        ("operation", Py::str("story_outline")),
        ("user_instruction", Py::str(&request.instruction)),
        (
            "story",
            Py::str(bridge.as_ref().map_or(&request.brief, |b| &b.story)),
        ),
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
    ]);
    if let Some(bridge) = bridge {
        out.push("grounding_context", bridge.grounding);
        out.push("opening_appearance_context", bridge.opening);
    }
    Ok(out)
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
    if let Some(bridge) = bridged(request, base)? {
        out.push(
            "grounding_context",
            effective_grounding(&bridge.grounding, draft.opening_appearance.as_ref())?,
        );
    }
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
/// Accepts an outline answer to `message`, whose markers it restores; with a
/// context's cast state the answer resolves each opening appearance.
pub fn accept_outline(draft: &mut Draft, text: &str, message: &director::Built) -> Result<()> {
    let request = &draft.request;
    let mut text = text.to_owned();
    let mut opening = None;
    if let Some(roster) = &message.contract.opening {
        let mut value = Py::parse(&text).ok_or("authoring_invalid_json")?;
        let answer = match &mut value {
            Py::Obj(rows) => rows
                .iter()
                .position(|(k, _)| k == "opening_appearance")
                .map(|i| rows.remove(i).1),
            _ => None,
        }
        .ok_or("story_opening_appearance_invalid")?;
        director::validate_opening(&answer, roster)
            .map_err(|_| "story_opening_appearance_invalid")?;
        opening = Some(
            serde_json::from_str::<Value>(&references::restore_all(
                &answer.compact(),
                &message.references,
            ))
            .unwrap(),
        );
        text = value.compact();
    }
    let text = references::restore_all(&text, &message.references);
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
    draft.opening_appearance = opening;
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
    let context_grounding = bridged(request, base)?
        .map(|b| b.grounding)
        .filter(|g| *g != Py::Null);
    if context_grounding.is_some() {
        scene_slots.push("grounding_context");
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
        grounding: context_grounding
            .is_some()
            .then(|| slot("grounding_context"))
            .as_deref(),
    });
    let outline = outline_message(draft, base, Notes::Slots)?;
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
    let mut bundle = json!({
        "profile":{"id":profile.resolved_id,"revision":profile.profile.revision,"prompt_files":profile.profile.prompt_files},
        "limits":limits(),
        "inventory":{"system":system(profile, INVENTORY)?,"messages":request.images.iter().map(|s|json!({"image_id":s.id,"user":inventory_message(s,&request.instruction)})).collect::<Vec<_>>()},
        "outline":{"system":system_for(profile,outline_key(request.add_dialogue),&outline.contract)?,"user":outline.user,"slots":outline_slots,"slot_values":{
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
    });
    if let Some(grounding) = context_grounding {
        bundle["expansion"]["slot_values"]["grounding_context"] = json!({
            "captured": grounding.dumps(),
            "expansions": "the captured grounding with each character's current_appearance replaced by your outline's opening_appearance (exposure, garments, worn_items, temporary_traits, visible_body_state), list items trimmed and repeats dropped, as JSON"});
    }
    Ok(bundle)
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
    accept_outline(draft, &submission.outline, &message).map_err(|code| {
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
    /// The grounding Director's Story bridge client sends after the outline:
    /// each current appearance replaced by the resolved opening appearance.
    #[test]
    fn opened_grounding_matches_the_director_source_oracle() {
        let fixture = Py::parse(include_str!(
            "../../../tests/fixtures/director-story-messages.json"
        ))
        .unwrap();
        let case = |name: &str| {
            fixture
                .at("cases")
                .as_list()
                .unwrap()
                .iter()
                .find(|c| c.at("name").text() == name)
                .unwrap()
                .clone()
        };
        let captured = director::normalize_grounding(
            case("scene_grounded").at("request").at("grounding_context"),
        )
        .unwrap();
        let opened = case("scene_opened");
        let request = opened.at("request");
        let restored = request
            .at("opening_answer")
            .at("opening_appearance")
            .compact()
            .replace("[[DIRECTOR_REF_0001]]", "@keeper")
            .replace("[[DIRECTOR_REF_0002]]", "@visitor");
        let effective =
            effective_grounding(&captured, Some(&serde_json::from_str(&restored).unwrap()))
                .unwrap();
        assert_eq!(
            effective,
            director::normalize_grounding(request.at("grounding_context")).unwrap()
        );
        let mut sent = request.clone();
        *sent.get_mut("grounding_context").unwrap() = effective;
        assert_eq!(
            director::build_scene(&sent, false).unwrap().user,
            opened.at("user").text()
        );
    }
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
