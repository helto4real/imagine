// Director's browser Story request helpers on synthetic inputs; source only,
// no browser, services or runtime data. Writes
// tests/fixtures/director-story-browser.json.
import { writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
const D = "/home/thhel/git/comfyui-helto-director";
const contract = await import(`${D}/web/timeline/prompt_studio_contract.js`);
const studio = await import(`${D}/web/timeline/prompt_studio.js`);
const storyCast = await import(`${D}/web/timeline/story_cast.js`);

// Neutral inputs: the Rust test builds the same project from them.
const scene = (id, start, end, prompt, sections) => ({ id, start, end, ...prompt, sections });
const continuationCases = [
  {
    name: "three_scenes_unsorted_sections",
    references_enabled: true,
    references: [
      { alias: "keeper", kind: "character", description: " Own fictional keeper ", enabled: true },
      { alias: "visitor", kind: "character", description: "Own visitor", enabled: true },
      { alias: "lamp", kind: "object", description: "Own lamp", enabled: true },
      { alias: "ghost", kind: "character", description: "Own ghost", enabled: false },
      { alias: "absent", kind: "character", description: "Own absent", enabled: true },
    ],
    scenes: [
      scene("own_c", 8, 12, { title: "", summary: "", description: " Own harbour at dusk @lamp ", ambient_sound: "Gulls", non_diegetic_music: "" }, [
        { id: "own_b", start: 10, end: 12, prompt: "  " },
        { id: "own_a", start: 8, end: 10, prompt: "@keeper waits\n" },
      ]),
      scene("own_a", 0, 4, { title: "Own Arrival", summary: " @visitor lands. ", description: "", ambient_sound: "", non_diegetic_music: "" }, []),
      scene("own_b", 4, 8, { title: "", summary: "", description: "Own middle", ambient_sound: "", non_diegetic_music: "" }, []),
    ],
  },
  {
    name: "recent_summaries_are_the_two_before_the_last",
    references_enabled: false,
    references: [{ alias: "keeper", kind: "character", description: "Own keeper", enabled: true }],
    scenes: [1, 2, 3, 4].map((n) => scene(`own_${n}`, (n - 1) * 4, n * 4,
      { title: n === 3 ? "" : `Own ${n}`, summary: `@keeper summary ${n}`, description: "", ambient_sound: "", non_diegetic_music: "" },
      [{ id: `own_${n}_s`, start: (n - 1) * 4, end: n * 4, prompt: `Own prompt ${n}` }])),
  },
  {
    name: "same_start_orders_by_end_then_id",
    references_enabled: true,
    references: [],
    scenes: [
      scene("own_z", 0, 4, { title: "Z", summary: "", description: "Own z", ambient_sound: "", non_diegetic_music: "" }, []),
      scene("own_y", 0, 4, { title: "Y", summary: "", description: "Own y", ambient_sound: "", non_diegetic_music: "" }, []),
      scene("own_x", 0, 2, { title: "X", summary: "", description: "Own x", ambient_sound: "", non_diegetic_music: "" }, []),
    ],
  },
  {
    name: "no_authored_text",
    references_enabled: true,
    references: [],
    scenes: [scene("own_empty", 0, 4, { title: "Own", summary: " ", description: "", ambient_sound: "", non_diegetic_music: "" },
      [{ id: "own_s", start: 0, end: 4, prompt: " " }])],
  },
];
const timeline = (c) => ({
  project: { metadata: { visual_references: c.references, visual_references_enabled: c.references_enabled } },
  sequence: {
    shots: c.scenes.map((s) => ({
      shot_id: s.id, start_time: s.start, end_time: s.end, name: s.title,
      scene_prompt: { summary: s.summary, description: s.description, ambient_sound: s.ambient_sound, non_diegetic_music: s.non_diegetic_music },
      section_ids: s.sections.map((x) => x.id),
    })),
  },
  director_track: {
    sections: c.scenes.flatMap((s) => s.sections.map((x) => ({ item_id: x.id, prompt: x.prompt, start_time: x.start, end_time: x.end }))),
  },
});
const continuation = continuationCases.map((c) => {
  const result = contract.storyContinuationAvailability(timeline(c));
  return { ...c, context: result.available ? result.context : null };
});

const timingCases = [
  [20, [[1, [1, 2]], [2, [1]]]],
  [10, [[1, [1, 1, 1]]]],
  [0.3, [[1, [1]], [1, [1]]]],
  [29.9, [[100, [1]], [1, [3, 1]]]],
  [7, [[1, [0.5, 0.25, 2]]]],
  [44.5, [[1, [1]], [1, [2]], [1.5, [1, 1]]]],
];
const timing = timingCases.map(([duration, scenes]) => {
  const outline = {
    scenes: scenes.map(([weight, sections], i) => ({
      weight,
      sections: sections.map((w, j) => ({ id: `scene_00${i + 1}_section_00${j + 1}`, weight: w })),
    })),
  };
  const timed = contract.allocateStoryTimings(outline, duration);
  return { duration, scenes, timing: timed.scenes.map((s) => studio.promptStudioDialogueTiming(s)) };
});

// Imagine images, and the Director visuals and directed beats they stand for.
const beats = [
  {
    images: [
      { id: "own_setting", usage: "setting_reference" },
      { id: "own_start", usage: "start_frame" },
      { id: "own_m2", usage: "required_moment" },
      { id: "own_end", usage: "end_frame" },
      { id: "own_m1", usage: "required_moment" },
    ],
    visuals: [
      { id: "own_setting", usage: "story_reference", note: "" },
      { id: "own_start", usage: "story_reference", note: "" },
      { id: "own_m2", usage: "required_moment", note: "" },
      { id: "own_end", usage: "story_reference", note: "" },
      { id: "own_m1", usage: "required_moment", note: "" },
    ],
    beats: [{ id: "beat_1", start_image_id: "own_start", end_image_id: "own_end", cue: "" }],
  },
].map((c) => ({ ...c, records: studio.storyVisualBeatPromptRecords(c.visuals, c.beats) }));

const members = [
  { title: "Own Keeper", alias: "keeper", description: " Own keeper ", language: "Swedish", voice: " calm " },
  { title: "keeper2", alias: "Keeper2", description: "", language: "", voice: "" },
  { title: "Dup", alias: "KEEPER", description: "Own duplicate", language: "", voice: "" },
  { title: "", alias: "visitor", description: "Own visitor", language: "English", voice: "" },
];
const candidates = members.map((m, i) => ({
  key: `own_${i}`, selected: true, selectionOrder: i, title: m.title,
  reference: { alias: m.alias, description: m.description, default_language: m.language, voice_profile: { description: m.voice } },
}));
const sceneLayout = {
  id: "scene_001", title: "Own", summary: "@keeper2 waits", description: "", ambient_sound: "", non_diegetic_music: "",
  weight: 1, image_ids: [], boundary_from_previous: "hard_cut",
  sections: [{ id: "scene_001_section_001", summary: "", prompt: "Then\n@visitor and mail@keeper", weight: 1, image_ids: [], first_frame_id: "", last_frame_id: "" }],
};
const cast = {
  members,
  records: storyCast.storyCastPromptRecords(candidates),
  scene: sceneLayout,
  scene_records: storyCast.storyCastPromptRecordsForScene(sceneLayout, candidates),
};
const draftScene = {
  title: " Own ", summary: "", description: " d ", ambient_sound: "", non_diegetic_music: " m ",
  sections: [{ prompt: " a " }, { prompt: "  " }, { prompt: "b" }],
};
const previous = { scene: draftScene, context: contract.storyScenePreviousContext(1, [draftScene], null) };

const commit = execFileSync("git", ["-C", D, "rev-parse", "HEAD"], { encoding: "utf8" }).trim();
const out = new URL("../tests/fixtures/director-story-browser.json", import.meta.url);
writeFileSync(out, `${JSON.stringify({
  source: `comfyui-helto-director ${commit} web/timeline/prompt_studio*.js, story_cast.js`,
  continuation, timing, beats, cast, previous,
}, null, 1)}\n`);
