# Interactive video mode for MCP agents

Status: **Implemented 2026-10-25.**

Date: 2026-10-25.

## 1. Goal

Today an agent produces a video in one pass: it picks the settings, writes the
whole storyboard and renders. In **interactive mode** the agent works with the
user instead: it shows the starting settings, settles the characters, shapes the
story and the outline together with the user, and renders only after an explicit
go. Headless production stays the default.

## 2. Activation

1. **MCP prompt `video_interactive`** (optional argument `idea`). MCP clients list
   server prompts; Claude Code shows it as `/mcp__imagine__video_interactive`. Its
   text tells the agent to follow the interactive steps of the guide, with the
   idea as the starting point.
2. **Plain words.** The guide tells the agent to use interactive mode when the
   user asks for it: "interactive", "step by step", "let's plan it together",
   "ask me first" or similar. This works in clients without MCP prompts.

Without either, the agent produces headlessly as today.

## 3. Defaults

The agent shows real values, not guesses. `video_capabilities` gains an
`mcp_defaults` block, filled by the proxy from its own policy and Imagine's
catalog:

| Setting | Default | Source |
| --- | --- | --- |
| Total length | **15 s** | proxy policy (new) |
| Size | 16:9 landscape, Draft | proxy policy (exists) |
| Prompt profile | the profile marked default (`normal`) | `list_video_profiles` |
| Scene preset | Default ref2v (Default i2v for one given start image) | preset catalog |
| Music | off | guide |
| Dialogue | off | storyboard default |
| Characters without a reference image | ask (interactive); see section 5 (headless) | this design |
| Limits | at most 15 s per scene, sections per scene, scene count | capabilities |

The **15 s total** applies in both modes whenever the user gives no length. The
storyboard request's `duration_seconds` defaults to it, and the guide says so.

## 4. Interactive steps

Each step ends at a checkpoint: the agent shows a short summary and waits for the
user before going on. The GPU is used only after the user's choice in step 4
(character sheets) and go in step 6 (render).

1. **Settings.** A compact card of the section 3 values and the available prompt
   profiles and scene presets (keys and one-line descriptions). The user changes
   any of them or accepts. Images the user supplies for characters are uploaded
   and noted here.
2. **Story.** One or two short premises (two or three sentences each). The user
   picks, merges or rewrites one.
3. **Outline.** The scenes with title, duration, boundary (hard cut or continuous
   shot), the characters in each and a one-line summary, totalling the agreed
   length. The user edits until it is right.
4. **Characters.** Once the outline is accepted, the agent asks, for each
   character without a supplied image (or once for all of them), whether to
   generate a character sheet or describe the character in the prompts
   (section 5). Chosen sheets are generated in one batch, shown with
   `workspace_view` and regenerated on request; accepted sheets and supplied
   images are registered as cast.
5. **Prompts and dry run.** The agent writes the full storyboard (inventory,
   outline, expansions, production block), runs the dry run and reports: per
   scene, model, preset, continuity and warnings. It offers to show any scene's
   full prompt.
6. **Render.** On the user's go: `video_run`, then progress until done.
7. **Review.** The agent reports the takes and offers: assemble the final video,
   regenerate a scene (new seed), change a scene's prompt and render it again, or
   stop. Assembly follows the user's choice.

The agent keeps each summary short and repeats only what changed. If the user
says "go ahead" or "you decide" at any checkpoint, the agent finishes the rest
headlessly with the current choices.

## 5. Characters without a reference image

When the user describes a character but supplies no image, there are two ways to
cast them:

- **Character sheet.** Imagine generates a sheet (the `qwen-character-sheet`
  profile: a portrait and a full-body turnaround on one image) from the
  character's appearance and clothing, and the whole sheet becomes the character's
  identity image. Best when the character appears in several scenes or must stay
  consistent.
- **Described in the prompt.** No image; the cast entry and the prompts carry the
  description. Faster, uses no extra GPU time, and keeps less consistency between
  scenes.

**Interactive mode asks** after the outline is accepted, for each such character
(or once for all of them): "Generate a character sheet for X, or describe X in
the prompts?" The outline shows which scenes each character is in, so the user
can weigh consistency against time.

**Batch generation.** All chosen sheets are generated in **one `generate_batch`
call** with the character-sheet profile, one item per character, so the GPU is
handed over once and the LLM is not reloaded between sheets. Each item carries
the character's appearance and clothing. The results are workspace files; the
agent shows them, regenerates any the user rejects (a second, smaller batch), and
registers each accepted sheet with `video_register_reference` as that
character's cast identity image.

**Headless mode** cannot ask. Proposed default: generate sheets in one batch when
a described character appears in more than one scene, and describe in the prompts
otherwise; the user's request overrides it ("no character sheets", "make sheets
for everyone").

## 6. Changes

| Part | Change |
| --- | --- |
| Proxy: MCP prompts | `prompts/list` returns `video_interactive` (argument `idea`); `prompts/get` returns its text. |
| Proxy: `video_capabilities` | Adds `mcp_defaults` (section 3). |
| Proxy: storyboard start | `duration_seconds` defaults to 15 when omitted. |
| `proxy/mcp/guide.txt` | New "Interactive video" section (sections 2, 4 and 5) and the character choice and the 15 s default in "Video production". |
| Tool descriptions | `video_storyboard` and `generate_batch` mention the 15 s default and the batch character sheets. |
| Agent video guidance | `agent_video.txt` mentions sheets as cast identity images and text-only cast. |

No new storyboard or rendering machinery is needed: the dry run already validates
a plan without committing it, and the conversation carries the iteration.

## 7. Tests

- Proxy: `prompts/list` and `prompts/get` shapes; `mcp_defaults` values;
  `duration_seconds` defaulting; the guide's interactive section exists and the
  instructions stay under the size limit.
- End to end (synthetic): a character sheet batch result registered as cast by
  file ID.
- Live: one interactive session driven by a scripted user (accept settings, pick
  premise 2, shorten scene 1, generate sheets for two described characters,
  approve render), checking that no GPU work starts before each approval.

## 8. Decisions (2026-10-25)

1. **Headless default for described characters:** a character sheet when the
   character appears in more than one scene, otherwise described in the prompts.
   The user's request overrides it.
2. **Identity image from a sheet:** the whole sheet (portrait and turnaround) is
   registered as the cast identity image.
3. **When to ask about sheets:** the agent asks whether to generate character
   sheets or describe the characters in the prompts after the outline is
   accepted (step 4), then generates the chosen sheets in one batch.

## 9. Implementation notes

- `proxy/mcp/prompts/video_interactive.txt` is the prompt text; `prompts/list`
  and `prompts/get` in `proxy/src/mcp.rs` serve it (idea up to 4,000 characters).
- `video_capabilities` adds `mcp_defaults` (`proxy/src/mcp_video.rs`): 15 s,
  Draft 16:9, the profile marked default, the default reference preset, no music,
  no dialogue, `qwen-character-sheet` and the character rules. Storyboard start
  fills `duration_seconds` with 15 when it is missing.
- `proxy/mcp/guide.txt` has the "Interactive video" section and the character
  and length rules in "Video production"; `agent_video.txt` says a cast image may
  be a character sheet and how text-only characters are written.

