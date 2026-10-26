# Rerunning individual scenes through MCP

Status: **Implemented 2026-10-25.** Decisions in section 12.

Date: 2026-10-25.

## 1. Goal

After a render, the user often wants to redo one scene: the same prompt with a
new seed, or a changed prompt with a new seed. An agent should be able to do that
in the existing project, and Imagine should work out which other scenes must be
rendered again because they continue from it. Nothing else is rendered, and the
user can go back to the earlier version.

## 2. What exists today

Verified in the code on 2026-10-25:

| Piece | Today | Gap |
| --- | --- | --- |
| Dependents | `video_run` mode `selected_with_dependents` renders the selected scene and every scene that continues from it, transitively (`src/video/runs.rs` `dependency`: explicit continuity or the resolved boundary). | Only one selected scene per run. |
| Seeds | A target reuses the seed of the scene's current or locked take; a fresh seed only when the scene has no take or the mode is `all_fresh_seeds`. | **A new seed for one scene is impossible.** Rerunning an unchanged scene repeats its seed and wastes the GPU on a near-identical clip. |
| Prompt changes | `video_project edit` with `UpdateSection` (the whole section: id, start, duration, prompt) or `SetCreative` for the scene summary. | Works, but the agent must resend timing to change text, and the edit and the run are two revisions. |
| Invalidation | A dependent's input revision includes its source's accepted take, so a new source take makes the dependent's take outdated; `review::invalidate_changed` clears it and remembers it in `locked_takes`. | None; this is the base of the design. |
| Run safety | `run_inputs_changed` and `run_take_selection_changed` stop a run when its inputs change underneath it. | None. |
| Takes | Every take is kept (Accepted, Candidate, Outdated, Rejected). The server has a review API (accept, reject). | MCP shows only take id, scene and status: no seed, no order, no preview, and no way to select an earlier take. |

The UI has the same seed limitation (Director's selected-scene run).

## 3. The dependency rule

Scene B **depends on** scene A when B is a continuous shot whose source is A: an
explicit continuity (motion context, last frame to first frame, with or without
carried audio) or a resolved incoming boundary. Dependency is transitive. A hard
cut ends a chain. Earlier scenes are never affected.

```
S1 ─cut─ S2 ═cont═ S3 ═cont═ S4 ─cut─ S5

rerun S3 → S3, S4
rerun S2 → S2, S3, S4
rerun S5 → S5
rerun S1 and S3 → S1, S3, S4   (one run, timeline order)
```

The rerun set is the selected scenes plus all their dependents, each scene once,
in timeline order, so a source always finishes before its continuation starts
(the scheduler already waits for I2V last-frame sources).

## 4. Seeds

Each target gets a seed policy:

- **`fresh`**: a new random seed, unique within the project.
- **`same`**: the seed of the scene's current take (or of the locked take when
  the current one was cleared).
- **`from_take`** `{take_id}`: the seed of an earlier take of that scene, for
  "the look of take 2 with the new prompt".

Defaults:

| Target | Default | Why |
| --- | --- | --- |
| Selected scene, prompt unchanged | `fresh` | The point of the rerun is a new result. |
| Selected scene, prompt changed | `fresh` | Matches "new seed and changed prompt"; `same` is available to keep the composition and change only the text. |
| Dependent | `same` | Its own prompt is unchanged; only its start (frames, motion context, audio) changes, so the composition stays as close as possible. `fresh` is available. |

A selected scene with `same`, unchanged inputs and an existing current take would
reproduce that take, so the plan rejects it with `rerun_would_repeat_take` before
any GPU work. A scene without an earlier take always gets `fresh`. Each run target
reports its `seed_source`: `fresh`, `locked_take` or `take` (from an earlier take).

## 5. Run request

The run contract gains a mode `rerun` (the existing modes are unchanged):

```json
{
  "schema_version": 1,
  "project_id": "…",
  "expected_project_revision": 12,
  "mode": "rerun",
  "rerun": {
    "scenes": [
      {"scene_id": "s3", "seed": "fresh",
       "prompt": {"sections": [{"section_id": "s3a", "prompt": "…"}],
                  "summary": "…"}}
    ],
    "dependents_seed": "same"
  }
}
```

- `scenes`: one or more selected scenes, each with an optional `seed`
  (`fresh`, `same` or `{"from_take": "…"}`) and optional `prompt` changes. A
  section change replaces only that section's prompt text; timing stays. The scene
  description and the summary, ambient sound and music fields of the scene prompt
  can change the same way.
- `dependents_seed`: `same` (default) or `fresh` for every dependent. A scene that
  is both selected and a dependent of another selected scene uses its own entry.
- Prompt changes and the run are **one transaction**: the server applies the edits,
  validates the project, plans the run and only then commits the new revision and
  starts. If planning fails, nothing changes. The `job_id` idempotency covers both.
- Edited scenes get `preset_state: modified`, as with any other edit.

## 6. Plan before the GPU

`video_run` gains a `plan` action with the same request. It changes nothing and
uses no GPU, and returns:

```json
{
  "valid": true,
  "targets": [
    {"scene_id": "s3", "title": "…", "reason": "selected", "seed": "fresh",
     "prompt_changed": true, "previous_take": "take_…", "planned_seconds": 5},
    {"scene_id": "s4", "title": "…", "reason": "continues_from", "source": "s3",
     "seed": "same", "previous_take": "take_…", "planned_seconds": 5}
  ],
  "unchanged": ["s1", "s2", "s5"],
  "total_seconds": 10,
  "warnings": ["final_video_outdated", "dependent_opening_may_mismatch:s4"]
}
```

Warnings:

- **`final_video_outdated`**: a final video was assembled from takes this run
  replaces. The old file stays in the workspace; the agent offers to assemble again
  after the run.
- **`dependent_opening_may_mismatch:<scene>`**: the selected scene's prompt changed
  and a dependent continues from it. The dependent's opening prompt may still
  describe the old ending; the agent reads it and proposes an adjustment (as a
  prompt change on that scene in the same request) before the user says go.
Interactive mode shows this as a card ("Scene 3 with a new seed and your new
prompt; scene 4 continues from it and is rendered again with its own seed;
10 s of video; the final video must be assembled again") and waits for go.
Headless mode plans, then starts.

## 7. Takes after the run

New takes are accepted automatically, as today. The replaced takes are kept: the
selected scene's old take becomes a Candidate, each dependent's old take becomes
Outdated.

A new **`video_take`** tool gives the agent the takes:

- **`list`** `{project_id, scene_id?}`: per take: id, scene, status, current, seed,
  created order, whether its prompt matches the scene's current prompt, and which
  source take it continued from.
- **`view`** `{take_id}`: three frames (first, middle, last) as image blocks.
  For a continuous scene it also returns the source take's last frame next to this
  take's first frame, so the agent and the user can judge the join. No video bytes
  through MCP; frames are taken with the server's existing frame capture.
- **`select`** `{take_id}`: go back to an earlier take (section 8).

## 8. Going back to an earlier version

`select` makes an earlier take current and restores its chain:

1. The chosen take becomes Accepted for its scene (server review Accept).
2. For each dependent, in order: if one of its kept takes was generated from the
   now-current source take and its input revision matches the current project, it
   becomes current again. Otherwise the dependent has no current take and the
   response lists it under `needs_render`.
3. Response: the restored takes, `needs_render`, `final_video_outdated` and the
   project.

So "go back to the first version of scene 3" also brings back the scene 4 that
continued from it, without rendering. If the prompt was changed since, the old
take no longer matches the current prompt; `select` then also restores the prompt
it was made with only when the agent passes `restore_prompt: true`, otherwise it
is refused with `take_prompt_changed`. New takes store their prompt text for this
(`authored` in the take provenance); a take made before this change cannot restore
its prompt (`take_prompt_unknown`). When other inputs changed (timing, settings or
an earlier scene's take), the take cannot be selected (`take_inputs_changed`).
Selecting the current take is `take_already_current`.

## 9. Failures

- A run that fails part-way keeps what finished: if S3 succeeded and S4 failed,
  S3's new take is current and S4 has no current take. The failure response names
  S4; `rerun` of S4 (`same` seed) or `missing_only` completes the chain. The agent
  never retries a failure unchanged (existing guide rule).
- Stop and cancel work as today; finished targets stay accepted.
- After a timeout: same `job_id` and request, or `get` with the run ID.

## 10. Agent guidance

- Guide, "Video production": how to rerun a scene, the dependency rule in one
  paragraph, always `plan` first and report the targets, adjust dependent openings
  on `dependent_opening_may_mismatch`, offer assembly after a rerun, `video_take
  view` before asking the user to compare, `select` to go back.
- Interactive step 7 (Review) maps to: regenerate a scene (rerun, `fresh`), change
  a scene's prompt and render it again (rerun with `prompt`), go back to an earlier
  take (`select`), assemble.
- `video_run` and `video_take` tool descriptions carry the request shape.

## 11. Changes

| Part | Change |
| --- | --- |
| Contract (`crates/video-contract/src/runs.rs`) | Mode `rerun`, `Rerun {scenes, dependents_seed}`, seed policy, prompt changes; `Target` gains `reason` and `source`. |
| Server runs (`src/video/runs.rs`) | Plan the rerun set (multi-scene union of dependents), seed policies, `rerun_would_repeat_take`, edits in the same transaction, plan-only endpoint. |
| Server takes | Take listing with seed, order and source take; frame view; `select` with chain restore (`review` Accept plus the dependent matching in section 8). |
| Proxy MCP | `video_run plan` and the `rerun` request; new `video_take` tool (list, view, select), workspace-confined like the other video tools. |
| Guide, tools.json, interactive prompt | Section 10. |
| UI (optional, same server API) | Director's "Regenerate scene" can offer new seed / same seed and show the affected scenes. |

## 12. Decisions (2026-10-25)

1. **Seed of dependents:** `same` by default; `fresh` on request.
2. **New takes:** accepted automatically as today; `select` goes back.
3. **Prompt changes:** inside the rerun request, one transaction with the run.
4. **Going back with a changed prompt:** refused unless `restore_prompt: true`.

## 13. Implementation notes

- Contract: `Mode::Rerun`, `Rerun`, `RerunScene`, `Seed`, `PromptChange` and
  `Target.reason` in `crates/video-contract/src/runs.rs`; `TakeState.inputs_current`
  in `review.rs`.
- Server: `src/video/runs.rs` plans the rerun set, seeds, prompt edits (applied to a
  copy with `ProjectDocument::edited`) and the plan report;
  `src/scheduler_video_runs.rs` publishes the edits with the run
  (`ProjectSession::publish`) and serves `POST /v1/video/runs/plan`.
  `src/video/review.rs` `select` and `src/video/takes_api.rs`
  (`POST /v1/video/projects/{id}/takes/{take}/select`,
  `GET …/takes/{take}/frames`, frames by ffmpeg at most 768 px).
- Proxy: `video_run plan`, the `video_take` tool (list from the project snapshot,
  view as image blocks, select), tool descriptions and the guide.
- Not done: the optional Director UI changes in section 11.

## 14. Tests

- Planning: rerun sets for the section 3 chains, including two selected scenes and
  a selected scene that is also a dependent; seed defaults and overrides;
  `rerun_would_repeat_take`; prompt edits applied atomically and rolled back on a
  planning error; plan-only changes nothing.
- Execution (synthetic jobs): S3 rerun outdates S4's old take and renders S4 with
  its old seed; a failure at S4 leaves S3 current and S4 in `needs_render`.
- `select`: going back to S3's old take restores S4's matching old take; a
  dependent without a matching take lands in `needs_render`; `take_prompt_changed`.
- Proxy: request and response shapes, workspace confinement of `video_take`, frame
  view returns image blocks.
- Live: render a three-scene chain, rerun the middle scene with a new prompt,
  check that only it and its continuation render, then go back to the first
  version without rendering.
