# Session-local timeline runs

The server owns each Run. Start, observe and stop use the current
`X-Imagine-Session`; admission also requires an `Idempotency-Key`.

```http
POST /v1/video/runs
Idempotency-Key: own-run-request
X-Imagine-Session: current-session
Content-Type: application/json

{"schema_version":1,"project_id":"project_own","expected_project_revision":4,"mode":"missing_only","selected_scene":null}
```

Modes:

- `missing_only`: generate scenes without a usable Accepted Take.
- `all_same_seeds`: require a locked Accepted/Outdated Take for every target.
- `all_fresh_seeds`: freeze distinct fresh seeds, excluding the locked seeds.
- `selected_with_dependents`: the selected scene and transitive descendants
  in the current explicit continuity graph, in document sequence order.

Single-scene Generate keeps manual Candidate/Accept semantics. Automatic Runs
accept complete scene results before admitting the next target through the common
FIFO. Image jobs and independent scene jobs retain their ordinary queue order.

`GET /v1/video/runs`, `GET /v1/video/projects/{id}/runs` and
`GET /v1/video/runs/{id}` expose mode, status, frozen seeds, target job IDs and
completed Take IDs. They contain no prompts, media bytes or file paths.

`POST /v1/video/runs/{id}/stop` takes
`{"schema_version":1,"cancel_current":false}` for Stop after current;
`true` also cancels the current queued/running job through normal cancellation.
Errors stop further admission. Completed results remain available for review.

Repeating an identical admission/key retrieves the original Run even after its
completion. Reusing a key with different intent fails. Each project can have one
active Run. An unconfirmed native-UI admission retains its original request/key
for explicit retrieval.

Scene authoring and scene/source Take choices are guarded again before admission
and automatic acceptance. A late edit or manual selection stops the Run and
retains the generated Candidate/Outdated result. Autoaccept cannot replace that
manual choice. Own source acceptance can invalidate downstream selections; the
Run updates only expectations changed by its own review operation.

The project document's optional `locked_takes` index retains the selected seed
source through dependency invalidation and Save/Open. Older documents remain
readable. An old Outdated result without a recorded selection cannot supply a
guessed locked seed.

Run state is memory-only. Closing the UI or changing workspace does not stop
the server's Run. Server sessions do not resume Runs automatically. The UI
marks observations from an ended session Interrupted.

Current acceptance uses synthetic CPU workers through the actual scheduler,
publication, storage and native UI. H3 numerical behavior is unchanged. Full
Imported-scene intent/filtering, historical Director dependency-policy variants,
explicit Plan Review approval and A/B remain part of the wider unfinished plan.
