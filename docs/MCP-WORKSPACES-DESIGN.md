# MCP workspaces: key-scoped agent storage

Status: **Implemented 2026-10-25.** See section 12 for where the implementation differs.

Date: 2026-10-25.

## 1. Today

Agents work with host paths. They place references in the MCP input directory
(`/run/media/veracrypt1/mcp-inputs`), register them by absolute path, create
video projects as folders there, and receive generated images and videos as
absolute `saved_path`s under the encrypted persistent root. So an agent:

- knows where all agent work lives and can list or read earlier work it was
  never given;
- needs filesystem access to the same host paths (containers need mappings);
- leaves files spread over the input directory, the persistent image root and
  project folders, with no grouping by task or date.

## 2. Goals

1. The agent never learns a storage path. Files go in and out only through MCP
   tools and the proxy's HTTP endpoints.
2. All agent work is grouped in a **workspace**. A workspace is reachable only
   with its **workspace key**, which the agent is told to save. Without the key,
   earlier work is unreachable through Imagine: there is no tool that lists
   workspaces.
3. Storage is easy to maintain by hand: one folder per workspace under a date
   folder, with images, videos and video projects inside.
4. The user can find a workspace's key in its folder (and in the UI) and hand it
   to an agent again.

## 3. Storage layout

Under the MCP root (server flag `--mcp-root`, default
`/run/media/veracrypt1/mcp-root`, on the verified encrypted volume):

```
<root>/
  2026-10-25/
    1432-forest-chase-k7q2/          <time>-<title slug>-<4 chars of the ID>
      AGENT-KEY.txt                  the key, plus one line on how to use it
      workspace.json                 id, title, created, key hash, client name
      uploads/                       files the agent uploaded
      images/                        generated and upscaled images
      videos/
        forest-chase/                one Imagine video project (document,
          project.imagine-video …      takes, final videos) as today
```

- The date folder is the creation date; the workspace keeps its folder for its
  whole life, even when used on later days.
- The title comes from the agent's `workspace_create` call (short, slugged,
  optional). Folder names are not private; titles and file names inside stay
  under the UI's normal privacy rules.
- Generated files keep readable names (`0001-scene-2-take-1.mp4`,
  `0007-portrait.png`); the index maps file IDs to them.
- Jobs, job history and normalized previews stay in RAM, as today. The
  workspace folder holds only files the user would want to keep.

## 4. The workspace key

- Format: `imk-` followed by 26 base32 characters (128 bits), for example
  `imk-7q4mxk2a…`. It is a bearer secret, not a session ID: it works across
  agent sessions, proxy restarts and days.
- Stored in plain text only in `AGENT-KEY.txt`, inside the encrypted volume.
  `workspace.json` stores its SHA-256, which the proxy uses to find the workspace
  without reading key files. The proxy builds its key-hash index by scanning
  `<root>/*/*/workspace.json` at start and on a miss.
- Never logged; redacted from request logs and diagnostics, like the MCP token.
- The UI's "Open private project" window lists workspaces with a **Copy agent
  key** button, and can **Replace key** (revokes the old one: rewrites both
  files). Deleting a workspace folder removes it.

## 5. What the agent sees

### MCP tools

| Tool | Purpose |
| --- | --- |
| `workspace_create {title?}` | New workspace; returns `workspace_key` and tells the agent to save it (see §6). Binds this MCP session to it. |
| `workspace_open {workspace_key}` | Binds this MCP session to an existing workspace. Unknown key: `workspace_unknown`, with no hint whether others exist. |
| `workspace_files {kind?, cursor?}` | Lists files of the bound workspace: `file_id`, kind, name, bytes, SHA-256, created, `download_url`. |
| `workspace_upload {name, content_base64}` | Small uploads (up to 4 MiB) for clients without HTTP access. |
| `workspace_view {file_id, max_side?}` | Returns an image content block (resized, default 1024 px) so the agent can look at a result without downloading it. |

Existing tools change from paths to file IDs and need a bound workspace (or an
explicit `workspace_key` argument):

- `register_image {file_id}` and `video_register_reference {file_id}` instead of
  `path`;
- `video_project create {title}` creates `videos/<slug>/` inside the workspace
  instead of taking a folder path;
- `generate`, `get_job`, `video_assembly get` return `file_id`, name, bytes,
  SHA-256 and `download_url` instead of `saved_path`;
- `confirm_saved` is removed; files are already kept in the workspace.

Binding per MCP session keeps the key out of every call. When the MCP session
ends (client restart), the agent calls `workspace_open` again with its saved key.

### HTTP (on the proxy, next to `/mcp`)

```
PUT  /v1/workspace/files?name=ref.png     upload; returns file_id
GET  /v1/workspace/files                  list
GET  /v1/workspace/files/{file_id}        download (Range supported)
Authorization: Bearer imk-…
```

Agents with a shell use `curl` for uploads and downloads, which keeps large
files out of the model's context. Downloads go wherever the agent chooses in
its own working directory. The key is accepted only in the header, never in a
URL, so it does not end up in shell history or logs as part of a URL.

Limits: images (PNG, JPEG, WebP) up to 64 MiB, audio and video up to 2 GiB,
checked by content, not extension; a per-workspace quota setting.

## 6. Agent guidance

Instructions and the `workspace_create` result say, briefly: *Save this
workspace key where you keep notes for this task (for example your project's
notes file). It is the only way to reach this work again; Imagine cannot list
or recover workspaces for you. Ask the user for the key if you have lost it.*
Guidance stops mentioning host paths, the input directory and `saved_path`.

## 7. How hidden is "hidden"

The proxy and server will never reveal a path and offer nothing that reaches
another workspace without its key. That stops an agent that uses the tools as
intended and keeps it from wandering into earlier work.

It is not an operating-system boundary. An agent that runs as the same Unix
user with shell access could still read the root folder directly or call the
server's local UI API (`127.0.0.1:8080`), which lists projects for the UI. To
enforce it, block that access in the agent's sandbox:

- **Claude Code (recommended):** deny file reads under the workspace root and
  network access to port 8080 in the agent's permission or sandbox settings;
  the proxy on 8190 stays allowed. The setup instructions will list the exact
  rules.
- **Stronger, optional:** run Imagine under its own Unix user with the root at
  mode 0700, so the agent's user cannot read it at all. That changes how the
  services are started and who owns the encrypted volume's files, so it is a
  separate decision.

## 8. Server and proxy responsibilities

- **Server** owns workspaces: the layout, key index, file records, quota and the
  existing project storage (`Layout::project_at` for `videos/<slug>/`). New
  endpoints are under `/v1/workspaces/…`, keyed by workspace ID (not the key),
  and the UI uses them for listing and key management.
- **Proxy** owns the agent surface: MCP tools, the HTTP file routes, key-to-ID
  resolution through the server, and binding MCP sessions to workspaces. It
  never returns server paths to the agent.
- Generated images move from the shared persistent image root into the
  workspace's `images/` folder: the server saves MCP jobs' outputs with the
  workspace as their destination.

## 9. Migration

- Existing folders under `mcp-inputs` and images under the persistent root are
  not moved or changed. The UI keeps opening them as today.
- Optional, later: a UI action "Make agent workspace" that moves an existing
  project folder into the dated layout and creates a key.
- Path-based tool arguments are removed rather than kept alongside, because
  keeping them would let agents bypass the workspace boundary.

## 10. Tests

- Key format, hashing, index rebuild after restart, unknown and revoked keys.
- Isolation: a key reaches only its workspace's files and projects; file IDs
  from another workspace return `file_not_found`.
- No response, error or log contains a host path or a key.
- Upload limits, content checks, Range downloads, quota.
- MCP flow end to end: create, upload by HTTP, register by file ID, generate,
  view, download, reopen with the saved key in a new MCP session.
- UI: workspace list, Copy agent key, Replace key.

## 11. Decisions (2026-10-25)

1. **Enforcement:** soft hiding plus agent sandbox rules. Claude Code's user
   settings deny reads, edits and sandboxed commands on the whole encrypted
   volume (`/run/media/veracrypt1`); the user configures Codex the same way.
2. **Key use:** bind per MCP session with `workspace_open`.
3. **Root:** `/run/media/veracrypt1/mcp-root` with the dated layout above.
4. **Scope:** the MCP root is for agent workspaces only. Projects created in the
   UI also get date folders: new project IDs carry their creation date
   (`project_20261025_<uuid>`) and live in
   `<private root>/projects/2026-10-25/<id>/`; existing projects keep their
   folders.
5. **Old path tools:** removed.

## 12. Implementation notes

- **Key on every call, not per MCP session.** The proxy speaks stateless MCP
  (newer protocol versions have no sessions), so there is no session to bind.
  Every workspace, image and video tool takes `workspace_key`; `workspace_open`
  checks a key and lists the workspace's files and video projects. The catalog
  tools (`list_profiles`, `describe_profile`, `list_video_profiles`,
  `describe_video_profile`, `video_capabilities`) need no key.
- **Confinement.** Every project, Draft, run and assembly plan an agent names is
  checked against its workspace (the project's folder must be inside it);
  reference IDs and image jobs belong to the workspace that made them; a file
  whose Imagine path is outside the workspace is never read.
- **Reopening.** A workspace project that is not open in the current Imagine
  session (after a restart) is reopened from its folder on first use.
- **Retry-safe create.** `video_project create` with the title of a project
  this session already made in the workspace returns that project.
- **No paths.** Results replace Imagine paths inside the workspace with
  `file_id`, `name` and `download_url`, and drop every other host path.
- **Server.** `src/workspaces.rs` (store, keys, file index, uploads),
  `src/video/workspace_api.rs` (`/v1/workspaces…` routes), `--mcp-root`
  (default `/run/media/veracrypt1/mcp-root`, an approved project root). Image
  jobs from the proxy carry the workspace's image folder as their destination.
- **Proxy.** `proxy/src/mcp_workspace.rs` (tools, scrubbing, HTTP routes
  `GET/PUT /v1/workspace/files`, `GET /v1/workspace/files/{id}` with the key as
  bearer and the same Host check as `/mcp`). `confirm_saved` and the
  `mcp_input_dir` setting are removed.
- **UI.** The Open private project window lists agent workspaces with **Copy
  agent key** (copied, never shown) and **Replace key** (confirmed first), and
  labels workspace projects "Agent · <workspace title>".
- **UI projects.** New project IDs are `project_YYYYMMDD_<uuid>` and live in
  `projects/YYYY-MM-DD/<id>/`; older projects keep their folders.
- **Sandbox.** Claude Code's user settings deny reads, edits and sandboxed
  commands on `/run/media/veracrypt1`.
- **Guidance.** Clients cap MCP server instructions (Claude Code cut ours near
  4,080 characters and marked them truncated; tool descriptions arrive whole).
  The instructions (`proxy/mcp/instructions.txt`) stay short: the workspace rule,
  the nsfw and `job_id` rules, and "call `imagine_guide` first". The full guidance
  is `proxy/mcp/guide.txt`, returned by the `imagine_guide` tool and included in
  every `workspace_create` and `workspace_open` result. A test keeps the
  instructions at 3,500 characters or less. `video_capabilities` reports Draft
  as the default canvas, and project views show each scene's `preset` and
  `preset_state`.

