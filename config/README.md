# Imagine configuration

Every editable profile, template and guide lives here. The server and the UI use
`<project>/config` by default and the proxy launcher passes the same folder; each
accepts `--config DIR` for another root. Most files are read once when a process
starts, so **restart the affected process after an edit** (the table says which).
A missing or invalid required file stops that process at startup with its path.

| Path | Contents | Read by | After an edit |
| --- | --- | --- | --- |
| `image/profiles/*.yaml` | Image generation profiles (Qwen, Turbo, MOSS, character sheet, upscale); see [image/README.md](image/README.md) | Server (`--profiles` overrides the folder) | Restart server |
| `image/guidance/<id>.md` | Prompt guides returned by `describe_profile` for profiles without inline guidance | Server | Restart server |
| `image/character-sheet/` | The server's instructions for the portrait and turnaround passes | Server | Restart server |
| `video/presets.yaml` | Your scene presets (git-ignored; copy `presets.example.yaml`) | Server (`--video-presets` overrides the file) | None: read on each use |
| `video/prompt-profiles/<id>/` | Video prompt profiles: `profile.json` and the step templates | Server | None: read on each use |
| `video/agent-guidance.txt` | Guidance returned to agents with a video prompt profile | Server | Restart server |
| `video/prompt-catalog/*.yaml` | Director prompt catalog | UI | Restart UI |
| `mcp/instructions.txt`, `mcp/guide.txt` | MCP server instructions and the full agent guide | Proxy | Restart proxy |
| `mcp/tools.json` | MCP tool catalog and argument schemas; keep in step with the proxy code | Proxy | Restart proxy |
| `mcp/prompts/` | MCP prompts, such as `video_interactive` | Proxy | Restart proxy |
| `proxy/model-profiles.json` | LLM model profiles for the proxy (`--model-profiles` overrides it) | Proxy | Restart proxy |

Retired image presets are kept in [docs/archive/image-profiles](../docs/archive/image-profiles/).
