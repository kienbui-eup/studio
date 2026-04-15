# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

ComfyUI v0.18.1 instance configured as **eStudio** — an automated video production studio. Generates short-form videos end-to-end using AI models on a remote vast.ai GPU (RTX 6000Ada).

## Common Commands

```bash
# Run ComfyUI server
python main.py --listen 0.0.0.0 --port 8188

# Run all tests
python -m pytest tests-unit tests

# Run unit tests only (no GPU needed)
python -m pytest tests-unit

# Run a single test file
python -m pytest tests-unit/path/to/test_file.py -v

# Skip inference/execution tests
python -m pytest -m "not inference" -m "not execution"

# Lint with ruff
ruff check .

# Regenerate video workflow JSON
OPENAI_API_KEY=sk-xxx python generate_video_workflow_v2.py > workflow-video-production-v2.json

# Build DirectorsConsole React frontend for ComfyUI integration
cd custom_nodes/ComfyUI-DirectorsConsole/CinemaPromptEngineering/frontend
BUILD_MODE=comfyui npm run build
# Output: ../../web/app/ (served at /directors/app/)
```

## Architecture

### ComfyUI Core

- **`main.py`** — Entry point. Sets up GPU/CUDA, loads custom nodes, starts server.
- **`server.py`** — aiohttp web server with WebSocket. Routes, middleware, asset serving.
- **`execution.py`** — DAG execution engine. Graph traversal, caching (Basic/Hierarchical/LRU/RAM), validation.
- **`nodes.py`** — Built-in node definitions (CLIPTextEncode, KSampler, VAE ops, etc.).
- **`comfy/`** — Core ML library: model loading, samplers, CLIP, memory management.
- **`comfy_extras/`** — Extended node implementations (ControlNet, IPAdapter, etc.).
- **`app/`** — Manager classes: `UserManager`, `ModelFileManager`, `CustomNodeManager`, `SubgraphManager`.
- **`api_server/`** — REST API routes and services.
- **`custom_nodes/`** — Plugin directory. Auto-loaded on startup. Each node registers via `NODE_CLASS_MAPPINGS`.

### Custom Node System

Nodes register through `__init__.py` exporting:
- `NODE_CLASS_MAPPINGS` — `{"NodeName": NodeClass}`
- `NODE_DISPLAY_NAME_MAPPINGS` — `{"NodeName": "Display Name"}`
- `WEB_DIRECTORY` — relative path to JS extensions (served at `/extensions/NodeName/`)

JS extensions: `import { app } from "../../../scripts/app.js"` (3 directory levels to root).

### Video Production Pipeline (`generate_video_workflow_v2.py`)

Python script generating ComfyUI workflow JSON (141 nodes, 166 links). Uses compact helpers: `N()` (node), `C()` (connection), `nid()`/`lid()` (ID generators).

| Step | Purpose |
|------|---------|
| 0 | Load shared models → 8 SetNodes (LLM_CONFIG, MODEL, CLIP, VAE, IPADAPTER, CLIP_VISION, WANMODEL, WANVAE) |
| 1 | LLM script generation from concept |
| 2 | Storyboard + START/END/Motion prompts + DirectorsCinemaPrompt |
| 3 | Character & style reference images via dual IPAdapter |
| 4+5 | Merged loop: image gen + WanVideo I2V per scene (`easy whileLoopStart/End`) |
| 6 | Voice (ElevenLabs/ChatterBox) + SRT subtitles |
| 7 | Dynamic assembly (NUM_SCENES configurable) |

Key patterns: SetNode/GetNode for resource sharing, While Loop for scene iteration, dual IPAdapter (identity + style).

### DirectorsConsole (`custom_nodes/ComfyUI-DirectorsConsole/`)

Cinema prompt engineering integrated as a ComfyUI custom node with reverse proxy architecture:

- **CPE Backend** (FastAPI, port 9800) → proxied at `/directors/cpe/`
- **Orchestrator** (FastAPI, port 9820) → proxied at `/directors/orch/` + WebSocket at `/directors/orch/ws/`
- **React Frontend** → static files at `/directors/app/`
- **Process Manager** (`proxy/process_manager.py`) — auto-starts/restarts FastAPI services as subprocesses
- **Reverse Proxy** (`proxy/reverse_proxy.py`) — registers aiohttp routes on `PromptServer.instance.routes`

All traffic flows through ComfyUI's single port 8188.

## Code Style

- **Python**: Ruff with Pyflakes + print detection. Line length not enforced (`E501` ignored). Python 3.10+.
- **Workflow generator**: Compact single-letter functions. Don't refactor to verbose style.
- **When modifying workflow**: always regenerate the JSON output file.
