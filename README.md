# blender-agent

An AI-powered Blender agent controlled via natural language. Speak to it in plain English — it plans, executes, and verifies changes to your Blender scene automatically.

```
You: add a violet sphere at the origin
You: now make it blue
You: move it up by 3
You: render
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Natural Language Input                 │
└─────────────────────────┬───────────────────────────────┘
                          │
              ┌───────────▼───────────┐
              │        Planner        │
              │  reads scene cache +  │
              │  action history →     │
              │  produces a plan      │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │       Executor        │
              │  ReAct loop — calls   │
              │  tools until plan     │
              │  is complete          │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │        Critic         │
              │  verifies outcome     │
              │  (disabled by default)│
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │     SessionMemory     │
              │  action log +         │
              │  scene cache          │
              └───────────┬───────────┘
                          │
              socket · localhost:6789
                          │
              ┌───────────▼───────────┐
              │     Blender Side      │
              │  server.py inside     │
              │  Blender — executes   │
              │  bpy on main thread   │
              └───────────────────────┘
```

Three-agent pipeline: **Planner → Executor → Critic**

| Agent | Role |
|---|---|
| **Planner** | Reads scene state and action history, produces a numbered step-by-step plan |
| **Executor** | Runs a ReAct loop, calls tools until the plan is fulfilled, updates scene cache |
| **Critic** | Verifies the outcome against the instruction (disabled by default for speed) |
| **SessionMemory** | Persists action log and scene state across turns so pronouns like "it" resolve correctly |

---

## Hardware & Environment

Developed and tested on:

- **CPU** — AMD Ryzen 7 5800H
- **GPU** — NVIDIA RTX 3060 Laptop 6GB VRAM
- **RAM** — 16GB
- **OS** — Arch Linux, KDE Plasma, Fish shell

---

## Prerequisites

- [Blender](https://www.blender.org/) — tested on the version shipping the `BLENDER_EEVEE` engine
- [Ollama](https://ollama.com/) with `qwen3.5:4b` pulled
- Python 3.11+
- A Google Gemini API key — only needed for `--gemini` mode

---

## Installation

```bash
git clone https://github.com/Vishrut2403/blender-agent
cd blender-agent
python -m venv .venv
source .venv/bin/activate        # fish: source .venv/bin/activate.fish
pip install openai google-genai
```

Pull the local model:

```bash
ollama pull qwen3.5:4b
```

For Gemini mode, set your API key:

```bash
# fish
set -Ux GEMINI_API_KEY "your-key-here"

# bash / zsh
export GEMINI_API_KEY="your-key-here"
```

---

## Configuration

All tuneable values live in `agent_side/config.py`. Nothing is hardcoded elsewhere.

| Constant | Default | Description |
|---|---|---|
| `BLENDER_HOST` | `localhost` | Blender bridge host |
| `BLENDER_PORT` | `6789` | Blender bridge port |
| `SOCKET_TIMEOUT` | `120` | Seconds to wait for Blender execution |
| `LOCAL_MODEL` | `qwen3.5:4b` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama API endpoint |
| `MAX_TOOL_CALLS` | `10` | Max tool calls per executor turn |
| `MAX_PLAN_RETRIES` | `3` | Max planner retries on empty plan |
| `ACTION_LOG_WINDOW` | `5` | Past turns visible to the planner |
| `DEFAULT_RENDER_PATH` | `/tmp/blender_render.png` | Default render output path |

---

## Startup Sequence

Order matters — Ollama must be loaded before Blender to maximise GPU layer offloading.

**Step 1 — Start Ollama and pre-load the model into VRAM:**

```fish
sudo kill -9 (sudo lsof -t -i :11434)   # kill any stale instance
ollama serve &
sleep 5
ollama run qwen3.5:4b "hello"            # pre-load into VRAM, then Ctrl+D
```

**Step 2 — Open Blender and start the bridge server:**

```fish
blender &
```

Inside Blender:
1. Switch to the **Scripting** workspace
2. Open `blender_side/server.py`
3. Click **Run Script**
4. Wait for the confirmation: `[BlenderBridge] Listening on localhost:6789`

**Step 3 — Run the agent:**

```fish
cd blender-agent
source .venv/bin/activate.fish

python main.py           # local mode (qwen3.5:4b)
python main.py --gemini  # cloud mode (Gemini)
```

---

## Usage

```
Blender Agent | Mode: qwen3.5:4b (local)
Type 'quit' to exit.

You: add a red cube
You: now make it blue
You: move it up by 3
You: render
```

Type `quit`, `exit`, or `q` to end the session.

---

## Available Tools

| Tool | Description |
|---|---|
| `get_scene_state` | Returns all objects in the scene — name, type, location, material |
| `add_object` | Adds a mesh primitive: `CUBE`, `SPHERE`, `CYLINDER`, `CONE`, `PLANE`, `MONKEY` |
| `set_material` | Applies a Principled BSDF material with an RGBA color |
| `set_location` | Moves an object to an absolute XYZ position |
| `render_scene` | Renders to PNG via EEVEE, auto-adds camera and light if missing |

---

## Persistent Session Memory

Every turn, the planner receives:

- **Scene cache** — current objects, their types, locations, and materials
- **Action log** — last N instructions and what tools were called, configurable via `ACTION_LOG_WINDOW`

This allows the agent to resolve pronouns and relative references across turns:

```
You: add a red sphere      ← Blender assigns the name "Sphere"
You: now make it blue      ← planner reads cache, resolves "it" → Sphere
You: move it up by 3       ← still resolves correctly without restating the name
```

---

## Logging

Log level is set to `WARNING` by default so the terminal stays clean. To see every tool dispatch and LLM call, change the level in `main.py`:

```python
logging.basicConfig(
    level=logging.DEBUG,          # WARNING → DEBUG
    format="%(levelname)s | %(name)s | %(message)s",
)
```

---

## Performance Notes

- `qwen3.5:4b` at Q4 quantization = 3.4GB VRAM footprint
- On RTX 3060 Laptop: 32/33 layers on GPU, output layer on CPU — ~20–40 sec per agent step
- KDE Plasma consumes ~1GB VRAM at idle — always pre-load Ollama before opening Blender
- Critic is disabled by default — re-enable in `run_agent()` inside `agent.py` when needed

---

## Known Limitations

- Executor occasionally calls tools redundantly — the 4B model being cautious, acceptable at this scale
- Planner sometimes returns an empty plan on complex instructions — retry logic handles it up to `MAX_PLAN_RETRIES`
- Gemini free tier quota exhausts quickly during heavy testing — resets daily at midnight Pacific
- Output layer stays on CPU due to VRAM constraints — 20–40 sec per step is expected

---

## License

MIT