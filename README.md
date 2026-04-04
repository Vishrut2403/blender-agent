# Blender Agent

An agentic AI system that controls Blender via natural language.

## Structure
- `blender_side/` — scripts that run inside Blender Python
- `agent_side/`   — external Python agent that talks to Blender

## Setup
1. Open Blender → Scripting workspace → Run `blender_side/server.py`
2. In a terminal: `python test_bridge.py`
