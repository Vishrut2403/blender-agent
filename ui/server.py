from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from agent_side.agent import SessionMemory, run_agent
from agent_side.config import (
	CHAT_HISTORY_DIR,
	RENDERS_DIR,
	UI_HOST,
	UI_PORT,
	DEFAULT_RENDER_PATH,
)

logger = logging.getLogger(__name__)

# Directory bootstrap 

def _ensure_dirs() -> None:
	"""Create required directories if they don't exist yet."""
	Path(CHAT_HISTORY_DIR).mkdir(parents=True, exist_ok=True)
	Path(RENDERS_DIR).mkdir(parents=True, exist_ok=True)

_ensure_dirs()

# App & static mounts 

app = FastAPI(title="Blender Agent UI", version="0.6.0")

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

#  Global state 

_global_memory: SessionMemory = SessionMemory()

_session_queues: dict[str, queue.Queue] = {}

_executor = ThreadPoolExecutor(max_workers=1)

# Render copy helper 

_render_counter: dict[str, int] = {}


def _copy_render(session_id: str, src_path: str) -> str | None:
	"""
	Copy a render file from its Blender output path to the served renders dir.

	Returns the relative URL path (/static/renders/...) or None on failure.
	"""
	src = Path(src_path)
	if not src.exists():
		logger.warning("Render source not found: %s", src_path)
		return None

	n = _render_counter.get(session_id, 0) + 1
	_render_counter[session_id] = n

	dest_name = f"{session_id}_{n:03d}.png"
	dest = Path(RENDERS_DIR) / dest_name

	try:
		shutil.copy2(src, dest)
		logger.info("Render copied: %s → %s", src, dest)
		return f"/static/renders/{dest_name}"
	except OSError:
		logger.exception("Failed to copy render from %s", src_path)
		return None

# Session persistence 

def _save_session(session_id: str, memory: SessionMemory) -> None:
	"""Serialize SessionMemory to ui/chat_history/{session_id}.json."""
	path = Path(CHAT_HISTORY_DIR) / f"{session_id}.json"
	payload = {
		"session_id": session_id,
		"saved_at": datetime.utcnow().isoformat(),
		**memory.to_dict(),
	}
	try:
		path.write_text(json.dumps(payload, indent=2))
		logger.info("Session saved: %s", path)
	except OSError:
		logger.exception("Failed to save session %s", session_id)

# Background agent runner 

_SENTINEL = None  # Signals the WS handler that the agent turn is finished.


def _run_agent_thread(
	session_id: str,
	instruction: str,
	q: queue.Queue,
) -> None:
	"""
	Entry point executed in the ThreadPoolExecutor worker thread.

	Calls run_agent() with a broadcast_fn that puts events onto the queue.
	Handles render copy-and-rewrite so the browser receives a servable URL.
	Saves memory and sends the SENTINEL when done.
	"""
	global _global_memory

	def broadcast_fn(event: dict[str, Any]) -> None:
		"""Put an agent event on the queue.  Runs in the worker thread."""
		# Intercept render events: copy the file and rewrite the URL.
		if event.get("type") == "render":
			raw_path = event.get("content", DEFAULT_RENDER_PATH)
			url = _copy_render(session_id, raw_path)
			if url:
				event = {"type": "render", "content": url}
		q.put(event)

	try:
		_global_memory = run_agent(instruction, _global_memory, broadcast_fn)
		_save_session(session_id, _global_memory)
	except Exception:
		logger.exception("Agent thread raised an exception for session %s", session_id)
		q.put({"type": "error", "content": "Agent encountered an unexpected error."})
	finally:
		q.put(_SENTINEL)  # Always send sentinel so WS handler can close cleanly.

# HTTP Endpoints 

@app.get("/", include_in_schema=False)
async def serve_index() -> FileResponse:
	"""Serve the single-page UI."""
	return FileResponse(str(_STATIC_DIR / "index.html"))


@app.post("/chat")
async def post_chat(body: dict[str, str]) -> JSONResponse:
	"""
	Accept an instruction, start the agent in a background thread, and return
	a session_id the browser uses to open the matching WebSocket.

	Why return immediately instead of streaming here?
	Server-Sent Events or streaming responses in FastAPI work, but a WebSocket
	gives us bidirectional control and cleaner reconnect logic on the client.
	"""
	instruction = (body.get("instruction") or "").strip()
	if not instruction:
		return JSONResponse({"error": "instruction must not be empty"}, status_code=422)

	session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
	q: queue.Queue = queue.Queue()
	_session_queues[session_id] = q

	# Submit to thread pool — returns immediately; agent runs in background.
	loop = asyncio.get_event_loop()
	loop.run_in_executor(_executor, _run_agent_thread, session_id, instruction, q)

	logger.info("Started session %s for: %s", session_id, instruction)
	return JSONResponse({"session_id": session_id})


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
	"""
	Stream agent events to the browser for the given session.

	Pattern: poll a thread-safe Queue in a tight async loop.
	- queue.get_nowait() raises queue.Empty when nothing is ready.
	- await asyncio.sleep(0) yields control back to the event loop between polls,
	  keeping the server responsive to other connections.
	- SENTINEL (None) means the agent is done; we close the WebSocket.
	"""
	await websocket.accept()
	q = _session_queues.get(session_id)

	if q is None:
		await websocket.send_json({"type": "error", "content": "Unknown session."})
		await websocket.close()
		return

	try:
		while True:
			try:
				event = q.get_nowait()
			except queue.Empty:
				# Nothing ready yet — yield to the event loop and retry.
				await asyncio.sleep(0.05)
				continue

			if event is _SENTINEL:
				# Agent finished.  Give the client a moment to process the last
				# event before we close.
				await asyncio.sleep(0.1)
				break

			await websocket.send_json(event)

	except WebSocketDisconnect:
		logger.info("WebSocket disconnected for session %s", session_id)
	finally:
		# Clean up queue reference so we don't leak memory between sessions.
		_session_queues.pop(session_id, None)
		try:
			await websocket.close()
		except Exception:
			pass


@app.get("/sessions")
async def list_sessions() -> JSONResponse:
	"""Return a list of saved session metadata (id + timestamp + turn count)."""
	history_dir = Path(CHAT_HISTORY_DIR)
	sessions = []
	for path in sorted(history_dir.glob("*.json"), reverse=True):
		try:
			data = json.loads(path.read_text())
			sessions.append({
				"id": data.get("session_id", path.stem),
				"saved_at": data.get("saved_at", ""),
				"turns": len(data.get("action_log", [])),
				"first_instruction": (
					data["action_log"][0]["instruction"]
					if data.get("action_log") else ""
				),
			})
		except (OSError, json.JSONDecodeError):
			logger.warning("Could not parse session file: %s", path)
	return JSONResponse(sessions)


@app.get("/sessions/{session_id}")
async def get_session(session_id: str) -> JSONResponse:
	"""Return the full action log for a specific session."""
	path = Path(CHAT_HISTORY_DIR) / f"{session_id}.json"
	if not path.exists():
		return JSONResponse({"error": "Session not found."}, status_code=404)
	try:
		return JSONResponse(json.loads(path.read_text()))
	except (OSError, json.JSONDecodeError):
		logger.exception("Could not read session %s", session_id)
		return JSONResponse({"error": "Failed to read session."}, status_code=500)

# Server entry point 

def start_server(host: str = UI_HOST, port: int = UI_PORT) -> None:
	"""Start the uvicorn server.  Called from main.py when --ui is passed."""
	uvicorn.run(
		app,
		host=host,
		port=port,
		log_level="warning",  # Suppress uvicorn's verbose access logs.
	)