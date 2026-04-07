import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from openai import OpenAI

from agent_side.bridge import BlenderResponse
from agent_side.config import (
	OLLAMA_BASE_URL,
	OLLAMA_API_KEY,
	LOCAL_MODEL,
	MAX_TOOL_CALLS,
	MAX_PLAN_RETRIES,
	ACTION_LOG_WINDOW,
)
from agent_side.tools import (
	get_scene_state,
	add_object,
	set_material,
	set_location,
	render_scene,
)

logger = logging.getLogger(__name__)

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)

# Type alias for the broadcast callback
BroadcastFn = Callable[[dict[str, Any]], None]


# ─── Session Memory ────────────────────────────────────────────────────────────

@dataclass
class SessionMemory:
	conversation_history: list[dict[str, str]] = field(default_factory=list)
	action_log: list[dict[str, Any]] = field(default_factory=list)
	scene_cache: dict | None = None

	def add_turn(
		self,
		instruction: str,
		tools_called: list[str],
		objects_affected: list[str],
		summary: str,
	) -> None:
		turn_num = len(self.action_log) + 1
		self.action_log.append({
			"turn": turn_num,
			"instruction": instruction,
			"tools_called": tools_called,
			"objects_affected": objects_affected,
			"summary": summary,
		})
		self.conversation_history.append({"role": "user", "content": instruction})
		self.conversation_history.append({"role": "assistant", "content": summary})

	def get_action_summary(self, last_n: int = ACTION_LOG_WINDOW) -> str:
		if not self.action_log:
			return "No previous actions."
		lines = []
		for entry in self.action_log[-last_n:]:
			lines.append(
				f"Turn {entry['turn']}: \"{entry['instruction']}\" → "
				f"{entry['summary']} (tools: {', '.join(entry['tools_called'])})"
			)
		return "\n".join(lines)

	def get_scene_summary(self) -> str:
		if not self.scene_cache:
			return "Scene state unknown — will be fetched."
		objects = self.scene_cache.get("objects", [])
		if not objects:
			return "Scene is empty."
		lines = []
		for obj in objects:
			name = obj.get("name", "?")
			kind = obj.get("type", "?")
			loc  = obj.get("location", [0, 0, 0])
			mat  = obj.get("material", "none")
			lines.append(f"  - {name} ({kind}) at {loc}, material: {mat}")
		return "\n".join(lines)

	def to_dict(self) -> dict[str, Any]:
		"""Serialize memory to a plain dict for JSON persistence."""
		return {
			"conversation_history": self.conversation_history,
			"action_log": self.action_log,
			"scene_cache": self.scene_cache,
		}

	@classmethod
	def from_dict(cls, data: dict[str, Any]) -> "SessionMemory":
		"""Restore memory from a persisted dict."""
		m = cls()
		m.conversation_history = data.get("conversation_history", [])
		m.action_log           = data.get("action_log", [])
		m.scene_cache          = data.get("scene_cache", None)
		return m


# ─── Tool Schema ───────────────────────────────────────────────────────────────

TOOLS: list[dict[str, Any]] = [
	{
		"type": "function",
		"function": {
			"name": "get_scene_state",
			"description": "Get all objects currently in the Blender scene.",
			"parameters": {"type": "object", "properties": {}, "required": []},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "add_object",
			"description": "Add a mesh primitive to the scene.",
			"parameters": {
				"type": "object",
				"properties": {
					"kind": {
						"type": "string",
						"enum": ["SPHERE", "CUBE", "CYLINDER", "CONE", "PLANE"],
					},
					"location": {
						"type": "array",
						"items": {"type": "number"},
						"description": "[x, y, z]",
					},
				},
				"required": ["kind"],
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "set_material",
			"description": "Set the material color of an object.",
			"parameters": {
				"type": "object",
				"properties": {
					"object_name": {"type": "string"},
					"color": {
						"type": "array",
						"items": {"type": "number"},
						"description": "[R, G, B, A] in 0-1 range",
					},
				},
				"required": ["object_name", "color"],
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "set_location",
			"description": "Move an object to a specific location.",
			"parameters": {
				"type": "object",
				"properties": {
					"object_name": {"type": "string"},
					"x": {"type": "number"},
					"y": {"type": "number"},
					"z": {"type": "number"},
				},
				"required": ["object_name", "x", "y", "z"],
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "render_scene",
			"description": "Render the current scene to a file.",
			"parameters": {
				"type": "object",
				"properties": {
					"output_path": {"type": "string"},
				},
				"required": ["output_path"],
			},
		},
	},
]


# ─── Tool Dispatch ─────────────────────────────────────────────────────────────

def _serialize(result: BlenderResponse | dict | Any) -> str:
	"""Convert any tool result to a JSON string safe for the messages list."""
	if isinstance(result, BlenderResponse):
		return json.dumps({
			"ok": result.ok,
			"stdout": result.stdout,
			"error": result.error,
			"value": result.value,
		})
	if isinstance(result, dict):
		return json.dumps(result)
	return str(result)


def _parse_scene(result: BlenderResponse | str | Any) -> dict | None:
	"""Parse a BlenderResponse or JSON string into a scene dict."""
	try:
		if isinstance(result, BlenderResponse):
			raw = result.stdout or result.value
		elif isinstance(result, str):
			raw = result
		else:
			return None
		parsed = json.loads(raw)
		if isinstance(parsed, list):
			return {"objects": parsed}
		return parsed
	except (json.JSONDecodeError, TypeError):
		return None


def dispatch_tool(name: str, args: dict[str, Any]) -> str:
	"""Route a tool call by name and return a serialized JSON string result."""
	logger.debug("Dispatching tool: %s | args: %s", name, args)
	if name == "get_scene_state":
		result = get_scene_state()
	elif name == "add_object":
		result = add_object(**args)
	elif name == "set_material":
		result = set_material(**args)
	elif name == "set_location":
		result = set_location(**args)
	elif name == "render_scene":
		result = render_scene(**args)
	else:
		logger.warning("Unknown tool requested: %s", name)
		return json.dumps({"error": f"Unknown tool: {name}"})
	return _serialize(result)


# ─── Emit helper ──────────────────────────────────────────────────────────────

def _emit(event_type: str, content: str, broadcast_fn: BroadcastFn | None) -> None:
	"""Print to terminal and optionally broadcast to WebSocket."""
	print(f"[{event_type}] {content}")
	if broadcast_fn:
		broadcast_fn({"type": event_type, "content": content})


# ─── Agents ───────────────────────────────────────────────────────────────────

def run_planner(
	instruction: str,
	memory: SessionMemory,
	broadcast_fn: BroadcastFn | None = None,
) -> str:
	"""Generate a step-by-step plan for the given instruction using session context."""
	system_prompt = f"""You are a Blender scene planner. Your job is to produce a clear,
step-by-step plan to fulfill the user's instruction using available tools.

CURRENT SCENE:
{memory.get_scene_summary()}

PREVIOUS ACTIONS THIS SESSION:
{memory.get_action_summary()}

Available tools: get_scene_state, add_object, set_material, set_location, render_scene

Rules:
- Output ONLY a numbered list of steps. No explanations. No code.
- NEVER invent object names. Object names are assigned by Blender after add_object runs.
- Always call add_object before set_material or set_location on a new object.
- If the instruction references a previous object (e.g. "it", "that"), resolve it using
  the scene state and action history above.
- For set_material, always specify color as [R, G, B, A] with values between 0 and 1."""

	logger.debug("Running planner for: %s", instruction)
	_emit("planner", "Thinking...", broadcast_fn)
	try:
		response = client.chat.completions.create(
			model=LOCAL_MODEL,
			messages=[
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": instruction},
			],
			extra_body={"think": False},
		)
		plan = response.choices[0].message.content.strip()
		_emit("planner", plan, broadcast_fn)
		return plan
	except Exception:
		logger.exception("Planner LLM call failed")
		return ""


def run_executor(
	plan: str,
	memory: SessionMemory,
	broadcast_fn: BroadcastFn | None = None,
) -> tuple[list[str], list[str], dict | None]:
	"""Execute a plan by calling tools. Returns (tools_called, objects_affected, updated_cache)."""
	system_prompt = f"""You are a Blender executor. You receive a plan and call tools to carry it out.

CURRENT SCENE:
{memory.get_scene_summary()}

Execute the plan step by step using the available tools.
After completing all steps, do NOT call any more tools."""

	messages: list[dict[str, Any]] = [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": f"Execute this plan:\n{plan}"},
	]

	tools_called: list[str] = []
	objects_affected: list[str] = []
	updated_cache = memory.scene_cache

	_emit("executor", "Running...", broadcast_fn)

	for _ in range(MAX_TOOL_CALLS):
		try:
			response = client.chat.completions.create(
				model=LOCAL_MODEL,
				messages=messages,
				tools=TOOLS,
				tool_choice="auto",
				extra_body={"think": False},
			)
		except Exception:
			logger.exception("Executor LLM call failed")
			break

		msg = response.choices[0].message
		if not msg.tool_calls:
			break

		messages.append(msg)

		for tc in msg.tool_calls:
			name = tc.function.name
			args = json.loads(tc.function.arguments)

			_emit("tool", f"Calling {name}({args})", broadcast_fn)
			result = dispatch_tool(name, args)

			# try to surface a clean result message
			try:
				parsed_result = json.loads(result)
				tool_msg = parsed_result.get("stdout") or parsed_result.get("error") or "done"
			except (json.JSONDecodeError, AttributeError):
				tool_msg = result
			_emit("tool_result", f"{name} → {tool_msg}", broadcast_fn)

			tools_called.append(name)

			if name == "add_object":
				try:
					stdout = json.loads(result).get("stdout", "")
					obj_name = stdout.split("Added ")[-1].split(" at")[0]
					if obj_name:
						objects_affected.append(obj_name)
				except (json.JSONDecodeError, AttributeError):
					logger.debug("Could not parse object name from add_object result")

			if name in ("add_object", "set_material", "set_location"):
				parsed = _parse_scene(get_scene_state())
				if parsed:
					updated_cache = parsed

			if name == "get_scene_state":
				parsed = _parse_scene(result)
				if parsed:
					updated_cache = parsed

			# notify UI when a render completes so it can show the image
			if name == "render_scene":
				try:
					output_path = args.get("output_path", "")
					_emit("render", output_path, broadcast_fn)
				except Exception:
					pass

			messages.append({
				"role": "tool",
				"tool_call_id": tc.id,
				"content": result,
			})

	_emit("executor", f"Done. Tools called: {tools_called}", broadcast_fn)
	return tools_called, objects_affected, updated_cache


def run_critic(
	instruction: str,
	tools_called: list[str],
	memory: SessionMemory,
	broadcast_fn: BroadcastFn | None = None,
) -> str:
	"""Review whether the instruction was fulfilled. Returns PASS/FAIL with one sentence."""
	system_prompt = (
		"You are a Blender critic. Review whether the instruction was fulfilled. "
		"Be concise. Say PASS or FAIL and one sentence why."
	)
	content = (
		f"Instruction: {instruction}\n"
		f"Tools called: {', '.join(tools_called) if tools_called else 'none'}\n"
		f"Current scene:\n{memory.get_scene_summary()}"
	)
	_emit("critic", "Reviewing...", broadcast_fn)
	try:
		response = client.chat.completions.create(
			model=LOCAL_MODEL,
			max_tokens=50,
			messages=[
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": content},
			],
		)
		verdict = response.choices[0].message.content.strip()
		_emit("critic", verdict, broadcast_fn)
		return verdict
	except Exception:
		logger.exception("Critic LLM call failed")
		return "SKIP"


# ─── Main Entry Point ──────────────────────────────────────────────────────────

def run_agent(
	instruction: str,
	memory: SessionMemory | None = None,
	broadcast_fn: BroadcastFn | None = None,
) -> SessionMemory:
	"""
	Run one turn of the agent pipeline (Planner → Executor → Critic).
	Pass memory from the previous turn to maintain session context.
	Optionally pass broadcast_fn to stream events to a WebSocket.
	Returns the updated SessionMemory.
	"""
	if memory is None:
		memory = SessionMemory()

	logger.info("New instruction: %s", instruction)
	_emit("instruction", instruction, broadcast_fn)

	# ─ Planner ─
	plan = ""
	for attempt in range(MAX_PLAN_RETRIES):
		plan = run_planner(instruction, memory, broadcast_fn)
		if plan.strip():
			break
		logger.warning("Planner returned empty plan (attempt %d/%d)", attempt + 1, MAX_PLAN_RETRIES)
		_emit("planner", f"Empty plan, retrying ({attempt+1}/{MAX_PLAN_RETRIES})...", broadcast_fn)

	if not plan.strip():
		logger.error("Planner failed after %d attempts", MAX_PLAN_RETRIES)
		_emit("error", "Planner failed to produce a plan. Skipping turn.", broadcast_fn)
		return memory

	# ─ Executor ─
	tools_called, objects_affected, updated_cache = run_executor(plan, memory, broadcast_fn)
	memory.scene_cache = updated_cache
	logger.info("Executor finished. Tools called: %s", tools_called)

	# ─ Critic (disabled for speed, re-enable when needed) ─
	# verdict = run_critic(instruction, tools_called, memory, broadcast_fn)
	verdict = "SKIP"

	# ─ Update Memory ─
	summary = (
		f"Called {', '.join(tools_called) or 'no tools'}. "
		f"Objects: {', '.join(objects_affected) or 'none modified'}. "
		f"Critic: {verdict}"
	)
	memory.add_turn(
		instruction=instruction,
		tools_called=tools_called,
		objects_affected=objects_affected,
		summary=summary,
	)

	_emit("done", "", broadcast_fn)
	return memory