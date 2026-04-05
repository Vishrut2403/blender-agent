from openai import OpenAI
from dataclasses import dataclass, field
from agent_side.tools import (
	BlenderResponse,
	get_scene_state,
	add_object,
	set_material,
	set_location,
	render_scene,
)
import json

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen3.5:4b"

# ─── Session Memory ────────────────────────────────────────────────────────────

@dataclass
class SessionMemory:
	conversation_history: list[dict] = field(default_factory=list)
	action_log: list[dict] = field(default_factory=list)
	scene_cache: dict | None = None

	def add_turn(self, instruction: str, tools_called: list[str],
				 objects_affected: list[str], summary: str):
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

	def get_action_summary(self, last_n: int = 5) -> str:
		if not self.action_log:
			return "No previous actions."
		recent = self.action_log[-last_n:]
		lines = []
		for entry in recent:
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


# ─── Tool Schema ───────────────────────────────────────────────────────────────

TOOLS = [
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
					"output_path": {"type": "string"}
				},
				"required": ["output_path"],
			},
		},
	},
]


# ─── Tool Dispatch ─────────────────────────────────────────────────────────────

def _serialize(result) -> str:
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


def dispatch_tool(name: str, args: dict) -> str:
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
		return json.dumps({"error": f"Unknown tool: {name}"})

	return _serialize(result)

def _parse_scene(result) -> dict | None:
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


# ─── Agents ───────────────────────────────────────────────────────────────────

def run_planner(instruction: str, memory: SessionMemory) -> str:
	system_prompt = f"""You are a Blender scene planner. Your job is to produce a clear, 
step-by-step plan to fulfill the user's instruction using available tools.

CURRENT SCENE:
{memory.get_scene_summary()}

PREVIOUS ACTIONS THIS SESSION:
{memory.get_action_summary()}

Available tools: get_scene_state, add_object, set_material, set_location, render_scene

Output ONLY a numbered list of steps. No explanations. No code. Just the plan.
If the instruction references a previous object (e.g. "it", "that"), resolve it using 
the scene state and action history above."""

	response = client.chat.completions.create(
		model=MODEL,
		messages=[
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": instruction},
		],
		extra_body={"think": False},
	)
	return response.choices[0].message.content.strip()


def run_executor(plan: str, memory: SessionMemory) -> tuple[list[str], list[str], dict]:
	"""Returns (tools_called, objects_affected, updated_scene_cache)."""
	system_prompt = f"""You are a Blender executor. You receive a plan and call tools to carry it out.

CURRENT SCENE:
{memory.get_scene_summary()}

Execute the plan step by step using the available tools. 
After completing all steps, do NOT call any more tools."""

	messages = [
		{"role": "system", "content": system_prompt},
		{"role": "user", "content": f"Execute this plan:\n{plan}"},
	]

	tools_called = []
	objects_affected = []
	updated_cache = memory.scene_cache

	for _ in range(10):  # max 10 tool calls per turn
		response = client.chat.completions.create(
			model=MODEL,
			messages=messages,
			tools=TOOLS,
			tool_choice="auto",
			extra_body={"think": False},
		)
		msg = response.choices[0].message

		if not msg.tool_calls:
			break

		messages.append(msg)

		for tc in msg.tool_calls:
			name = tc.function.name
			args = json.loads(tc.function.arguments)
			result = dispatch_tool(name, args)  # always a JSON string now

			tools_called.append(name)

			# refresh cache when scene changes
			if name in ("add_object", "set_material", "set_location"):
				# track added object name from result string
				if name == "add_object":
					try:
						parsed_result = json.loads(result)
						obj_name = parsed_result.get("stdout", "").split("Added ")[-1].split(" at")[0]
						if obj_name:
							objects_affected.append(obj_name)
					except (json.JSONDecodeError, AttributeError):
						pass

				scene = get_scene_state()
				parsed = _parse_scene(scene)
				if parsed:
					updated_cache = parsed

			# if scene was explicitly queried, cache it
			if name == "get_scene_state":
				parsed = _parse_scene(result)
				if parsed:
					updated_cache = parsed

			messages.append({
				"role": "tool",
				"tool_call_id": tc.id,
				"content": result,
			})

	return tools_called, objects_affected, updated_cache


def run_critic(instruction: str, tools_called: list[str], memory: SessionMemory) -> str:
	system_prompt = """You are a Blender critic. Review whether the instruction was fulfilled.
Be concise. Say PASS or FAIL and one sentence why."""

	content = f"""Instruction: {instruction}
Tools called: {', '.join(tools_called) if tools_called else 'none'}
Current scene:
{memory.get_scene_summary()}"""

	response = client.chat.completions.create(
		model=MODEL,
		messages=[
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": content},
		],
	)
	return response.choices[0].message.content.strip()


# ─── Main Entry Point ──────────────────────────────────────────────────────────

def run_agent(instruction: str, memory: SessionMemory | None = None) -> SessionMemory:
	"""
	Run one turn of the agent. Pass in memory from the previous turn to maintain
	session context. Returns the updated memory object.
	"""
	if memory is None:
		memory = SessionMemory()

	print(f"\n{'='*50}")
	print(f"INSTRUCTION: {instruction}")
	print(f"{'='*50}")

	# — Planner —
	print("\n[Planner] Thinking...")
	plan = ""
	for attempt in range(3):
		plan = run_planner(instruction, memory)
		if plan.strip():
			break
		print(f"[Planner] Empty plan, retrying ({attempt+1}/3)...")

	if not plan.strip():
		print("[Planner] Failed to produce a plan. Skipping turn.")
		return memory

	print(f"[Planner] Plan:\n{plan}")

	# — Executor —
	print("\n[Executor] Running...")
	tools_called, objects_affected, updated_cache = run_executor(plan, memory)
	memory.scene_cache = updated_cache
	print(f"[Executor] Tools called: {tools_called}")

	# — Critic —
	print("\n[Critic] Reviewing...")
	#verdict = run_critic(instruction, tools_called, memory)
	#print(f"[Critic] {verdict}")
	verdict = "SKIP"

	# — Update Memory —
	summary = f"Called {', '.join(tools_called)}. Objects: {', '.join(objects_affected) or 'none modified'}. Critic: {verdict}"
	memory.add_turn(
		instruction=instruction,
		tools_called=tools_called,
		objects_affected=objects_affected,
		summary=summary,
	)

	return memory