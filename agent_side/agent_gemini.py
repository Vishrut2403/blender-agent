import json
import time
from typing import Any
from google import genai
from google.genai import types
from agent_side.tools import (
	get_scene_state,
	add_object,
	set_material,
	set_location,
	render_scene,
)

client = genai.Client()

PLANNER_MODEL  = "gemini-2.0-flash-lite"
EXECUTOR_MODEL = "gemini-2.0-flash-lite"

TOOLS = types.Tool(function_declarations=[
	types.FunctionDeclaration(
		name="get_scene_state",
		description="Returns all objects in the Blender scene with name, type, location, material.",
		parameters=types.Schema(type=types.Type.OBJECT, properties={})
	),
	types.FunctionDeclaration(
		name="add_object",
		description="Add a mesh primitive to the scene.",
		parameters=types.Schema(
			type=types.Type.OBJECT,
			properties={
				"kind": types.Schema(
					type=types.Type.STRING,
					enum=["CUBE", "SPHERE", "CYLINDER", "CONE", "PLANE", "MONKEY"]
				),
				"location": types.Schema(
					type=types.Type.ARRAY,
					items=types.Schema(type=types.Type.NUMBER),
					description="[x, y, z]"
				),
			},
			required=["kind", "location"]
		)
	),
	types.FunctionDeclaration(
		name="set_material",
		description="Apply an RGBA color material to a named object.",
		parameters=types.Schema(
			type=types.Type.OBJECT,
			properties={
				"object_name": types.Schema(type=types.Type.STRING),
				"color": types.Schema(
					type=types.Type.ARRAY,
					items=types.Schema(type=types.Type.NUMBER),
					description="[r, g, b, a] each 0-1"
				),
			},
			required=["object_name", "color"]
		)
	),
	types.FunctionDeclaration(
		name="set_location",
		description="Move a named object to x, y, z coordinates.",
		parameters=types.Schema(
			type=types.Type.OBJECT,
			properties={
				"object_name": types.Schema(type=types.Type.STRING),
				"x": types.Schema(type=types.Type.NUMBER),
				"y": types.Schema(type=types.Type.NUMBER),
				"z": types.Schema(type=types.Type.NUMBER),
			},
			required=["object_name", "x", "y", "z"]
		)
	),
	types.FunctionDeclaration(
		name="render_scene",
		description="Render the scene and save as PNG.",
		parameters=types.Schema(
			type=types.Type.OBJECT,
			properties={
				"output_path": types.Schema(type=types.Type.STRING)
			}
		)
	),
])

PLANNER_SYSTEM = """You are a Blender 3D planning assistant.
Given a user instruction and the current scene state, write a clear numbered
step-by-step plan for what tools to call and in what order.
Be specific about object names, colors (RGBA 0-1), and locations (x,y,z).
Do not call any tools yourself — just write the plan as text.
Be concise — maximum 5 steps."""

EXECUTOR_SYSTEM = """You are a Blender 3D executor. You receive a plan and current scene state.
You MUST execute ALL steps in the plan using the available tools.
Do not stop until every single step is done.
Use exact object names from the scene. Colors are RGBA floats 0-1.
Do not explain — just call the tools."""

CRITIC_SYSTEM = """You are a Blender 3D critic. Write a 2 sentence summary of what was done. Be direct."""


def dispatch_tool(name: str, args: dict[str, Any]) -> str:
	match name:
		case "get_scene_state":
			return str(get_scene_state())
		case "add_object":
			return str(add_object(kind=args["kind"], location=tuple(args["location"])))
		case "set_material":
			return str(set_material(object_name=args["object_name"], color=tuple(args["color"])))
		case "set_location":
			return str(set_location(object_name=args["object_name"], x=args["x"], y=args["y"], z=args["z"]))
		case "render_scene":
			return str(render_scene(output_path=args.get("output_path", "/tmp/blender_render.png")))
		case _:
			return f"Unknown tool: {name}"


def run_planner(user_instruction: str, scene_state: str) -> str:
	print(f"\n[Planner/{PLANNER_MODEL}] Thinking...")
	response = client.models.generate_content(
		model=PLANNER_MODEL,
		config=types.GenerateContentConfig(system_instruction=PLANNER_SYSTEM),
		contents=f"Instruction: {user_instruction}\n\nCurrent scene:\n{scene_state}"
	)
	plan = response.text or ""
	print(f"[Planner] Plan:\n{plan}")
	return plan


def run_executor(plan: str, scene_state: str) -> list[dict[str, Any]]:
	print(f"\n[Executor/{EXECUTOR_MODEL}] Executing plan...")
	config = types.GenerateContentConfig(
		system_instruction=EXECUTOR_SYSTEM,
		tools=[TOOLS],
	)
	history: list[types.Content] = [
		types.Content(
			role="user",
			parts=[types.Part(text=f"Execute ALL steps of this plan:\n{plan}\n\nCurrent scene:\n{scene_state}")]
		)
	]

	actions_taken: list[dict[str, Any]] = []

	while True:
		response = client.models.generate_content(
			model=EXECUTOR_MODEL,
			config=config,
			contents=history,
		)

		candidate = response.candidates[0]
		history.append(candidate.content)

		text_parts = [p.text for p in candidate.content.parts if hasattr(p, "text") and p.text]
		tool_parts = [p for p in candidate.content.parts if p.function_call]

		if text_parts:
			print(f"[Executor] {' '.join(text_parts)}")

		if not tool_parts:
			break

		results = []
		for part in tool_parts:
			fc = part.function_call
			args: dict[str, Any] = dict(fc.args)

			print(f"[Tool]   {fc.name}({json.dumps(args)})")
			result = dispatch_tool(fc.name, args)
			preview = result[:150] + "..." if len(result) > 150 else result
			print(f"[Result] {preview}")

			actions_taken.append({"tool": fc.name, "args": args, "result": result})

			results.append(types.Part(
				function_response=types.FunctionResponse(
					name=fc.name,
					response={"result": result}
				)
			))

		history.append(types.Content(role="user", parts=results))

	return actions_taken


def run_critic(user_instruction: str, plan: str, actions: list[dict[str, Any]]) -> str:
	print(f"\n[Critic/{PLANNER_MODEL}] Reviewing...")
	actions_text = "\n".join(
		f"- {a['tool']}({json.dumps(a['args'])}) → {str(a['result'])[:80]}"
		for a in actions
	)
	response = client.models.generate_content(
		model=PLANNER_MODEL,
		config=types.GenerateContentConfig(system_instruction=CRITIC_SYSTEM),
		contents=f"Instruction: {user_instruction}\nActions taken:\n{actions_text}\nWrite a 2 sentence summary."
	)
	return response.text or "Done."


def run_agent(user_instruction: str) -> str:
	print(f"\n{'='*50}")
	print(f"Task: {user_instruction}")
	print(f"{'='*50}")

	scene_state = str(get_scene_state())
	print(f"\n[Scene] {scene_state[:200]}")

	plan = run_planner(user_instruction, scene_state)
	time.sleep(5)

	if not plan.strip():
		print("[Planner] Empty plan, retrying...")
		plan = run_planner(user_instruction, scene_state)
		time.sleep(5)

	if not plan.strip():
		return "Could not generate a plan. Try rephrasing."

	actions = run_executor(plan, scene_state)
	time.sleep(5)
	summary = run_critic(user_instruction, plan, actions)

	print(f"\n[Done] {summary}")
	return summary