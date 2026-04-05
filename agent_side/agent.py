import json
from typing import Any
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from agent_side.tools import (
	get_scene_state,
	add_object,
	set_material,
	set_location,
	render_scene,
)

client = OpenAI(
	base_url="http://localhost:11434/v1",
	api_key="ollama",
)

PLANNER_MODEL  = "qwen3.5:4b"
EXECUTOR_MODEL = "qwen3.5:4b"

TOOLS: list[dict[str, Any]] = [
	{
		"type": "function",
		"function": {
			"name": "get_scene_state",
			"description": "Returns all objects in the Blender scene with name, type, location, material.",
			"parameters": {"type": "object", "properties": {}, "required": []}
		}
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
						"enum": ["CUBE", "SPHERE", "CYLINDER", "CONE", "PLANE", "MONKEY"]
					},
					"location": {
						"type": "array",
						"items": {"type": "number"},
						"description": "[x, y, z]"
					}
				},
				"required": ["kind", "location"]
			}
		}
	},
	{
		"type": "function",
		"function": {
			"name": "set_material",
			"description": "Apply an RGBA color material to a named object.",
			"parameters": {
				"type": "object",
				"properties": {
					"object_name": {"type": "string"},
					"color": {
						"type": "array",
						"items": {"type": "number"},
						"description": "[r, g, b, a] each between 0 and 1"
					}
				},
				"required": ["object_name", "color"]
			}
		}
	},
	{
		"type": "function",
		"function": {
			"name": "set_location",
			"description": "Move a named object to x, y, z coordinates.",
			"parameters": {
				"type": "object",
				"properties": {
					"object_name": {"type": "string"},
					"x": {"type": "number"},
					"y": {"type": "number"},
					"z": {"type": "number"}
				},
				"required": ["object_name", "x", "y", "z"]
			}
		}
	},
	{
		"type": "function",
		"function": {
			"name": "render_scene",
			"description": "Render the scene and save as PNG.",
			"parameters": {
				"type": "object",
				"properties": {
					"output_path": {"type": "string"}
				},
				"required": []
			}
		}
	}
]


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
	messages: list[ChatCompletionMessageParam] = [
		{
			"role": "system",
			"content": """You are a Blender 3D planning assistant.
Given a user instruction and the current scene state, write a clear numbered
step-by-step plan for what tools to call and in what order.
Be specific about object names, colors (RGBA 0-1), and locations (x,y,z).
Do not call any tools yourself — just write the plan as text.
Be concise — maximum 5 steps."""
		},
		{
			"role": "user",
			"content": f"Instruction: {user_instruction}\n\nCurrent scene:\n{scene_state}"
		}
	]
	response = client.chat.completions.create(
		model=PLANNER_MODEL,
		messages=messages,
		extra_body={"think": False},
	)
	plan = response.choices[0].message.content or ""
	print(f"[Planner] Plan:\n{plan}")
	return plan


def run_executor(plan: str, scene_state: str) -> list[dict[str, Any]]:
	print(f"\n[Executor/{EXECUTOR_MODEL}] Executing plan...")
	messages: list[ChatCompletionMessageParam] = [
		{
			"role": "system",
			"content": """You are a Blender 3D executor. You receive a plan and current scene state.
You MUST execute ALL steps in the plan using the available tools.
Do not stop until every single step is done.
Use exact object names from the scene. Colors are RGBA floats 0-1.
Do not explain what you are doing — just call the tools."""
		},
		{
			"role": "user",
			"content": f"Execute ALL steps of this plan:\n{plan}\n\nCurrent scene:\n{scene_state}\n\nCall tools for every step. Do not stop early."
		}
	]

	actions_taken: list[dict[str, Any]] = []

	while True:
		response = client.chat.completions.create(
			model=EXECUTOR_MODEL,
			tools=TOOLS,  # type: ignore[arg-type]
			messages=messages,
			extra_body={"think": False},
		)

		msg = response.choices[0].message
		finish = response.choices[0].finish_reason

		if msg.content:
			print(f"[Executor] {msg.content}")

		if finish == "stop" or not msg.tool_calls:
			break

		messages.append(msg)  # type: ignore[arg-type]

		for tool_call in msg.tool_calls:
			name = tool_call.function.name
			args: dict[str, Any] = json.loads(tool_call.function.arguments)

			print(f"[Tool]   {name}({json.dumps(args)})")
			result = dispatch_tool(name, args)
			preview = result[:150] + "..." if len(result) > 150 else result
			print(f"[Result] {preview}")

			actions_taken.append({"tool": name, "args": args, "result": result})

			messages.append({
				"role": "tool",
				"tool_call_id": tool_call.id,
				"content": result,
			})

	return actions_taken


def run_critic(user_instruction: str, plan: str, actions: list[dict[str, Any]]) -> str:
	print(f"\n[Critic/{PLANNER_MODEL}] Reviewing...")
	actions_text = "\n".join(
		f"- {a['tool']}({json.dumps(a['args'])}) → {str(a['result'])[:80]}"
		for a in actions
	)
	messages: list[ChatCompletionMessageParam] = [
		{
			"role": "system",
			"content": "You are a Blender 3D critic. Write a 2 sentence summary of what was done. Be direct."
		},
		{
			"role": "user",
			"content": f"Instruction: {user_instruction}\nActions taken:\n{actions_text}\nWrite a 2 sentence summary."
		}
	]
	response = client.chat.completions.create(
		model=PLANNER_MODEL,
		messages=messages,
	)
	result = response.choices[0].message.content
	return result if result else "Done."

def run_agent(user_instruction: str) -> str:
	print(f"\n{'='*50}")
	print(f"Task: {user_instruction}")
	print(f"{'='*50}")

	scene_state = str(get_scene_state())
	print(f"\n[Scene] {scene_state[:200]}")

	plan = run_planner(user_instruction, scene_state)
	actions = run_executor(plan, scene_state)
	summary = run_critic(user_instruction, plan, actions)

	print(f"\n[Done] {summary}")
	return summary