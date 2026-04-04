from agent_side.agent import run_agent

def main():
	print("Blender Multi-Agent | type 'quit' to exit")
	print("Models: planner=qwen3.5:9b  executor=qwen3.5:4b")
	print("Make sure Blender is open with server.py running.\n")

	while True:
		try:
			instruction = input("You: ").strip()
		except (EOFError, KeyboardInterrupt):
			print("\nBye!")
			break
		if not instruction:
			continue
		if instruction.lower() in ("quit", "exit", "q"):
			print("Bye!")
			break
		run_agent(instruction)
		print()

if __name__ == "__main__":
	main()