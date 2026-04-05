import sys
from agent_side.agent import run_agent as run_local, SessionMemory
from agent_side.agent_gemini import run_agent as run_gemini


def main():
	use_gemini = "--gemini" in sys.argv

	if use_gemini:
		print("Blender Agent | Mode: Gemini 1.5 Flash (cloud)")
		_run = run_gemini
	else:
		print("Blender Agent | Mode: qwen3.5:4b (local)")
		_run = run_local

	print("Type 'quit' to exit.\n")

	memory = SessionMemory() if not use_gemini else None

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

		if use_gemini:
			_run(instruction)
		else:
			memory = run_local(instruction, memory)

		print()


if __name__ == "__main__":
	main()