import sys
from agent_side.agent import run_agent as run_local
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
		_run(instruction)
		print()


if __name__ == "__main__":
	main()