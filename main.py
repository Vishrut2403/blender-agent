import sys
import logging

from agent_side.agent import run_agent as run_local, SessionMemory
from agent_side.agent_gemini import run_agent as run_gemini
from agent_side.config import LOCAL_MODEL

# Logging setup 
# Configure once at the entry point. All modules inherit this config.
logging.basicConfig(
    level=logging.WARNING,          # change to DEBUG to see all internal logs
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Constants 
QUIT_COMMANDS = frozenset({"quit", "exit", "q"})


def main() -> None:
    use_gemini = "--gemini" in sys.argv

    if use_gemini:
        print("Blender Agent | Mode: Gemini (cloud)")
    else:
        print(f"Blender Agent | Mode: {LOCAL_MODEL} (local)")

    print("Type 'quit' to exit.\n")

    memory: SessionMemory | None = SessionMemory() if not use_gemini else None

    while True:
        try:
            instruction = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not instruction:
            continue

        if instruction.lower() in QUIT_COMMANDS:
            print("Bye!")
            break

        logger.debug("User instruction: %s", instruction)

        if use_gemini:
            run_gemini(instruction)
        else:
            memory = run_local(instruction, memory)

        print()


if __name__ == "__main__":
    main()