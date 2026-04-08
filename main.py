import sys
import logging
import webbrowser
import threading

from agent_side.agent import run_agent as run_local, SessionMemory
from agent_side.agent_gemini import run_agent as run_gemini
from agent_side.config import LOCAL_MODEL, UI_HOST, UI_PORT

logging.basicConfig(
	level=logging.WARNING,
	format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

QUIT_COMMANDS = frozenset({"quit", "exit", "q"})


def _open_browser(host: str, port: int) -> None:
	"""Open the browser after a short delay so uvicorn has time to start."""
	import time
	time.sleep(1.2)
	webbrowser.open(f"http://{host}:{port}")


def run_terminal() -> None:
	"""Original terminal mode — unchanged from Phase 5."""
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


def run_ui() -> None:
	"""
	Web UI mode — starts the FastAPI server and opens the browser.

	Import is deferred so that the terminal mode doesn't pay the cost of
	importing FastAPI / uvicorn when they're not needed.
	"""
	# Deferred import: only needed in UI mode.
	from ui.server import start_server  # noqa: PLC0415

	print(f"Blender Agent | Mode: Web UI → http://{UI_HOST}:{UI_PORT}")
	print("Press Ctrl+C to stop.\n")

	# Open the browser in a background thread so it doesn't block uvicorn.
	threading.Thread(
		target=_open_browser, args=(UI_HOST, UI_PORT), daemon=True
	).start()

	start_server(host=UI_HOST, port=UI_PORT)


def main() -> None:
	if "--ui" in sys.argv:
		run_ui()
	else:
		run_terminal()


if __name__ == "__main__":
	main()