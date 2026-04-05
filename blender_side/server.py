import bpy
import socket
import threading
import json
import traceback
import io
import sys

# Config 
# Cannot import from agent_side — this runs inside Blender's Python environment.
# Keep constants local and in sync with agent_side/config.py manually.
HOST            = "localhost"
PORT            = 6789
TIMEOUT         = 120       # seconds to wait for Blender main thread execution
RECV_BUFFER     = 4096
LISTEN_BACKLOG  = 5

# State 
_server_thread: threading.Thread | None = None
_server_socket: socket.socket | None = None


# Code Execution 

def execute_code(code: str) -> dict:
	"""Execute arbitrary bpy code and capture stdout, return value, and errors."""
	old_stdout = sys.stdout
	sys.stdout = buffer = io.StringIO()

	result: dict = {"ok": True, "stdout": "", "error": None, "value": None}
	try:
		local_ns: dict = {}
		exec(code, {"bpy": bpy, "__builtins__": __builtins__}, local_ns)
		if "result" in local_ns:
			result["value"] = str(local_ns["result"])
	except Exception:
		result["ok"] = False
		result["error"] = traceback.format_exc()
	finally:
		sys.stdout = old_stdout
		result["stdout"] = buffer.getvalue()

	return result


# Client Handler 

def handle_client(conn: socket.socket) -> None:
	"""Receive a code payload, execute it on the main thread, send back the result."""
	with conn:
		raw = b""
		while True:
			chunk = conn.recv(RECV_BUFFER)
			if not chunk:
				break
			raw += chunk
			if raw.endswith(b"\n"):
				break

		try:
			payload = json.loads(raw.decode("utf-8").strip())
			code = payload.get("code", "")

			result_container: dict = {}
			event = threading.Event()

			def main_thread_task() -> None:
				# bpy is not thread-safe — must execute on Blender's main thread
				result_container["data"] = execute_code(code)
				event.set()

			bpy.app.timers.register(main_thread_task, first_interval=0.0)

			completed = event.wait(timeout=TIMEOUT)
			if not completed:
				response = {"ok": False, "error": f"Timeout after {TIMEOUT}s"}
			else:
				response = result_container.get(
					"data", {"ok": False, "error": "No result returned"}
				)

		except json.JSONDecodeError as e:
			response = {"ok": False, "error": f"Invalid JSON payload: {e}"}
		except Exception:
			response = {"ok": False, "error": traceback.format_exc()}

		conn.sendall((json.dumps(response) + "\n").encode("utf-8"))


# Server Loop 

def server_loop() -> None:
	global _server_socket
	_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	_server_socket.bind((HOST, PORT))
	_server_socket.listen(LISTEN_BACKLOG)
	print(f"[BlenderBridge] Listening on {HOST}:{PORT}")

	while True:
		try:
			conn, addr = _server_socket.accept()
			print(f"[BlenderBridge] Connection from {addr}")
			t = threading.Thread(target=handle_client, args=(conn,), daemon=True)
			t.start()
		except OSError:
			# Socket was closed by stop_server()
			print("[BlenderBridge] Server socket closed, shutting down.")
			break


# Start / Stop 

def start_server() -> None:
	global _server_thread
	if _server_thread and _server_thread.is_alive():
		print("[BlenderBridge] Already running.")
		return
	_server_thread = threading.Thread(target=server_loop, daemon=True)
	_server_thread.start()
	print("[BlenderBridge] Server started.")


def stop_server() -> None:
	global _server_socket
	if _server_socket:
		_server_socket.close()
		_server_socket = None
	print("[BlenderBridge] Server stopped.")


# Entry Point 
start_server()