import bpy
import socket
import threading
import json
import traceback
import io
import sys


HOST = "localhost"
PORT = 6789
_server_thread = None
_server_socket = None


def execute_code(code: str) -> dict:
	old_stdout = sys.stdout
	sys.stdout = buffer = io.StringIO()

	result = {"stdout": "", "error": None, "ok": True}
	try:
		local_ns = {}
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


def handle_client(conn: socket.socket):
	with conn:
		raw = b""
		while True:
			chunk = conn.recv(4096)
			if not chunk:
				break
			raw += chunk
			if raw.endswith(b"\n"):
				break

		try:
			payload = json.loads(raw.decode("utf-8").strip())
			code = payload.get("code", "")
			# Schedule execution on Blender's main thread
			result_container = {}
			event = threading.Event()

			def main_thread_task():
				result_container["data"] = execute_code(code)
				event.set()

			bpy.app.timers.register(main_thread_task, first_interval=0.0)
			event.wait(timeout=120)
			response = result_container.get("data", {"ok": False, "error": "Timeout"})
		except Exception:
			response = {"ok": False, "error": traceback.format_exc()}

		conn.sendall((json.dumps(response) + "\n").encode("utf-8"))


def server_loop():
	global _server_socket
	_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	_server_socket.bind((HOST, PORT))
	_server_socket.listen(5)
	print(f"[BlenderBridge] Listening on {HOST}:{PORT}")

	while True:
		try:
			conn, addr = _server_socket.accept()
			t = threading.Thread(target=handle_client, args=(conn,), daemon=True)
			t.start()
		except OSError:
			break


def start_server():
	global _server_thread
	if _server_thread and _server_thread.is_alive():
		print("[BlenderBridge] Already running.")
		return
	_server_thread = threading.Thread(target=server_loop, daemon=True)
	_server_thread.start()
	print("[BlenderBridge] Server started.")


def stop_server():
	if _server_socket:
		_server_socket.close()
	print("[BlenderBridge] Server stopped.")


start_server()