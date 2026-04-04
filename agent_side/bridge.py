import socket
import json
from dataclasses import dataclass
from typing import Optional


HOST = "localhost"
PORT = 6789


@dataclass
class BlenderResponse:
	ok: bool
	stdout: str = ""
	error: Optional[str] = None
	value: Optional[str] = None

	def __str__(self):
		if not self.ok:
			return f"ERROR:\n{self.error}"
		parts = []
		if self.stdout:
			parts.append(self.stdout.strip())
		if self.value:
			parts.append(f"=> {self.value}")
		return "\n".join(parts) if parts else "(no output)"


def send_code(code: str, host: str = HOST, port: int = PORT) -> BlenderResponse:
	payload = json.dumps({"code": code}) + "\n"
	try:
		with socket.create_connection((host, port), timeout=10) as sock:
			sock.sendall(payload.encode("utf-8"))
			raw = b""
			while True:
				chunk = sock.recv(4096)
				if not chunk:
					break
				raw += chunk
				if raw.endswith(b"\n"):
					break
		data = json.loads(raw.decode("utf-8").strip())
		return BlenderResponse(**data)
	except ConnectionRefusedError:
		return BlenderResponse(
			ok=False,
			error="Could not connect to Blender. Is the server script running?"
		)
	except Exception as e:
		return BlenderResponse(ok=False, error=str(e))