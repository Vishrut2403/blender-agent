import socket
import json
import logging
from dataclasses import dataclass
from typing import Optional

from agent_side.config import BLENDER_HOST, BLENDER_PORT, SOCKET_TIMEOUT

logger = logging.getLogger(__name__)


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


def send_code(
	code: str,
	host: str = BLENDER_HOST,
	port: int = BLENDER_PORT,
	timeout: int = SOCKET_TIMEOUT,
) -> BlenderResponse:
	payload = json.dumps({"code": code}) + "\n"
	logger.debug("Sending %d chars to Blender at %s:%d", len(code), host, port)
	try:
		with socket.create_connection((host, port), timeout=timeout) as sock:
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
		response = BlenderResponse(**data)
		if not response.ok:
			logger.warning("Blender returned error: %s", response.error)
		else:
			logger.debug("Blender response: %s", response)
		return response
	except ConnectionRefusedError:
		logger.error("Connection refused — is blender_side/server.py running?")
		return BlenderResponse(
			ok=False,
			error="Could not connect to Blender. Is the server script running?"
		)
	except Exception as e:
		logger.exception("Unexpected error communicating with Blender")
		return BlenderResponse(ok=False, error=str(e))