# Blender Bridge Configuration 
BLENDER_HOST    = "localhost"
BLENDER_PORT    = 6789
SOCKET_TIMEOUT  = 120

# Local LLM (Ollama) Configuration
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY  = "ollama"
LOCAL_MODEL     = "qwen3.5:4b"

# Agent Behaviour Configuration
MAX_TOOL_CALLS    = 10
MAX_PLAN_RETRIES  = 3
ACTION_LOG_WINDOW = 5

# Defaults Configuration
DEFAULT_RENDER_PATH = "/tmp/blender_render.png"