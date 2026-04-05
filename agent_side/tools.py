import logging
from typing import Optional
from agent_side.bridge import send_code, BlenderResponse
from agent_side.config import DEFAULT_RENDER_PATH

logger = logging.getLogger(__name__)

# Supported mesh primitives 

SUPPORTED_OBJECTS: dict[str, str] = {
	"CUBE":     "bpy.ops.mesh.primitive_cube_add",
	"SPHERE":   "bpy.ops.mesh.primitive_uv_sphere_add",
	"CYLINDER": "bpy.ops.mesh.primitive_cylinder_add",
	"CONE":     "bpy.ops.mesh.primitive_cone_add",
	"PLANE":    "bpy.ops.mesh.primitive_plane_add",
	"MONKEY":   "bpy.ops.mesh.primitive_monkey_add",
}

# Camera defaults 

_CAMERA_LOCATION  = (7.36, -6.93, 4.96)
_CAMERA_ROTATION  = (63.6, 0.0, 46.7)   # degrees, converted to radians in bpy code
_LIGHT_LOCATION   = (5.0, 5.0, 10.0)
_RENDER_ENGINE    = "BLENDER_EEVEE"
_RENDER_FORMAT    = "PNG"


# Tools 

def get_scene_state() -> BlenderResponse:
	"""Return all objects in the current Blender scene as a JSON list."""
	logger.debug("Fetching scene state")
	code = """
import json
objects = []
for obj in bpy.data.objects:
	mat_name = None
	if obj.data and hasattr(obj.data, 'materials') and obj.data.materials:
		mat_name = obj.data.materials[0].name if obj.data.materials[0] else None
	objects.append({
		"name": obj.name,
		"type": obj.type,
		"location": list(obj.location),
		"material": mat_name
	})
result = json.dumps(objects, indent=2)
"""
	return send_code(code)


def add_object(
	kind: str = "CUBE",
	location: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> BlenderResponse:
	"""Add a mesh primitive to the scene at the given location."""
	kind = kind.upper()
	if kind not in SUPPORTED_OBJECTS:
		logger.warning("Unsupported object kind requested: %s", kind)
		return BlenderResponse(
			ok=False,
			error=f"Unknown object type '{kind}'. Choose from: {list(SUPPORTED_OBJECTS.keys())}"
		)
	x, y, z = location
	op = SUPPORTED_OBJECTS[kind]
	logger.debug("Adding %s at (%s, %s, %s)", kind, x, y, z)
	code = f"""
before = set(bpy.data.objects.keys())
{op}(location=({x}, {y}, {z}))
after = set(bpy.data.objects.keys())
new_names = after - before
added_name = new_names.pop() if new_names else "unknown"
result = f"Added {{added_name}} at ({x}, {y}, {z})"
"""
	return send_code(code)


def set_material(
	object_name: str,
	color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> BlenderResponse:
	"""Apply a Principled BSDF material with the given RGBA color to an object."""
	if len(color) == 3:
		r, g, b, a = *color, 1.0 
	elif len(color) == 4:
		r, g, b, a = color
	else:
		return BlenderResponse(
			ok=False,
			error=f"color must be [R, G, B] or [R, G, B, A], got {len(color)} values"
		)

	logger.debug("Setting material on %s → RGBA(%.2f, %.2f, %.2f, %.2f)", object_name, r, g, b, a)
	code = f"""
obj = bpy.data.objects.get("{object_name}")
if obj is None:
	raise ValueError("Object '{object_name}' not found in scene")

mat_name = "AgentMat_{object_name}"
mat = bpy.data.materials.get(mat_name) or bpy.data.materials.new(name=mat_name)
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get("Principled BSDF")
if bsdf:
	bsdf.inputs["Base Color"].default_value = ({r}, {g}, {b}, {a})

if obj.data.materials:
	obj.data.materials[0] = mat
else:
	obj.data.materials.append(mat)

result = f"Material applied to {object_name} with color ({r:.2f}, {g:.2f}, {b:.2f})"
"""
	return send_code(code)


def set_location(
	object_name: str,
	x: float,
	y: float,
	z: float,
) -> BlenderResponse:
	"""Move an object to an absolute position."""
	logger.debug("Moving %s to (%.2f, %.2f, %.2f)", object_name, x, y, z)
	code = f"""
obj = bpy.data.objects.get("{object_name}")
if obj is None:
	raise ValueError("Object '{object_name}' not found in scene")
obj.location = ({x}, {y}, {z})
result = f"Moved {object_name} to ({x}, {y}, {z})"
"""
	return send_code(code)


def ensure_camera() -> BlenderResponse:
	"""Add a camera pointed at the origin if none exists."""
	cx, cy, cz = _CAMERA_LOCATION
	rx, ry, rz = _CAMERA_ROTATION
	logger.debug("Ensuring camera exists")
	code = f"""
import bpy, math
if not any(o.type == 'CAMERA' for o in bpy.data.objects):
	bpy.ops.object.camera_add(location=({cx}, {cy}, {cz}))
	cam = sorted([o for o in bpy.data.objects if o.type == 'CAMERA'], key=lambda o: o.name)[-1]
	cam.rotation_euler = (math.radians({rx}), math.radians({ry}), math.radians({rz}))
	bpy.context.scene.camera = cam
	result = "Camera added"
else:
	result = "Camera already present"
"""
	return send_code(code)


def ensure_light() -> BlenderResponse:
	"""Add a sun lamp if no lights exist."""
	lx, ly, lz = _LIGHT_LOCATION
	logger.debug("Ensuring light exists")
	code = f"""
import bpy
if not any(o.type == 'LIGHT' for o in bpy.data.objects):
	bpy.ops.object.light_add(type='SUN', location=({lx}, {ly}, {lz}))
	result = "Light added"
else:
	result = "Light already present"
"""
	return send_code(code)


def render_scene(output_path: str = DEFAULT_RENDER_PATH) -> BlenderResponse:
	"""Render the current scene to a PNG file."""
	logger.info("Rendering scene to %s", output_path)
	ensure_camera()
	ensure_light()
	code = f"""
import bpy
bpy.context.scene.render.engine = '{_RENDER_ENGINE}'
bpy.context.scene.render.filepath = "{output_path}"
bpy.context.scene.render.image_settings.file_format = '{_RENDER_FORMAT}'
bpy.ops.render.render(write_still=True)
result = "Rendered to {output_path}"
"""
	return send_code(code)