from agent_side.bridge import send_code, BlenderResponse


def get_scene_state() -> BlenderResponse:
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


def add_object(kind: str = "CUBE", location: tuple = (0, 0, 0)) -> BlenderResponse:
	ops = {
		"CUBE":     "bpy.ops.mesh.primitive_cube_add",
		"SPHERE":   "bpy.ops.mesh.primitive_uv_sphere_add",
		"CYLINDER": "bpy.ops.mesh.primitive_cylinder_add",
		"CONE":     "bpy.ops.mesh.primitive_cone_add",
		"PLANE":    "bpy.ops.mesh.primitive_plane_add",
		"MONKEY":   "bpy.ops.mesh.primitive_monkey_add",
	}
	kind = kind.upper()
	if kind not in ops:
		return BlenderResponse(
			ok=False,
			error=f"Unknown object type '{kind}'. Choose from: {list(ops.keys())}"
		)
	x, y, z = location
	code = f"""
before = set(bpy.data.objects.keys())
{ops[kind]}(location=({x}, {y}, {z}))
after = set(bpy.data.objects.keys())
new_names = after - before
added_name = new_names.pop() if new_names else "unknown"
result = f"Added {{added_name}} at ({x}, {y}, {z})"
"""
	return send_code(code)


def set_material(object_name: str, color: tuple = (1, 1, 1, 1)) -> BlenderResponse:
	r, g, b, a = color
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


def set_location(object_name: str, x: float, y: float, z: float) -> BlenderResponse:
	code = f"""
obj = bpy.data.objects.get("{object_name}")
if obj is None:
	raise ValueError("Object '{object_name}' not found in scene")
obj.location = ({x}, {y}, {z})
result = f"Moved {object_name} to ({x}, {y}, {z})"
"""
	return send_code(code)


def ensure_camera() -> BlenderResponse:
	"""Add a camera and point it at the origin if none exists."""
	code = """
import bpy, math
if not any(o.type == 'CAMERA' for o in bpy.data.objects):
	bpy.ops.object.camera_add(location=(7.36, -6.93, 4.96))
	cam = sorted([o for o in bpy.data.objects if o.type == 'CAMERA'], key=lambda o: o.name)[-1]
	cam.rotation_euler = (math.radians(63.6), 0, math.radians(46.7))
	bpy.context.scene.camera = cam
	result = "Camera added"
else:
	result = "Camera already present"
"""
	return send_code(code)


def ensure_light() -> BlenderResponse:
	"""Add a sun lamp if no lights exist."""
	code = """
import bpy
if not any(o.type == 'LIGHT' for o in bpy.data.objects):
	bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
	result = "Light added"
else:
	result = "Light already present"
"""
	return send_code(code)


def render_scene(output_path: str = "/tmp/blender_render.png") -> BlenderResponse:
	ensure_camera()
	ensure_light()
	code = f"""
import bpy
bpy.context.scene.render.engine = 'BLENDER_EEVEE'
bpy.context.scene.render.filepath = "{output_path}"
bpy.context.scene.render.image_settings.file_format = "PNG"
bpy.ops.render.render(write_still=True)
result = "Rendered to {output_path}"
"""
	return send_code(code)