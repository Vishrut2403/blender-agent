from agent_side.tools import (
    get_scene_state,
    add_object,
    set_material,
    set_location,
    render_scene,
)

print("--- Scene state (before) ---")
print(get_scene_state())

print("\n--- Add a sphere ---")
print(add_object("SPHERE", location=(2, 0, 0)))

print("\n--- Add a cone ---")
print(add_object("CONE", location=(-2, 0, 0)))

print("\n--- Paint the sphere red ---")
print(set_material("Sphere", color=(1, 0, 0, 1)))

print("\n--- Move the cone up ---")
print(set_location("Cone", 0, 0, 2))

print("\n--- Scene state (after) ---")
print(get_scene_state())

print("\n--- Render ---")
print(render_scene("/tmp/agent_render.png"))
print("Check /tmp/agent_render.png")