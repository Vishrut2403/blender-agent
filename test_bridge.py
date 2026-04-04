from agent_side.bridge import send_code

print("--- Test 1: print from Blender ---")
r = send_code("print('Hello from inside Blender!')")
print(r)

print("\n--- Test 2: list all objects ---")
r = send_code("""
names = [obj.name for obj in bpy.data.objects]
result = names
""")
print(r)

print("\n--- Test 3: add a cube ---")
r = send_code("""
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
result = "Cube added at origin"
""")
print(r)

print("\n--- Test 4: intentional error ---")
r = send_code("this_will_fail()")
print(r)