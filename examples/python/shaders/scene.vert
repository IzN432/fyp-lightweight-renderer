#version 450

// Vertex layout built in scene_viewer.py from lr.Mesh arrays (engine attribute names in comments).
layout(location = 0) in vec3 inPosition; // Mesh.positions
layout(location = 1) in vec3 inNormal;   // Mesh.attribute("normal")
layout(location = 2) in vec2 inUv;       // Mesh.attribute("uv")

layout(set = 0, binding = 0) uniform Camera
{
    mat4 view;
    mat4 proj;
} camera;

layout(push_constant) uniform Draw
{
    mat4 model; // SceneObject.world_matrix; the rest of the block is read by scene.frag
} draw;

layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec2 outUv;

void main()
{
    outNormal   = mat3(draw.model) * inNormal;
    outUv       = inUv;
    gl_Position = camera.proj * camera.view * draw.model * vec4(inPosition, 1.0);
}
