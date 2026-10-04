#version 450

// One camera-facing quad (6 vertices) per particle, drawn instanced: gl_InstanceIndex picks the particle.

layout(set = 0, binding = 0) uniform Camera
{
    mat4 view;
    mat4 proj;
} camera;

struct Particle
{
    vec4 position; // xyz, w unused
    vec4 color;    // rgb, a unused
};

layout(std430, set = 0, binding = 1) readonly buffer Particles
{
    Particle particles[];
};

layout(push_constant) uniform Size
{
    float radius;
} size;

layout(location = 0) out vec3 outColor;
layout(location = 1) out vec2 outCorner;

const vec2 kCorners[6] = vec2[](vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, -1), vec2(1, 1), vec2(-1, 1));

void main()
{
    Particle particle = particles[gl_InstanceIndex];
    vec2     corner   = kCorners[gl_VertexIndex];
    vec4     viewPos  = camera.view * vec4(particle.position.xyz, 1.0);
    viewPos.xy += corner * size.radius;
    gl_Position = camera.proj * viewPos;
    outColor    = particle.color.rgb;
    outCorner   = corner;
}
