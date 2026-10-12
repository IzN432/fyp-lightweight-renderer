#version 450

layout(set = 0, binding = 0) uniform CameraUbo
{
    mat4 view;
    mat4 proj;
    mat4 viewProj;
    mat4 invView;
    mat4 invProj;
    vec4 cameraPosition;
} cameraUbo;

layout(std430, set = 0, binding = 1) readonly buffer Particles
{
    vec4 positions[];
} particles;

layout(push_constant) uniform ParticleBillboardPC
{
    vec4 colorAndRadius;
} pc;

layout(location = 0) out vec2 outSphereCoord;
layout(location = 1) out vec3 outViewCenter;

const vec2 corners[6] = vec2[](
    vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
    vec2(-1.0, -1.0), vec2( 1.0,  1.0), vec2(-1.0,  1.0)
);

void main()
{
    vec2 corner = corners[gl_VertexIndex];
    vec3 viewCenter = (cameraUbo.view * vec4(particles.positions[gl_InstanceIndex].xyz, 1.0)).xyz;
    vec3 viewPosition = viewCenter + vec3(corner * pc.colorAndRadius.w, 0.0);

    outSphereCoord = corner;
    outViewCenter = viewCenter;
    gl_Position = cameraUbo.proj * vec4(viewPosition, 1.0);
}
