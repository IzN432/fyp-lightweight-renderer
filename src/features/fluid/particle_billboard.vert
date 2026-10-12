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

struct ParticleGpuData
{
    vec4 positionAndRadius;
    vec4 colorAndType;
};

layout(std430, set = 0, binding = 1) readonly buffer Particles
{
    ParticleGpuData particles[];
} particles;

layout(location = 0) out vec2 outSphereCoord;
layout(location = 1) out vec3 outViewCenter;
layout(location = 2) flat out vec4 outColorAndType;
layout(location = 3) flat out float outRadius;

const vec2 corners[6] = vec2[](
    vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0),
    vec2(-1.0, -1.0), vec2( 1.0,  1.0), vec2(-1.0,  1.0)
);

void main()
{
    vec2 corner = corners[gl_VertexIndex];
    ParticleGpuData particle = particles.particles[gl_InstanceIndex];
    vec3 viewCenter = (cameraUbo.view * vec4(particle.positionAndRadius.xyz, 1.0)).xyz;
    vec3 viewPosition = viewCenter + vec3(corner * particle.positionAndRadius.w, 0.0);

    outSphereCoord = corner;
    outViewCenter = viewCenter;
    outColorAndType = particle.colorAndType;
    outRadius = particle.positionAndRadius.w;
    gl_Position = cameraUbo.proj * vec4(viewPosition, 1.0);
}
