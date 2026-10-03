#version 450

layout(location = 0) in vec3 inPosition;    // per vertex
layout(location = 1) in vec3 inNormal;      // per vertex
layout(location = 2) in vec4 inOffsetScale; // per instance (written by instances_cull.comp)
layout(location = 3) in vec4 inColor;       // per instance

layout(set = 0, binding = 0) uniform Camera
{
    mat4 view;
    mat4 proj;
} camera;

layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec3 outColor;

void main()
{
    const vec3 world = inPosition * inOffsetScale.w + inOffsetScale.xyz;
    outNormal        = inNormal;
    outColor         = inColor.rgb;
    gl_Position      = camera.proj * camera.view * vec4(world, 1.0);
}
