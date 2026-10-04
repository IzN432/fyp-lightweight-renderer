#version 450

// Distance fog over the engine's lit HDR image: reads PbrPass's "pbr" and GeometryPass's "gbufferDepth",
// writes HDR colour for CompositePass to tone map.

layout(location = 0) in vec2 inUV;

// SceneGpu's camera UBO (see docs/python_building_blocks.md).
layout(set = 0, binding = 0) uniform Camera
{
    mat4 view;
    mat4 proj;
    mat4 viewProj;
    mat4 invView;
    mat4 invProj;
    vec4 position;
} camera;

layout(set = 0, binding = 1) uniform sampler2D litColor;
layout(set = 0, binding = 2) uniform sampler2D depth;

layout(push_constant) uniform Fog
{
    vec4 color;    // rgb, a unused
    float density; // per world unit
} fog;

layout(location = 0) out vec4 outColor;

void main()
{
    vec3  lit = texture(litColor, inUV).rgb;
    float d   = texture(depth, inUV).r;

    vec4 viewPos  = camera.invProj * vec4(inUV * 2.0 - 1.0, d, 1.0);
    float dist    = length(viewPos.xyz / viewPos.w);
    float amount  = 1.0 - exp(-fog.density * dist);
    outColor = vec4(mix(lit, fog.color.rgb, amount), 1.0);
}
