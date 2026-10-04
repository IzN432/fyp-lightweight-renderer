#version 450

// MATERIAL_COUNT is defined by scene_viewer.py when it compiles this file (one texture per material).
layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec2 inUv;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 1) uniform sampler2D baseColors[MATERIAL_COUNT];

layout(push_constant) uniform Draw
{
    layout(offset = 64) vec4 baseColorFactor; // the material's "baseDiffuse"
    layout(offset = 80) uint material;
} draw;

void main()
{
    // As the engine's geometry.frag: base colour texture times the material's base colour factor.
    const vec4  albedo  = texture(baseColors[draw.material], inUv) * draw.baseColorFactor;
    const vec3  n       = normalize(inNormal);
    const float diffuse = max(dot(n, normalize(vec3(0.4, 1.0, 0.6))), 0.0);
    outColor            = vec4(albedo.rgb * (0.25 + 0.75 * diffuse), 1.0);
}
