#version 450

layout(location = 0) in vec3 inNormal;

layout(location = 0) out vec4 outColor;

// Edited live from the ImGui panel in examples/python/interactive.py.
layout(set = 0, binding = 1) uniform Settings
{
    vec4 albedo;
    vec4 lightDirection; // xyz, towards the light
} settings;

void main()
{
    const vec3  n       = normalize(inNormal);
    const float diffuse = max(dot(n, normalize(settings.lightDirection.xyz)), 0.0);
    outColor            = vec4(settings.albedo.rgb * (0.12 + 0.88 * diffuse), 1.0);
}
