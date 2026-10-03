#version 450

layout(location = 0) in vec3 inNormal;

layout(location = 0) out vec4 outColor;

void main()
{
    vec3 n = normalize(inNormal);
    vec3 l = normalize(vec3(0.6, 1.0, 0.4));

    // Normal-tinted albedo, hemisphere ambient and one directional light, in linear HDR —
    // the post pass tone-maps it.
    vec3 albedo  = 0.35 + 0.65 * (0.5 + 0.5 * n);
    vec3 ambient = mix(vec3(0.04, 0.04, 0.06), vec3(0.25, 0.25, 0.3), 0.5 + 0.5 * n.y);
    float diffuse = max(dot(n, l), 0.0);

    outColor = vec4(albedo * (ambient + 2.5 * diffuse), 1.0);
}
