#version 450

layout(location = 0) in vec3 inNormal;

layout(location = 0) out vec4 outColor;

void main()
{
    vec3  n       = normalize(inNormal);
    float diffuse = max(dot(n, normalize(vec3(0.6, 1.0, 0.4))), 0.0);
    vec3  albedo  = vec3(0.55, 0.58, 0.62);
    outColor      = vec4(albedo * (0.15 + 0.85 * diffuse), 1.0);
}
