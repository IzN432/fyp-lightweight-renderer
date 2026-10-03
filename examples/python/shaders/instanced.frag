#version 450

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec4 outColor;

void main()
{
    const float diffuse = max(dot(normalize(inNormal), normalize(vec3(0.4, 1.0, 0.3))), 0.0);
    outColor            = vec4(inColor * (0.25 + 0.75 * diffuse), 1.0);
}
