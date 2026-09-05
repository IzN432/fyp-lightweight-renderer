#version 450

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

void main()
{
    outColor = vec4(1.0 - vec3(uv, 0.35), 1.0);
}
