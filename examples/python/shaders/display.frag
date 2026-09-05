#version 450

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

void main()
{
    vec3 color = vec3(uv, 0.35);
#ifdef APPLY_GAMMA
    color = pow(color, vec3(1.0 / 2.2));
#endif
    outColor = vec4(color, 1.0);
}
