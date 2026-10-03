#version 450

// Deliberately slow, so each frame stays on the GPU long enough for the CPU to start the next one.
layout(location = 0) in vec2 inUV;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PC
{
    uint  iterations;
    float time;
} pc;

void main()
{
    float v = inUV.x * 0.37 + inUV.y * 0.11;
    for (uint i = 0u; i < pc.iterations; ++i)
    {
        v = fract(sin(v * 12.9898 + float(i)) * 43758.5453);
    }
    const vec3 wave = 0.5 + 0.5 * cos(vec3(0.0, 2.0, 4.0) + inUV.x * 6.0 + pc.time);
    outColor        = vec4(wave * (0.85 + 0.15 * v), 1.0);
}
