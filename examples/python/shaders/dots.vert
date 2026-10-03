#version 450

// One small screen-aligned quad (6 vertices) per instance, centred on the trail point that is this
// instance's per-instance attribute.
layout(location = 0) in vec2 inCenter; // per instance

layout(push_constant) uniform PC
{
    vec2 halfSize; // in clip space, so dots stay square at any aspect ratio
} pc;

void main()
{
    const vec2 corners[6] = vec2[](vec2(-1, -1), vec2(1, -1), vec2(1, 1), vec2(-1, -1), vec2(1, 1), vec2(-1, 1));
    gl_Position           = vec4(inCenter + corners[gl_VertexIndex] * pc.halfSize, 0.0, 1.0);
}
