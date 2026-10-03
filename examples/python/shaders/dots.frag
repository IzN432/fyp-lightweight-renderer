#version 450

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform Style
{
    vec4 lineColor;
    vec4 dotColor;
} style;

void main()
{
    outColor = style.dotColor;
}
