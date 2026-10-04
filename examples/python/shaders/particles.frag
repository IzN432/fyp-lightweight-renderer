#version 450

layout(location = 0) in vec3 inColor;
layout(location = 1) in vec2 inCorner;

layout(location = 0) out vec4 outColor;

void main()
{
    float r2 = dot(inCorner, inCorner);
    if (r2 > 1.0)
    {
        discard; // round particles
    }
    // Shade as a little sphere lit from the camera.
    outColor = vec4(inColor * (0.35 + 0.65 * sqrt(1.0 - r2)), 1.0);
}
