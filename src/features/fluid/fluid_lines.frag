#version 450

// As in fluid_particles.frag: the swapchain is sRGB, so the picked colour is converted to linear
// here and Vulkan encodes it back on write, leaving the configured colour intact on screen.

layout(location = 0) in vec3 inColor;

layout(location = 0) out vec4 outColor;

vec3 srgbToLinear(vec3 c)
{
    const bvec3 cutoff = lessThanEqual(c, vec3(0.04045));
    const vec3  low    = c / 12.92;
    const vec3  high   = pow((c + 0.055) / 1.055, vec3(2.4));
    return mix(high, low, cutoff);
}

void main()
{
    outColor = vec4(srgbToLinear(inColor), 1.0);
}
