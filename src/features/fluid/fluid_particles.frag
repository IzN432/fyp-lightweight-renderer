#version 450

// Cheap headlight diffuse so the spheres read as round instead of as flat discs.
//
// The swapchain is an sRGB format, so Vulkan encodes whatever this shader writes from linear to
// sRGB on the way out. A colour picked in the UI is already an sRGB value, so it is converted to
// linear here first; that round trip is what makes the colour on screen match the one configured.

layout(location = 0) in vec3 inNormal;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PC
{
    vec4 albedo; // RGB is sRGB, straight from the colour picker
    float visualScale;
} pc;

vec3 srgbToLinear(vec3 c)
{
    const bvec3 cutoff = lessThanEqual(c, vec3(0.04045));
    const vec3  low    = c / 12.92;
    const vec3  high   = pow((c + 0.055) / 1.055, vec3(2.4));
    return mix(high, low, cutoff);
}

void main()
{
    const vec3  lightDir = normalize(vec3(0.4, 0.8, 0.5));
    const float diffuse  = max(dot(normalize(inNormal), lightDir), 0.0);
    // Floor the shading at an ambient term so particles facing away stay visible.
    const float shade = mix(0.45, 1.0, diffuse);
    outColor = vec4(srgbToLinear(pc.albedo.rgb) * shade, 1.0);
}
