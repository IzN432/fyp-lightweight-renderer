#version 450

layout(location = 0) in vec2 inUV;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D hdrColor;

layout(push_constant) uniform Post
{
    float exposure;
    float vignette;
    float applyGamma; // 1 when the swapchain is UNORM and needs manual sRGB encoding
} pc;

void main()
{
    vec3 c = texture(hdrColor, inUV).rgb * pc.exposure;
    c      = c / (1.0 + c); // Reinhard

    vec2 d = inUV - 0.5;
    c *= 1.0 - pc.vignette * dot(d, d) * 2.0;

    if (pc.applyGamma > 0.5)
    {
        c = pow(c, vec3(1.0 / 2.2));
    }
    outColor = vec4(c, 1.0);
}
