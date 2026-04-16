#version 450

layout(location = 0) in vec2 inUV;

layout(set = 0, binding = 0) uniform CameraUbo
{
	mat4 view;
	mat4 proj;
	mat4 viewProj;
	vec4 cameraPosition;
} cameraUbo;

layout(set = 0, binding = 1) uniform samplerCube skybox;

layout(set = 0, binding = 2) uniform sampler2D gbufferDepth;
layout(set = 0, binding = 3) uniform sampler2D gbufferAlbedo;
layout(set = 0, binding = 4) uniform sampler2D gbufferNormal;
layout(set = 0, binding = 5) uniform sampler2D gbufferMaterial;

layout(location = 0) out vec4 outColor;

void main()
{
    float depth = texture(gbufferDepth, inUV).r;
    if (1.0 - depth < 1e-6)
    {
        // If depth is 1.0, it means the pixel didn't hit any geometry. We can directly sample the skybox without reconstructing the view ray.
        // Reconstruct a view ray from fullscreen UV, then rotate it to world space.
        vec2 ndc = inUV * 2.0 - 1.0;
        vec4 clip = vec4(ndc, 1.0, 1.0);

        vec4 viewPos = inverse(cameraUbo.proj) * clip;
        vec3 viewDir = normalize(viewPos.xyz / max(viewPos.w, 1e-6));
        vec3 worldDir = normalize(mat3(inverse(cameraUbo.view)) * viewDir);

        vec3 hdr = texture(skybox, worldDir).rgb;

        // Simple tone map for HDR sky values before writing to the swapchain.
        vec3 mapped = hdr / (hdr + vec3(1.0));
        outColor = vec4(mapped, 1.0);
    }
    else
    {
        outColor = vec4(texture(gbufferAlbedo, inUV).rgb, 1.0);
    }
}