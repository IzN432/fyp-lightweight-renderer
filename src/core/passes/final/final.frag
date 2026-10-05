#version 450

#include "../utility/tonemap.glslh"

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

layout(set = 0, binding = 3) uniform sampler2D overlayDepth;

layout(set = 0, binding = 4) uniform sampler2D pbr;

layout(set = 0, binding = 5) uniform sampler2D overlay;

layout(set = 0, binding = 6) uniform sampler2D overlayPoints;

layout(set = 0, binding = 7) uniform sampler2D heatmap;

layout(set = 0, binding = 8) uniform sampler2D hbaoAo;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform FinalPC
{
    vec4 backgroundColor;
    uint showEnvironmentBackground;
} pc;

void main()
{
    float depth = min(texture(gbufferDepth, inUV).r, texture(overlayDepth, inUV).r);

    vec3 baseColor;
    if (isBackground(depth))
    {
        if (pc.showEnvironmentBackground != 0u)
        {
            vec3 worldDir = viewRayWorld(inUV, cameraUbo.view, cameraUbo.proj);
            baseColor = reinhard(texture(skybox, worldDir).rgb);
        }
        else
        {
            baseColor = pc.backgroundColor.rgb;
        }
    }
    else
    {
        vec3 pbrColor = texture(pbr, inUV).rgb;
        vec4 heatmapSample = texture(heatmap, inUV);
        vec3 litColor = mix(pbrColor, heatmapSample.rgb, heatmapSample.a);
        vec4 overlaySample = texture(overlay, inUV);
        vec3 color = (1.0 - overlaySample.a) * litColor + overlaySample.a * overlaySample.rgb;
        baseColor = reinhard(color);
    }

    vec4 pointsSample = texture(overlayPoints, inUV);
    outColor = vec4(mix(baseColor, pointsSample.rgb, pointsSample.a), 1.0);
    // outColor = vec4(vec3(1.0 - texture(hbaoAo, inUV).r), 1.0);
}
