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

layout(set = 0, binding = 3) uniform sampler2D overlayDepth;

layout(set = 0, binding = 4) uniform sampler2D pbr;

layout(set = 0, binding = 5) uniform sampler2D overlay;

layout(set = 0, binding = 6) uniform sampler2D overlayPoints;

layout(location = 0) out vec4 outColor;

void main()
{
    float depth = min(texture(gbufferDepth, inUV).r, texture(overlayDepth, inUV).r);

    vec3 baseColor;
    if (1.0 - depth < 1e-6)
    {
        vec2 ndc = inUV * 2.0 - 1.0;
        vec4 clip = vec4(ndc, 1.0, 1.0);

        vec4 viewPos = inverse(cameraUbo.proj) * clip;
        vec3 viewDir = normalize(viewPos.xyz / max(viewPos.w, 1e-6));
        vec3 worldDir = normalize(mat3(inverse(cameraUbo.view)) * viewDir);

        vec3 hdr = texture(skybox, worldDir).rgb;
        baseColor = hdr / (hdr + vec3(1.0));
    }
    else
    {
        vec3 pbrColor = texture(pbr, inUV).rgb;
        vec4 overlaySample = texture(overlay, inUV);
        vec3 color = (1.0 - overlaySample.a) * pbrColor + overlaySample.a * overlaySample.rgb;
        baseColor = color / (color + vec3(1.0));
    }

    vec4 pointsSample = texture(overlayPoints, inUV);
    outColor = vec4(mix(baseColor, pointsSample.rgb, pointsSample.a), 1.0);
}