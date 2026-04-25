#version 450

#include "../utility/geometry.glslh"

struct LightData
{
    vec3 position;
    uint type;
    vec4 rotation;

    vec3 color;
    float intensity;
    
    float innerConeAngle;
    float outerConeAngle;
    vec2 areaSize;
};

layout(set = 0, binding = 0) uniform CameraUbo
{
    mat4 view;
    mat4 proj;
    mat4 viewProj;
    mat4 invView;
    mat4 invProj;
    vec3 cameraPosition;
    float padding;
} cameraUbo;

layout(set = 0, binding = 1) uniform samplerCube irradianceMap;
layout(set = 0, binding = 2) uniform samplerCube prefilterMap;
layout(set = 0, binding = 3) uniform sampler2D gbufferDepth;
layout(set = 0, binding = 4) uniform sampler2D gbufferAlbedo;
layout(set = 0, binding = 5) uniform sampler2D gbufferNormal;
layout(set = 0, binding = 6) uniform sampler2D gbufferRoughnessMetallic;
layout(set = 0, binding = 7) readonly buffer LightBuffer
{
    LightData lights[];
};

layout(push_constant) uniform PC {
    uint numLights;
} pc;

layout(location = 0) out vec4 outColor;

void main()
{
    // Screen space UV coordinates for sampling G-buffer textures
    vec2 uv = gl_FragCoord.xy / vec2(textureSize(gbufferAlbedo, 0));
    
    vec3 albedo = texture(gbufferAlbedo, uv).rgb;
    vec3 normal = unpackViewNormal(texture(gbufferNormal, uv).rg);
    float roughness = texture(gbufferRoughnessMetallic, uv).r;
    float metallic = texture(gbufferRoughnessMetallic, uv).g;
    float depth = texture(gbufferDepth, uv).r;

    vec3 position = depthToViewPosition(depth, uv, inverse(cameraUbo.proj));

    outColor = vec4(vec3(0.0), 1.0);
    for (uint i = 0; i < pc.numLights; ++i)
    {
        LightData light = lights[i];

        vec3 lightViewPos = (cameraUbo.view * vec4(light.position, 1.0)).xyz;
        vec3 L = normalize(lightViewPos - position);
        vec3 N = normal;
        float L_dot_N = max(0.0, dot(L, N));
        
        outColor += vec4(light.color * light.intensity * L_dot_N, 1.0);
    }
    outColor = vec4(outColor.rgb * albedo, 1.0);
}