#version 450

#include "../utility/geometry.glslh"
#include "../utility/quaternion.glslh"

layout (location = 0) in vec2 inUV;

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
layout(set = 0, binding = 3) uniform sampler2D brdfLut;
layout(set = 0, binding = 4) uniform sampler2D gbufferDepth;
layout(set = 0, binding = 5) uniform sampler2D gbufferAlbedo;
layout(set = 0, binding = 6) uniform sampler2D gbufferNormal;
layout(set = 0, binding = 7) uniform sampler2D gbufferRoughnessMetallic;
layout(set = 0, binding = 8) readonly buffer LightBuffer
{
    LightData lights[];
};

layout(push_constant) uniform PC {
    uint numLights;
} pc;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;
const float epsilon = 1e-7;

#define LIGHT_TYPE_POINT 0
#define LIGHT_TYPE_SPOT 1
#define LIGHT_TYPE_AREA 2
#define LIGHT_TYPE_DIRECTIONAL 3

vec3 CalcPointLight(LightData light, vec3 position, vec3 normal, vec3 albedo, float roughness, float metallic);\
vec3 CalcDirectionalLight(LightData light, vec3 position, vec3 normal, vec3 albedo, float roughness, float metallic);
vec3 CalcSpotLight(LightData light, vec3 position, vec3 normal, vec3 albedo, float roughness, float metallic);
vec3 CalcAreaLight(LightData light, vec3 position, vec3 normal, vec3 albedo, float roughness, float metallic);

float DistributionGGX(float N_dot_H, float roughness);
float GeometrySchlick(float N_dot_V, float roughness);
vec3 FresnelSchlick(float V_dot_H, vec3 F0);

void main()
{
    // Screen space UV coordinates for sampling G-buffer textures
    vec3 albedo = texture(gbufferAlbedo, inUV).rgb;
    vec3 normal = unpackViewNormal(texture(gbufferNormal, inUV).rg);
    float roughness = max(texture(gbufferRoughnessMetallic, inUV).r, 0.02); // Safe roughness
    float metallic = texture(gbufferRoughnessMetallic, inUV).g;
    float depth = texture(gbufferDepth, inUV).r;

    vec3 position = depthToViewPosition(depth, inUV, inverse(cameraUbo.proj));

    outColor = vec4(vec3(0.0), 1.0);
    for (uint i = 0; i < pc.numLights; ++i)
    {
        LightData light = lights[i];
        switch (light.type)
        {
        case LIGHT_TYPE_POINT:
            outColor += vec4(CalcPointLight(light, position, normal, albedo, roughness, metallic), 1.0);
            break;
        case LIGHT_TYPE_DIRECTIONAL:
            outColor += vec4(CalcDirectionalLight(light, position, normal, albedo, roughness, metallic), 1.0);
            break;
        case LIGHT_TYPE_SPOT:
            outColor += vec4(CalcSpotLight(light, position, normal, albedo, roughness, metallic), 1.0);
            break;
        case LIGHT_TYPE_AREA:
            outColor += vec4(CalcAreaLight(light, position, normal, albedo, roughness, metallic), 1.0);
            break;
        }
    }
    outColor = vec4(outColor.rgb * albedo, 1.0);
}

vec3 CalcPointLight(LightData light, vec3 position, vec3 normal, vec3 albedo, float roughness, float metallic)
{
    vec3 lightViewPos = (cameraUbo.view * vec4(light.position, 1.0)).xyz;
    vec3 L = normalize(lightViewPos - position);
    vec3 N = normal;
    
    vec3 V = normalize(-position);
    vec3 H = normalize(L + V);

    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 F = FresnelSchlick(max(dot(H, V), 0.0), F0);
    float D = DistributionGGX(max(dot(N, H), 0.0), roughness);
    float G = pow(GeometrySchlick(max(dot(N, V), 0.0), roughness), 2.0);
    
    vec3 Diffuse = (1.0 - F) * albedo / PI;
    vec3 Specular = (D * F * G) / (4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + epsilon);
    
    vec3 BRDF = Diffuse + Specular;

    // Point lights are treated as a dirac delta function. Using the sifting property, we get that
    // the ingtegral over the hemisphere is just the value of the function at the point light direction
    float L_dot_N = max(dot(L, N), 0.0);
    return light.color * light.intensity * BRDF * L_dot_N;
}

vec3 CalcDirectionalLight(LightData light, vec3 position, vec3 normal, vec3 albedo, float roughness, float metallic)
{
    vec3 L = normalize((cameraUbo.view * vec4(-quaternionToForwardVector(light.rotation), 0.0)).xyz);
    vec3 N = normal;
    
    vec3 V = normalize(-position);
    vec3 H = normalize(L + V);

    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 F = FresnelSchlick(max(dot(H, V), 0.0), F0);
    float D = DistributionGGX(max(dot(N, H), 0.0), roughness);
    float G = pow(GeometrySchlick(max(dot(N, V), 0.0), roughness), 2.0);
    
    vec3 Diffuse = (1.0 - F) * albedo / PI;
    vec3 Specular = (D * F * G) / (4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + epsilon);
    
    vec3 BRDF = Diffuse + Specular;

    // Directional lights are also treated as a dirac delta function. Using the sifting property, we get that
    // the ingtegral over the hemisphere is just the value of the function at the directional light direction
    float L_dot_N = max(dot(L, N), 0.0);
    return light.color * light.intensity * BRDF * L_dot_N;
}

vec3 CalcSpotLight(LightData light, vec3 position, vec3 normal, vec3 albedo, float roughness, float metallic)
{
    return vec3(0.0); // Placeholder, implement spot light calculations here
}

vec3 CalcAreaLight(LightData light, vec3 position, vec3 normal, vec3 albedo, float roughness, float metallic)
{
    return vec3(0.0); // Placeholder, implement area light calculations here
}

float DistributionGGX(float N_dot_H, float roughness)
{
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float N_dot_H2 = N_dot_H * N_dot_H;
    float numerator = alpha2;
    float d = (N_dot_H2 * (alpha2 - 1.0) + 1.0);
    float denominator = PI * d * d + epsilon;
    return numerator / denominator;
}

float GeometrySchlick(float N_dot_V, float roughness)
{
    float k = (roughness + 1.0);
    k = (k * k) / 8.0;
    float numerator = N_dot_V;
    float denominator = N_dot_V * (1.0 - k) + k + epsilon;
    return numerator / denominator;
}

vec3 FresnelSchlick(float V_dot_H, vec3 F0)
{
    return F0 + (1.0 - F0) * exp2(-5.55473 * V_dot_H - 6.98316 * V_dot_H);   
}