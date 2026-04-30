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
#define LIGHT_TYPE_IMAGE 4

vec3 CalcPointLight(LightData light, vec3 position, vec3 normal, vec3 albedo, float roughness, float metallic);\
vec3 CalcDirectionalLight(LightData light, vec3 position, vec3 normal, vec3 albedo, float roughness, float metallic);
vec3 CalcSpotLight(LightData light, vec3 position, vec3 normal, vec3 albedo, float roughness, float metallic);
vec3 CalcAreaLight(LightData light, vec3 position, vec3 normal, vec3 albedo, float roughness, float metallic);
vec3 CalcImageLight(LightData light, vec3 position, vec3 normal, vec3 albedo, float roughness, float metallic);

float DistributionGGX(float N_dot_H, float roughness);
float VisibilitySmithGGX(float N_dot_V, float N_dot_L, float roughness);
vec3 FresnelSchlick(float cosTheta, vec3 F0);
vec3 FresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness);

void main()
{
    // Screen space UV coordinates for sampling G-buffer textures
    vec3 albedo = texture(gbufferAlbedo, inUV).rgb;
    vec3 normal = unpackViewNormal(texture(gbufferNormal, inUV).rg);
    float roughness = max(texture(gbufferRoughnessMetallic, inUV).r, 0.045);
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
        case LIGHT_TYPE_IMAGE:
            outColor += vec4(CalcImageLight(light, position, normal, albedo, roughness, metallic), 1.0);
            break;
        }
    }
    outColor = vec4(outColor.rgb, 1.0);
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
    float Vis = VisibilitySmithGGX(max(dot(N, V), 0.0), max(dot(N, L), 0.0), roughness);

    vec3 Diffuse = (1.0 - F) * (1.0 - metallic) * albedo / PI;
    vec3 Specular = D * F * Vis;

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
    float Vis = VisibilitySmithGGX(max(dot(N, V), 0.0), max(dot(N, L), 0.0), roughness);

    vec3 Diffuse = (1.0 - F) * (1.0 - metallic) * albedo / PI;
    vec3 Specular = D * F * Vis;

    vec3 BRDF = Diffuse + Specular;

    // Directional lights are also treated as a dirac delta function. Using the sifting property, we get that
    // the ingtegral over the hemisphere is just the value of the function at the directional light direction
    float L_dot_N = max(dot(L, N), 0.0);
    return light.color * light.intensity * BRDF * L_dot_N;
}

vec3 CalcSpotLight(LightData light, vec3 position, vec3 normal, vec3 albedo, float roughness, float metallic)
{
    vec3 lightViewPos = (cameraUbo.view * vec4(light.position, 1.0)).xyz;
    vec3 L = normalize(lightViewPos - position);
    vec3 N = normal;
    
    vec3 V = normalize(-position);
    vec3 H = normalize(L + V);

    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 F = FresnelSchlick(max(dot(H, V), 0.0), F0);
    float D = DistributionGGX(max(dot(N, H), 0.0), roughness);
    float Vis = VisibilitySmithGGX(max(dot(N, V), 0.0), max(dot(N, L), 0.0), roughness);

    vec3 Diffuse = (1.0 - F) * (1.0 - metallic) * albedo / PI;
    vec3 Specular = D * F * Vis;

    vec3 BRDF = Diffuse + Specular;

    // Spot lights are treated as a dirac delta function. Using the sifting property, we get that
    // the ingtegral over the hemisphere is just the value of the function at the spot light direction
    float L_dot_N = max(dot(L, N), 0.0);

    // Spot lights differ from point lights in that they fall off from inner cone to outer cone.
    vec3 Dir = normalize((cameraUbo.view * vec4(quaternionToForwardVector(light.rotation), 0.0)).xyz);
    float L_dot_Dir = dot(-L, Dir);

    float cosOuter = cos(radians(light.outerConeAngle));
    float cosInner = cos(radians(light.innerConeAngle));

    float spotlightIntensity = smoothstep(cosOuter, cosInner, L_dot_Dir);

    return light.color * light.intensity * BRDF * L_dot_N * spotlightIntensity;
}

vec3 CalcAreaLight(LightData light, vec3 position, vec3 normal, vec3 albedo, float roughness, float metallic)
{
    return vec3(0.0); // Placeholder, implement area light calculations here
}

vec3 CalcImageLight(LightData light, vec3 position, vec3 normal, vec3 albedo, float roughness, float metallic)
{
    vec3 V_world = normalize(vec3(cameraUbo.invView * vec4(-position, 0.0)));
    vec3 N_world = normalize(vec3(cameraUbo.invView * vec4(normal, 0.0)));
    vec3 R_world = reflect(-V_world, N_world);

    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 F = FresnelSchlickRoughness(max(dot(N_world, V_world), 0.0), F0, roughness);
    
    vec3 kS = F;
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - metallic;

    vec3 irradiance = texture(irradianceMap, N_world).rgb;
    vec3 Diffuse = kD * irradiance * albedo;

    const float MAX_REFLECTION_LOD = 4.0;
    vec3 prefilteredColor = textureLod(prefilterMap, R_world, roughness * MAX_REFLECTION_LOD).rgb;
    vec2 envBRDF = texture(brdfLut, vec2(max(dot(normalize(R_world + V_world), V_world), 0.0), roughness)).rg;
    vec3 Specular = prefilteredColor * (F * envBRDF.x + envBRDF.y);

    return Diffuse + Specular;
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

// Combined height-correlated Smith GGX visibility: G(NdotV, NdotL) / (4 * NdotV * NdotL)
// Analytically combines the geometry term and denominator, avoiding the division singularity.
float VisibilitySmithGGX(float N_dot_V, float N_dot_L, float roughness)
{
    float a2 = roughness * roughness * roughness * roughness;
    float GGXV = N_dot_L * sqrt(N_dot_V * N_dot_V * (1.0 - a2) + a2);
    float GGXL = N_dot_V * sqrt(N_dot_L * N_dot_L * (1.0 - a2) + a2);
    return 0.5 / (GGXV + GGXL + epsilon);
}

vec3 FresnelSchlick(float V_dot_H, vec3 F0)
{
    return F0 + (1.0 - F0) * exp2((-5.55473 * V_dot_H - 6.98316) * V_dot_H);
}

vec3 FresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * exp2((-5.55473 * cosTheta - 6.98316) * cosTheta);
}