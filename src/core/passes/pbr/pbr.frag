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
layout(set = 0, binding = 4) uniform sampler2D ltc1; // inverse M for area light cosine warp
layout(set = 0, binding = 5) uniform sampler2D ltc2; // GGX norm, fresnel, unused, horizon-clip
layout(set = 0, binding = 6) uniform sampler2D gbufferDepth;
layout(set = 0, binding = 7) uniform sampler2D gbufferAlbedo;
layout(set = 0, binding = 8) uniform sampler2D gbufferNormal;
layout(set = 0, binding = 9) uniform sampler2D gbufferRoughnessMetallic;
layout(set = 0, binding = 10) uniform sampler2D gbufferEmissive;
layout(set = 0, binding = 11) readonly buffer LightBuffer
{
    LightData lights[];
};
layout(set = 0, binding = 12) uniform sampler2D ao;

layout(push_constant) uniform PC {
    uint pfMips;
    uint numLights;
} pc;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;
const float epsilon = 1e-4;

// LTC lookup table is 64x64, indexed by (roughness, sqrt(1 - N_dot_V))
const float LTC_LUT_SIZE = 64.0;
const float LTC_LUT_SCALE = (LTC_LUT_SIZE - 1.0) / LTC_LUT_SIZE;
const float LTC_LUT_BIAS = 0.5 / LTC_LUT_SIZE;

#define LIGHT_TYPE_POINT 0
#define LIGHT_TYPE_SPOT 1
#define LIGHT_TYPE_AREA 2
#define LIGHT_TYPE_DIRECTIONAL 3
#define LIGHT_TYPE_IMAGE 4

vec3 CalcPointLight(LightData light, vec3 position, vec3 normal, vec3 albedo, float roughness, float metallic);\
vec3 CalcDirectionalLight(LightData light, vec3 position, vec3 normal, vec3 albedo, float roughness, float metallic);
vec3 CalcSpotLight(LightData light, vec3 position, vec3 normal, vec3 albedo, float roughness, float metallic);
vec3 CalcAreaLight(LightData light, vec3 position, vec3 normal, vec3 albedo, float roughness, float metallic);
vec3 CalcImageLight(LightData light, vec3 position, vec3 normal, vec3 albedo, float roughness, float metallic, float ao);

float DistributionGGX(float N_dot_H, float roughness);
float GeometrySchlickGGX(float N_dot_V, float roughness);
float GeometrySmithGGX(float N_dot_V, float N_dot_L, float roughness);
vec3 FresnelSchlick(float cosTheta, vec3 F0);
vec3 FresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness);

vec3 IntegrateEdgeVec(vec3 v1, vec3 v2);
void ClipQuadToHorizon(inout vec3 L[5], out int n);
vec3 LTC_Evaluate(vec3 N, vec3 V, vec3 P, mat3 Minv, vec3 points[4], bool twoSided);

void main()
{
    // Screen space UV coordinates for sampling G-buffer textures
    vec3 albedo = texture(gbufferAlbedo, inUV).rgb;
    vec3 normal = unpackViewNormal(texture(gbufferNormal, inUV).rg);
    float roughness = max(texture(gbufferRoughnessMetallic, inUV).r, 0.045);
    float metallic = texture(gbufferRoughnessMetallic, inUV).g;
    float depth = texture(gbufferDepth, inUV).r;

    vec3 position = depthToViewPosition(depth, inUV, cameraUbo.invProj);

    vec3 color = vec3(0.0);
    for (uint i = 0; i < pc.numLights; ++i)
    {
        LightData light = lights[i];
        switch (light.type)
        {
        case LIGHT_TYPE_POINT:
            color += CalcPointLight(light, position, normal, albedo, roughness, metallic);
            break;
        case LIGHT_TYPE_DIRECTIONAL:
            color += CalcDirectionalLight(light, position, normal, albedo, roughness, metallic);
            break;
        case LIGHT_TYPE_SPOT:
            color += CalcSpotLight(light, position, normal, albedo, roughness, metallic);
            break;
        case LIGHT_TYPE_AREA:
            color += CalcAreaLight(light, position, normal, albedo, roughness, metallic);
            break;
        case LIGHT_TYPE_IMAGE:
            color += CalcImageLight(light, position, normal, albedo, roughness, metallic, texture(ao, inUV).r);
            break;
        }
    }

    // Emissive is a direct, unlit contribution (e.g. the visual quad representing an area light) —
    // it doesn't go through the light loop above.
    color += texture(gbufferEmissive, inUV).rgb;

    outColor = vec4(color, 1.0);
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
    float G = GeometrySmithGGX(max(dot(N, V), 0.0), max(dot(N, L), 0.0), roughness);

    vec3 Diffuse = (1.0 - F) * (1.0 - metallic) * albedo / PI;
    vec3 Specular = D * F * G / (4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + epsilon);

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
    float G = GeometrySmithGGX(max(dot(N, V), 0.0), max(dot(N, L), 0.0), roughness);

    vec3 Diffuse = (1.0 - F) * (1.0 - metallic) * albedo / PI;
    vec3 Specular = D * F * G / (4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + epsilon);

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
    float G = GeometrySmithGGX(max(dot(N, V), 0.0), max(dot(N, L), 0.0), roughness);

    vec3 Diffuse = (1.0 - F) * (1.0 - metallic) * albedo / PI;
    vec3 Specular = D * F * G / (4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + epsilon);

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
    // Rectangular area light shaded with Linearly Transformed Cosines (Heitz et al. 2016).
    // The light is a quad centered at light.position, spanning light.areaSize along its local
    // right/up axes, emitting towards its local forward direction.
    vec3 lightViewPos = (cameraUbo.view * vec4(light.position, 1.0)).xyz;
    vec3 right = (cameraUbo.view * vec4(quaternionToRightVector(light.rotation), 0.0)).xyz * (light.areaSize.x * 0.5);
    vec3 up = (cameraUbo.view * vec4(quaternionToUpVector(light.rotation), 0.0)).xyz * (light.areaSize.y * 0.5);

    vec3 points[4];
    points[0] = lightViewPos - right - up;
    points[1] = lightViewPos - right + up;
    points[2] = lightViewPos + right + up;
    points[3] = lightViewPos + right - up;

    vec3 N = normal;
    vec3 V = normalize(-position);
    float N_dot_V = clamp(dot(N, V), 0.0, 1.0);

    vec2 uv = vec2(roughness, sqrt(1.0 - N_dot_V));
    uv = uv * LTC_LUT_SCALE + LTC_LUT_BIAS;

    vec4 t1 = texture(ltc1, uv);
    vec4 t2 = texture(ltc2, uv);

    mat3 Minv = mat3(
        vec3(t1.x, 0.0, t1.y),
        vec3(0.0,  1.0, 0.0),
        vec3(t1.z, 0.0, t1.w)
    );

    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    vec3 specular = LTC_Evaluate(N, V, position, Minv, points, false);
    specular *= F0 * t2.x + (vec3(1.0) - F0) * t2.y;

    vec3 diffuse = LTC_Evaluate(N, V, position, mat3(1.0), points, false);

    vec3 result = light.color * light.intensity * (albedo * (1.0 - metallic) * diffuse + specular);
    return result / (2.0 * PI);
}

vec3 CalcImageLight(LightData light, vec3 position, vec3 normal, vec3 albedo, float roughness, float metallic, float ao)
{
    vec3 V_world = normalize(vec3(cameraUbo.invView * vec4(-position, 0.0)));
    vec3 N_world = normalize(vec3(cameraUbo.invView * vec4(normal, 0.0)));
    vec3 R_world = reflect(-V_world, N_world);

    float N_dot_V = clamp(dot(N_world, V_world), 0.001, 0.999);

    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 F = FresnelSchlickRoughness(N_dot_V, F0, roughness);

    vec3 kS = F;
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - metallic;

    vec3 irradiance = texture(irradianceMap, N_world).rgb;
    vec3 Diffuse = kD * irradiance * albedo;

    vec3 prefilteredColor = textureLod(prefilterMap, R_world, roughness * float(pc.pfMips - 1u)).rgb;
    vec2 envBRDF = texture(brdfLut, vec2(N_dot_V, roughness)).rg;
    vec3 Specular = prefilteredColor * (F * envBRDF.x + envBRDF.y);

    return (Diffuse + Specular) * light.color * light.intensity * ao;
}

float DistributionGGX(float N_dot_H, float roughness)
{
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float N_dot_H2 = N_dot_H * N_dot_H;
    float numerator = alpha2;
    float d = (N_dot_H2 * (alpha2 - 1.0) + 1.0);
    float denominator = PI * d * d;
    return numerator / denominator;
}

float GeometrySchlickGGX(float N_dot_V, float roughness)
{
    float k = roughness + 1;
    k = (k * k) / 8.0; // UE4's implementation uses this remapping for better visual results

    return N_dot_V / (N_dot_V * (1.0 - k) + k);
}

float GeometrySmithGGX(float N_dot_V, float N_dot_L, float roughness)
{
    return GeometrySchlickGGX(N_dot_V, roughness) * GeometrySchlickGGX(N_dot_L, roughness);
}

vec3 FresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

vec3 FresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Analytic integral of the cosine-weighted solid angle contribution of one polygon edge,
// using the rational polynomial fit for acos() from Heitz et al.'s reference implementation.
vec3 IntegrateEdgeVec(vec3 v1, vec3 v2)
{
    float x = dot(v1, v2);
    float y = abs(x);

    float a = 0.8543985 + (0.4965155 + 0.0145206 * y) * y;
    float b = 3.4175940 + (4.1616724 + y) * y;
    float v = a / b;

    float theta_sintheta = (x > 0.0) ? v : 0.5 * inversesqrt(max(1.0 - x * x, 1e-7)) - v;

    return cross(v1, v2) * theta_sintheta;
}

// Clips the (already transformed) quad L[0..3] against the horizon plane z = 0, keeping only the
// part of the polygon on the visible side (z > 0) of the shading point's tangent plane. Vertices are
// in the LTC-transformed local frame, so z corresponds to the cosine-weighted "up" axis. Produces a
// polygon of up to 5 vertices (config numbers reference which of V1..V4 survive, 1-indexed to match
// the reference derivation). Ported from Heitz et al.'s reference implementation.
void ClipQuadToHorizon(inout vec3 L[5], out int n)
{
    // detect clipping config
    int config = 0;
    if (L[0].z > 0.0) config += 1;
    if (L[1].z > 0.0) config += 2;
    if (L[2].z > 0.0) config += 4;
    if (L[3].z > 0.0) config += 8;

    // clip
    n = 0;

    if (config == 0)
    {
        // clip all
    }
    else if (config == 1) // V1 clip V2 V3 V4
    {
        n = 3;
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
        L[2] = -L[3].z * L[0] + L[0].z * L[3];
    }
    else if (config == 2) // V2 clip V1 V3 V4
    {
        n = 3;
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
    }
    else if (config == 3) // V1 V2 clip V3 V4
    {
        n = 4;
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
        L[3] = -L[3].z * L[0] + L[0].z * L[3];
    }
    else if (config == 4) // V3 clip V1 V2 V4
    {
        n = 3;
        L[0] = -L[3].z * L[2] + L[2].z * L[3];
        L[1] = -L[1].z * L[2] + L[2].z * L[1];
    }
    else if (config == 5) // V1 V3 clip V2 V4 (impossible)
    {
        n = 0;
    }
    else if (config == 6) // V2 V3 clip V1 V4
    {
        n = 4;
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
        L[3] = -L[3].z * L[2] + L[2].z * L[3];
    }
    else if (config == 7) // V1 V2 V3 clip V4
    {
        n = 5;
        L[4] = -L[3].z * L[0] + L[0].z * L[3];
        L[3] = -L[3].z * L[2] + L[2].z * L[3];
    }
    else if (config == 8) // V4 clip V1 V2 V3
    {
        n = 3;
        L[0] = -L[0].z * L[3] + L[3].z * L[0];
        L[1] = -L[2].z * L[3] + L[3].z * L[2];
        L[2] = L[3];
    }
    else if (config == 9) // V1 V4 clip V2 V3
    {
        n = 4;
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
        L[2] = -L[2].z * L[3] + L[3].z * L[2];
    }
    else if (config == 10) // V2 V4 clip V1 V3 (impossible)
    {
        n = 0;
    }
    else if (config == 11) // V1 V2 V4 clip V3
    {
        n = 5;
        L[4] = L[3];
        L[3] = -L[2].z * L[3] + L[3].z * L[2];
        L[2] = -L[2].z * L[1] + L[1].z * L[2];
    }
    else if (config == 12) // V3 V4 clip V1 V2
    {
        n = 4;
        L[1] = -L[1].z * L[2] + L[2].z * L[1];
        L[0] = -L[0].z * L[3] + L[3].z * L[0];
    }
    else if (config == 13) // V1 V3 V4 clip V2
    {
        n = 5;
        L[4] = L[3];
        L[3] = L[2];
        L[2] = -L[1].z * L[2] + L[2].z * L[1];
        L[1] = -L[1].z * L[0] + L[0].z * L[1];
    }
    else if (config == 14) // V2 V3 V4 clip V1
    {
        n = 5;
        L[4] = -L[0].z * L[3] + L[3].z * L[0];
        L[0] = -L[0].z * L[1] + L[1].z * L[0];
    }
    else if (config == 15) // V1 V2 V3 V4 (nothing clipped)
    {
        n = 4;
    }

    if (n == 3)
    {
        L[3] = L[0];
    }
    if (n == 4)
    {
        L[4] = L[0];
    }
}

// Evaluates the LTC-transformed irradiance of a quadrilateral light for a linearly transformed
// cosine distribution defined by Minv (identity for diffuse, LTC1-derived for GGX specular).
// The quad is clipped against the shading point's horizon plane before integration, so lights that
// straddle the surface (partially below the horizon) are handled correctly rather than as all-or-nothing.
vec3 LTC_Evaluate(vec3 N, vec3 V, vec3 P, mat3 Minv, vec3 points[4], bool twoSided)
{
    vec3 T1 = normalize(V - N * dot(V, N));
    vec3 T2 = cross(N, T1);

    Minv = Minv * transpose(mat3(T1, T2, N));

    // Polygon in the LTC-transformed local frame; index 4 is reserved for clipping.
    vec3 L[5];
    L[0] = Minv * (points[0] - P);
    L[1] = Minv * (points[1] - P);
    L[2] = Minv * (points[2] - P);
    L[3] = Minv * (points[3] - P);

    int n;
    ClipQuadToHorizon(L, n);

    if (n == 0)
    {
        return vec3(0.0);
    }

    L[0] = normalize(L[0]);
    L[1] = normalize(L[1]);
    L[2] = normalize(L[2]);
    L[3] = normalize(L[3]);
    L[4] = normalize(L[4]);

    float sum = 0.0;
    sum += IntegrateEdgeVec(L[0], L[1]).z;
    sum += IntegrateEdgeVec(L[1], L[2]).z;
    sum += IntegrateEdgeVec(L[2], L[3]).z;
    if (n >= 4)
    {
        sum += IntegrateEdgeVec(L[3], L[4]).z;
    }
    if (n == 5)
    {
        sum += IntegrateEdgeVec(L[4], L[0]).z;
    }

    sum = twoSided ? abs(sum) : max(0.0, sum);

    return vec3(sum);
}