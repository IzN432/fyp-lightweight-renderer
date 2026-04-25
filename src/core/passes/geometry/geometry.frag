#version 450

#extension GL_EXT_nonuniform_qualifier : require

#include "../utility/geometry.glslh"

struct MaterialData
{
    vec4  baseColorFactor;
    vec4  emissiveFactor;
    float roughnessFactor;
    float metallicFactor;
    float _pad[2];
};

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inTangent;
layout(location = 3) in vec2 inUv;

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

layout(set = 0, binding = 1) uniform sampler2D diffuseTex[];
layout(set = 0, binding = 2) uniform sampler2D normalTex[];
layout(set = 0, binding = 3) uniform sampler2D metallicRoughnessTex[];
layout(set = 0, binding = 4) uniform sampler2D emissiveTex[];
layout(set = 0, binding = 5) readonly buffer FaceGroupIndices
{
    uint values[];
} faceGroupIndices;
layout(set = 0, binding = 6) readonly buffer Materials
{
    MaterialData data[];
} materials;

layout(location = 0) out vec4 outAlbedo;
layout(location = 1) out vec2 outNormal;
layout(location = 2) out vec4 outMaterial;

void main()
{
	uint faceGroupIndex = faceGroupIndices.values[gl_PrimitiveID];
	MaterialData mat = materials.data[faceGroupIndex];

	vec3 albedo = texture(diffuseTex[nonuniformEXT(faceGroupIndex)], inUv).rgb * mat.baseColorFactor.rgb;
	float roughness = texture(metallicRoughnessTex[nonuniformEXT(faceGroupIndex)], inUv).g * mat.roughnessFactor;
	float metallic = texture(metallicRoughnessTex[nonuniformEXT(faceGroupIndex)],  inUv).b * mat.metallicFactor;
	
	// ========== SECTION 1 - Normal Mapping ==========

	vec3 N = normalize(inNormal); // Geometric world-space normal
	vec3 T = normalize(inTangent.xyz);
	T = normalize(T - dot(T, N) * N); // Gram-Schmidt orthogonalization to ensure T is orthogonal to N
	vec3 B = cross(N, T) * inTangent.w; // inTangent.w is the handedness, which can be used to determine the direction of the bitangent
	mat3 TBN = mat3(T, B, N);

	vec3 tangentNormal = texture(normalTex[nonuniformEXT(faceGroupIndex)], inUv).rgb * 2.0 - 1.0;
	vec3 worldNormal = TBN * tangentNormal;
	vec3 viewNormal = normalize(mat3(cameraUbo.view) * worldNormal);
	outNormal = packViewNormal(viewNormal);

	// ========== SECTION 2 - Material Properties ==========
	
	outAlbedo = vec4(albedo, 1.0);
	outMaterial = vec4(clamp(roughness, 0.0, 1.0), clamp(metallic, 0.0, 1.0), 0.0, 0.0);
}

