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
layout(location = 2) in vec2 inUv;

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
layout(location = 3) out vec4 outPreview;

void main()
{
	uint faceGroupIndex = faceGroupIndices.values[gl_PrimitiveID];
	MaterialData mat = materials.data[faceGroupIndex];

	vec3 albedo = texture(diffuseTex[nonuniformEXT(faceGroupIndex)], inUv).rgb * mat.baseColorFactor.rgb;
	float roughness = texture(metallicRoughnessTex[nonuniformEXT(faceGroupIndex)], inUv).g * mat.roughnessFactor;
	float metallic = texture(metallicRoughnessTex[nonuniformEXT(faceGroupIndex)],  inUv).b * mat.metallicFactor;
	vec3 normal = normalize(inNormal);

	outAlbedo = vec4(albedo, 1.0);
	outNormal = packViewNormal(normal);
	outMaterial = vec4(
		clamp(roughness, 0.0, 1.0),
		clamp(metallic, 0.0, 1.0),
		0.0,
		0.0);

	outPreview = vec4(albedo, 1.0);
}
