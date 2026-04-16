#version 450

#extension GL_EXT_nonuniform_qualifier : require

#include "../utility/geometry.glslh"

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUv;

layout(set = 0, binding = 1) uniform sampler2D diffuseTex[];
layout(set = 0, binding = 2) uniform sampler2D specularTex[];
layout(set = 0, binding = 3) uniform sampler2D normalTex[];
layout(set = 0, binding = 4) uniform sampler2D roughnessTex[];
layout(set = 0, binding = 5) uniform sampler2D metallicTex[];
layout(set = 0, binding = 6) uniform sampler2D emissiveTex[];
layout(set = 0, binding = 7) readonly buffer FaceGroupIndices
{
    uint values[];
} faceGroupIndices;

layout(location = 0) out vec4 outAlbedo;
layout(location = 1) out vec2 outNormal;
layout(location = 2) out vec4 outMaterial;
layout(location = 3) out vec4 outPreview;

void main()
{
	uint faceGroupIndex = faceGroupIndices.values[gl_PrimitiveID];

	vec3 albedo = texture(diffuseTex[nonuniformEXT(faceGroupIndex)], inUv).rgb;
	vec3 specular = texture(specularTex[nonuniformEXT(faceGroupIndex)], inUv).rgb;
	vec3 emissive = texture(emissiveTex[nonuniformEXT(faceGroupIndex)], inUv).rgb;
	float roughness = texture(roughnessTex[nonuniformEXT(faceGroupIndex)], inUv).r;
	float metallic = texture(metallicTex[nonuniformEXT(faceGroupIndex)], inUv).r;
	vec3 normal = normalize(inNormal);

	outAlbedo = vec4(albedo, 1.0);
	outNormal = packViewNormal(normal);
	outMaterial = vec4(
		clamp(roughness, 0.0, 1.0),
		clamp(metallic, 0.0, 1.0),
		clamp(max(max(specular.r, specular.g), specular.b), 0.0, 1.0),
		clamp(max(max(emissive.r, emissive.g), emissive.b), 0.0, 1.0));

	outPreview = vec4(albedo, 1.0);
}
