#pragma once

#include "core/scene/Mesh.hpp"
#include "core/loaders/Material.hpp"

#include <filesystem>

namespace lr
{
struct ObjMeshLoadResult
{
	Mesh mesh;
	std::vector<Material> materials;
};

struct ObjLoaderConfig
{
	// Per-vertex attributes
	std::string normalAttributeName = "normal";
	std::string tangentAttributeName = "tangent";
	std::string uvAttributeName = "uv";

	// Material
	std::string diffuseTextureName = "diffuseTexture";
	std::string ambientTextureName = "ambientTexture";
	std::string specularTextureName = "specularTexture";
	std::string normalTextureName = "normalTexture";
	// OBJ/MTL's PBR extension stores roughness and metallic as two separate grayscale images
	// (roughness_texname/metallic_texname), unlike glTF's single combined texture — they're
	// packed into one image (G=roughness, B=metallic, matching GltfLoaderConfig's convention)
	// under this one key so OBJ- and glTF-sourced materials can share one GpuMaterialLayout.
	std::string metallicRoughnessTextureName = "metallicRoughnessTexture";
	std::string emissiveTextureName = "emissiveTexture";

	std::string baseDiffuseName = "baseDiffuse";
	std::string baseAmbientName = "baseAmbient";
	std::string baseSpecularName = "baseSpecular";
	std::string shininessName = "shininess";
	std::string baseRoughnessName = "baseRoughness";
	std::string baseMetallicName = "baseMetallic";
	std::string baseEmissiveName = "baseEmissive";
};

class ObjLoader
{
public:
	ObjMeshLoadResult load(const std::filesystem::path &path, const ObjLoaderConfig &config = {}) const;
};

}  // namespace lr
