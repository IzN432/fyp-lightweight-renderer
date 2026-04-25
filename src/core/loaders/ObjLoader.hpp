#pragma once

#include "core/scene/Mesh.hpp"
#include "core/loaders/Material.hpp"

#include <filesystem>

namespace lr
{
struct ObjMeshLoadResult
{
	Mesh mesh;
	MeshLayout layout;
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
	std::string metallicTextureName = "metallicTexture";
	std::string roughnessTextureName = "roughnessTexture";
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
