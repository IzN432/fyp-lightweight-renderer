#pragma once

#include "core/scene/Mesh.hpp"

#include <glm/glm.hpp>

#include <filesystem>
#include <string>
#include <vector>

namespace lr
{

struct MaterialTextures
{
	std::filesystem::path baseColor;
	std::filesystem::path specular;
	std::filesystem::path normal;
	std::filesystem::path roughness;
	std::filesystem::path metallic;
	std::filesystem::path emissive;
};

struct MaterialInfo
{
	std::string name;

	glm::vec3 baseColorFactor{1.0f, 1.0f, 1.0f};
	glm::vec3 emissiveFactor{0.0f, 0.0f, 0.0f};
	float     roughnessFactor = 1.0f;
	float     metallicFactor  = 0.0f;

	MaterialTextures textures;
};

// Unified loader result: static meshes are represented as one frame.
struct MeshSequence
{
	std::vector<Mesh> frames;
	float             frameRate = 0.0f;  // 0 => not time-sampled / unknown

	// Material table from source asset (OBJ/MTL/etc).
	std::vector<MaterialInfo> materials;

	// Unique texture files referenced by materials, resolved against source file directory.
	std::vector<std::filesystem::path> textureFiles;

	bool isAnimated() const { return frames.size() > 1; }
	bool empty() const { return frames.empty(); }
};

// Base loader interface. Concrete loaders (OBJ, Alembic, etc.)
// return a unified MeshSequence result.
class Loader
{
public:
	virtual ~Loader() = default;

	virtual MeshSequence load(const std::filesystem::path &path,
							 const MeshLayout &layout) const = 0;
};

}  // namespace lr
