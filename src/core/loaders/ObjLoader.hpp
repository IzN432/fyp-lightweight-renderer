#pragma once

#include "core/scene/Mesh.hpp"
#include "core/loaders/Material.hpp"
#include "core/loaders/MaterialStore.hpp"

#include <filesystem>

namespace lr
{
struct ObjMeshLoadResult
{
    Mesh mesh;
    // Parallel to the OBJ's material indices (index 0 = the synthetic default material) — already
    // resolved to their MaterialStore slots, matching what got baked into the Mesh's faceGroups.
    std::vector<MaterialHandle> materialHandles;
};

struct ObjLoaderConfig
{
    // Per-vertex attributes
    std::string normalAttributeName  = "normal";
    std::string tangentAttributeName = "tangent";
    std::string uvAttributeName      = "uv";

    // Material
    std::string diffuseTextureName   = "diffuseTexture";
    std::string ambientTextureName   = "ambientTexture";
    std::string specularTextureName  = "specularTexture";
    std::string normalTextureName    = "normalTexture";
    std::string metallicTextureName  = "metallicTexture";
    std::string roughnessTextureName = "roughnessTexture";
    std::string emissiveTextureName  = "emissiveTexture";

    std::string baseDiffuseName   = "baseDiffuse";
    std::string baseAmbientName   = "baseAmbient";
    std::string baseSpecularName  = "baseSpecular";
    std::string shininessName     = "shininess";
    std::string baseRoughnessName = "baseRoughness";
    std::string baseMetallicName  = "baseMetallic";
    std::string baseEmissiveName  = "baseEmissive";
};

class ObjLoader
{
public:
    // Materials are registered into `materialStore` as they're parsed, so the returned Mesh's
    // faceGroups already hold global MaterialHandles — no remapping needed at the call site.
    ObjMeshLoadResult load(const std::filesystem::path &path, MaterialStore &materialStore,
                           const ObjLoaderConfig &config = {}) const;
};

} // namespace lr
