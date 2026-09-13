#pragma once

#include "core/loaders/MeshLoadResult.hpp"
#include "core/loaders/Material.hpp"
#include "core/loaders/MaterialStore.hpp"

#include <filesystem>

namespace lr
{
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
    static MeshLoadResult load(const std::filesystem::path &path, MaterialStore &materialStore,
                               const ObjLoaderConfig &config = {});
};

} // namespace lr
