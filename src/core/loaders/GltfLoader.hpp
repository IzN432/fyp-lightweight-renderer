#pragma once

#include <vector>

#include "MeshSequence.hpp"
#include "Material.hpp"
#include "MaterialStore.hpp"
#include "core/scene/Transform.hpp"
#include "features/linear_blend_skinning/Skin.hpp"

#include <cstdint>
#include <optional>

namespace lr
{

struct GltfMeshInstance
{
    uint32_t                sourceNodeIndex = 0;
    uint32_t                meshIndex       = 0;
    std::optional<uint32_t> skinIndex;
    Transform               localTransform;
};

struct GltfMeshLoadResult
{
    MeshSequence sequence;
    // Parallel to the glTF material indices (index 0 = the synthetic default material) — already
    // resolved to their MaterialStore slots, matching what got baked into each Mesh's faceGroups.
    std::vector<MaterialHandle> materialHandles;
    // Parallel to glTF skins. Joint order is preserved exactly so mesh JOINTS_n values index
    // directly into Skin::joints() and Skin::jointMatrices().
    std::vector<Skin> skins;
    // One entry per glTF node that instantiates a mesh. This preserves mesh-to-skin association,
    // which belongs to nodes in glTF rather than to mesh definitions.
    std::vector<GltfMeshInstance> meshInstances;
};

struct GltfLoaderConfig
{
    // Per-vertex attributes
    std::string normalAttributeName  = "normal";
    std::string tangentAttributeName = "tangent";
    std::string uvAttributeName      = "uv";

    // Material
    std::string diffuseTextureName           = "diffuseTexture";
    std::string normalTextureName            = "normalTexture";
    std::string metallicRoughnessTextureName = "metallicRoughnessTexture";
    std::string emissiveTextureName          = "emissiveTexture";

    std::string baseDiffuseName   = "baseDiffuse";
    std::string baseRoughnessName = "baseRoughness";
    std::string baseMetallicName  = "baseMetallic";
    std::string baseEmissiveName  = "baseEmissive";
};

/**
 * Loads glTF/GLB files into the internal Mesh format.
 *
 * Per-vertex attributes (names from config):
 *   - normalAttr  (vec3)  vertex normal
 *   - uvAttr      (vec2)  primary texture coordinate
 *   - tangentAttr (vec3)  vertex tangent
 *
 * Skinning attributes:
 *   - JOINTS_n/WEIGHTS_n are combined into arbitrary-length sparse vertex groups.
 *     Group indices remain local indices into the glTF skin's joints array.
 *
 * Material scalars:
 *   - baseDiffuse    (vec4)   from baseColorFactor
 *   - baseRoughness  (float)  from roughnessFactor
 *   - baseMetallic   (float)  from metallicFactor
 *   - baseEmissive   (vec3)   from emissiveFactor
 *
 * Material textures:
 *   - diffuseTexture           from baseColorTexture
 *   - metallicRoughnessTexture from metallicRoughnessTexture (B=metallic, G=roughness)
 *   - emissiveTexture          from emissiveTexture
 */
class GltfLoader
{
public:
    // Materials are registered into `materialStore` as they're parsed, so the returned Mesh's
    // faceGroups already hold global MaterialHandles — no remapping needed at the call site.
    GltfMeshLoadResult load(const std::filesystem::path &path, MaterialStore &materialStore,
                            const GltfLoaderConfig &config = {}) const;
};

} // namespace lr
