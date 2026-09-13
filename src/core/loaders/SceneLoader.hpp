#pragma once

#include "GltfLoader.hpp"
#include "ObjLoader.hpp"
#include "core/scene/MeshStore.hpp"
#include "core/scene/SceneObject.hpp"
#include "features/linear_blend_skinning/Skin.hpp"

#include <filesystem>
#include <optional>
#include <vector>

namespace lr
{

class SceneManager;

struct SceneLoaderConfig
{
    GltfLoaderConfig gltf;
    ObjLoaderConfig  obj;
};

struct SceneLoadResult
{
    SceneObjectId rootObject;
    // Parallel to MeshLoadResult::nodes. Nodes outside the selected source
    // scene are not instantiated and therefore have no SceneObjectId.
    std::vector<std::optional<SceneObjectId>> nodeObjects;
    std::vector<MeshHandle>                   meshHandles;
    std::vector<Skin>                         skins;
    std::optional<SceneObjectId>              firstMeshObject;
};

// Selects a format loader from the filename and instantiates its common
// MeshLoadResult directly into a scene managed by SceneManager.
class SceneLoader
{
public:
    static SceneLoadResult load(const std::filesystem::path &path, SceneManager &sceneManager,
                                const SceneLoaderConfig &config = {});
};

} // namespace lr
