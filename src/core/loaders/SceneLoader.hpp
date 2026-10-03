#pragma once

#include "GltfLoader.hpp"
#include "ObjLoader.hpp"
#include "core/scene/Scene.hpp"
#include "core/scene/MeshStore.hpp"
#include "core/scene/SceneObject.hpp"

#include <filesystem>
#include <optional>
#include <vector>

namespace lr
{

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
    std::optional<SceneObjectId>              firstMeshObject;
    std::vector<MaterialHandle>               materialHandles;
};

// Selects a format loader from the filename and instantiates its common
// MeshLoadResult directly into a scene managed by SceneManager.
class SceneLoader
{
public:
    static SceneLoadResult load(const std::filesystem::path &path, Scene &scene, MeshStore &meshStore,
                                MaterialStore &materialStore,
                                const SceneLoaderConfig &config = {});
};

} // namespace lr
