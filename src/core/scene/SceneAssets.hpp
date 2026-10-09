#pragma once

#include "EngineConventions.hpp"
#include "MeshStore.hpp"
#include "Scene.hpp"

#include "core/loaders/MaterialStore.hpp"
#include "core/loaders/SceneLoader.hpp"
#include "features/animation/AnimationLibrary.hpp"

#include <filesystem>

namespace lr
{

// A scene together with the asset stores its components reference: what the loaders fill, and what
// GPU-sync code (SceneManager today) packs into the buffers the engine's passes read. CPU-only — no
// Vulkan objects — so it can be built and inspected without a Viewer.
//
// Scene objects and MeshComponents hold pointers into these members, so a SceneAssets never moves.
class SceneAssets
{
public:
    SceneAssets() : materials(conventions::materialCapacity, conventions::defaultMaterial) {}

    SceneAssets(const SceneAssets &)            = delete;
    SceneAssets &operator=(const SceneAssets &) = delete;
    SceneAssets(SceneAssets &&)                 = delete;
    SceneAssets &operator=(SceneAssets &&)      = delete;

    // Loads OBJ, glTF or GLB content into the scene and stores, under a new root object (see
    // SceneLoader). Meshes and materials use the names in EngineConventions.hpp.
    SceneLoadResult load(const std::filesystem::path &path,
                         const SceneLoaderConfig     &config = conventions::sceneLoaderConfig())
    {
        return SceneLoader::load(path, scene, meshes, materials, animations, config);
    }

    Scene         scene;
    MeshStore     meshes;
    MaterialStore materials;
    AnimationLibrary animations;
};

} // namespace lr
