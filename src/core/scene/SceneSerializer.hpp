#pragma once

#include <filesystem>
#include <memory>

namespace lr
{

class SceneAssets;
class Scene;
class MeshStore;
class MaterialStore;

// Versioned, CPU-only scene persistence. A single .lrscene file contains its JSON manifest followed
// by the bulk mesh/texture payload. Runtime object, mesh and material handles are remapped to file-local IDs.
class SceneSerializer
{
public:
    static void                         save(const SceneAssets &assets, const std::filesystem::path &path);
    static void                         save(const Scene &scene, const MeshStore &meshes,
                                             const MaterialStore &materials,
                                             const std::filesystem::path &path);
    static std::unique_ptr<SceneAssets> load(const std::filesystem::path &path);
    // Appends a saved scene to caller-owned stores. Existing objects and assets are preserved.
    static void load(const std::filesystem::path &path, Scene &scene, MeshStore &meshes,
                     MaterialStore &materials);
};

} // namespace lr
