#pragma once

#include <filesystem>
#include <memory>

namespace lr
{

class SceneAssets;

// Versioned, CPU-only scene persistence. Metadata lives in <path>/scene.json and bulk mesh/texture
// data in <path>/assets.bin. Runtime object, mesh and material handles are remapped to file-local IDs.
class SceneSerializer
{
public:
    static void                         save(const SceneAssets &assets, const std::filesystem::path &path);
    static std::unique_ptr<SceneAssets> load(const std::filesystem::path &path);
};

} // namespace lr
