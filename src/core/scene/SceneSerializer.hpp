#pragma once

#include <filesystem>
#include <memory>

namespace lr
{

class SceneAssets;

// Versioned, CPU-only scene persistence. Checkpoint 1 stores hierarchy, transforms, cameras and
// lights in <path>/scene.json. Asset-bearing components are deliberately rejected until the binary
// asset portion of the format is implemented.
class SceneSerializer
{
public:
    static void                         save(const SceneAssets &assets, const std::filesystem::path &path);
    static std::unique_ptr<SceneAssets> load(const std::filesystem::path &path);
};

} // namespace lr
