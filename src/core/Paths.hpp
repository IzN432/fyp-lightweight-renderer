#pragma once
#include <filesystem>

namespace lr
{
namespace paths
{
// Resolved once on first use, not baked in at compile time: a shipped executable is run from a
// directory that has nothing to do with the machine it was built on, so the build tree's paths
// would be dangling. Both look next to the executable first (the shipping layout) and fall back
// to the build-time location so a developer can run straight out of the build tree.
const std::filesystem::path &shaderDirectory();
const std::filesystem::path &assetDirectory();
} // namespace paths
} // namespace lr
