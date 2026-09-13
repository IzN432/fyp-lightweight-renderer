#pragma once

#include "MaterialStore.hpp"
#include "MeshSequence.hpp"
#include "core/scene/Transform.hpp"
#include "features/linear_blend_skinning/Skin.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace lr
{

// A format-independent imported scene node. Node indices are stable within a
// MeshLoadResult and all hierarchy references index the same nodes vector.
struct MeshNode
{
    std::string             name;
    Transform               localTransform;
    std::optional<uint32_t> parent;
    std::vector<uint32_t>   children;
    std::optional<uint32_t> meshIndex;
    std::optional<uint32_t> skinIndex;
};

// Common output of scene-capable mesh loaders. Formats without hierarchy or
// skinning still produce the same shape: a single root node and no skins.
struct MeshLoadResult
{
    MeshSequence                sequence;
    std::vector<MaterialHandle> materialHandles;
    std::vector<Skin>           skins;
    std::vector<MeshNode>       nodes;
    std::vector<uint32_t>       sceneRoots;
};

} // namespace lr
