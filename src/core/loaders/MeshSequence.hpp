#pragma once

#include "core/scene/Mesh.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace lr
{

// Decoded image data, always RGBA8 (4 bytes per pixel).
// Empty when a texture is absent or failed to load.


// Unified loader result: static meshes are represented as one frame.
struct MeshSequence
{
    // Disable copying but allow moving, since MeshSequences can contain large meshes and materials.
    MeshSequence() = default;
    MeshSequence(const MeshSequence&) = delete;
    MeshSequence& operator=(const MeshSequence&) = delete;
    MeshSequence(MeshSequence&&) = default;
    MeshSequence& operator=(MeshSequence&&) = default;

    std::vector<Mesh>         frames;
    float                     frameRate = 0.0f;  // 0 => not time-sampled / unknown

    bool isAnimated() const { return frames.size() > 1; }
    bool empty()      const { return frames.empty(); }
};

}  // namespace lr
