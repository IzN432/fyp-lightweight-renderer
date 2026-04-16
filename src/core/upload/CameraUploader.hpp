#pragma once

#include "core/scene/Camera.hpp"
#include "core/framegraph/ResourceRegistry.hpp"

#include <glm/glm.hpp>

namespace lr
{

// GPU-side layout — must match the GLSL uniform block exactly (std140).
// vec3 is padded to 16 bytes in std140, so position is stored as vec4.
struct CameraGpuData
{
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 viewProj;
    glm::vec4 position;  // .xyz = world position; .w unused
};

static_assert(sizeof(CameraGpuData) == 3 * 64 + 16,
              "CameraGpuData layout does not match expected std140 size");

class CameraUploader
{
public:
    static constexpr const char *kBufferName = "camera";

    explicit CameraUploader(ResourceRegistry &registry);

    void upload(const Camera &camera, float aspectRatio);

private:
    ResourceRegistry &m_registry;
};

}  // namespace lr
