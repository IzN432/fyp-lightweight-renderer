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
    glm::mat4 invView;
    glm::mat4 invProj;
    glm::vec3 position;
    float padding;     // pad to 16 bytes for std140
};

struct CameraUploadResult
{
    std::string bufferName;
};

static_assert(sizeof(CameraGpuData) == 5 * 64 + 16,
              "CameraGpuData layout does not match expected std140 size");

class CameraUploader
{
public:
    explicit CameraUploader(ResourceRegistry &registry, const std::string name = "camera");

    void upload(const SceneObject &camera, float aspectRatio);

    const std::string &bufferName() const { return m_bufferName; }
private:
    ResourceRegistry &m_registry;
    std::string m_bufferName;
};

}  // namespace lr
