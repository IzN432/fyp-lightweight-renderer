#include "core/upload/CameraUploader.hpp"

#include <vulkan/vulkan.h>

namespace lr
{

CameraUploader::CameraUploader(ResourceRegistry &registry, const std::string name)
    : m_registry(registry)
{
    m_bufferName = name + "_cb";
    registry.registerDynamicBuffer(m_bufferName,
                                    sizeof(CameraGpuData),
                                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
}

void CameraUploader::upload(const Camera &camera, float aspectRatio)
{
    CameraGpuData data{};
    data.view = camera.viewMatrix();
    data.proj = camera.projectionMatrix(aspectRatio);
    data.viewProj = data.proj * data.view;
    data.invView = glm::inverse(data.view);
    data.invProj = glm::inverse(data.proj);
    data.position = glm::vec4(camera.transform.position, 0.0f);
    m_registry.updateBuffer(m_bufferName, &data, sizeof(data));
}

}  // namespace lr
