#include "core/upload/CameraUploader.hpp"

#include <vulkan/vulkan.h>

namespace lr
{

CameraUploader::CameraUploader(ResourceRegistry &registry)
    : m_registry(registry)
{
    registry.registerDynamicBuffer(kBufferName,
                                    sizeof(CameraGpuData),
                                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
}

void CameraUploader::upload(const Camera &camera, float aspectRatio)
{
    CameraGpuData data{};
    data.view     = camera.viewMatrix();
    data.proj     = camera.projectionMatrix(aspectRatio);
    data.viewProj = data.proj * data.view;
    data.position = glm::vec4(camera.transform.position, 0.0f);
    m_registry.updateBuffer(kBufferName, &data, sizeof(data));
}

}  // namespace lr
