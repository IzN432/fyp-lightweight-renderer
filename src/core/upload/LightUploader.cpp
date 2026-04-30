#include "core/upload/LightUploader.hpp"

#include "core/scene/Transform.hpp"
#include "core/scene/SceneObject.hpp"

namespace lr
{

LightUploader::LightUploader(ResourceRegistry &registry, const std::string name, uint32_t maxLights)
    : m_registry(registry), m_maxLights(maxLights)
{
    m_bufferName = name + "_lb";
    registry.registerDynamicBuffer(m_bufferName,
                                    sizeof(LightGpuData) * m_maxLights,
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
}

void LightUploader::upload(std::vector<SceneObject*> &lights)
{
    std::vector<LightGpuData> data;    
    data.reserve(lights.size());
    
    for (const auto &lightObject : lights)
    {
        Light &light = lightObject->getComponent<Light>();

        LightGpuData &gpuData = data.emplace_back();
        
        std::visit([&gpuData, &lightObject](auto &&l)
        {
            using T = std::decay_t<decltype(l)>;

            gpuData.color = l.color;
            gpuData.position = glm::vec3(0.0f); // default for directional and area lights
            gpuData.rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f); // default for directional and area lights
            gpuData.intensity = l.intensity;
            gpuData.innerConeAngle = 0.0f; // default for point and directional
            gpuData.outerConeAngle = 0.0f; // default for point and directional
            gpuData.areaSize = glm::vec2(0.0f); // default for point and directional
            
            if constexpr (std::is_same_v<T, PointLight>)
            {
                Transform &transform = lightObject->getComponent<Transform>();
                gpuData.position = transform.position;
                gpuData.type = 0;
            }
            else if constexpr (std::is_same_v<T, SpotLight>)
            {
                Transform &transform = lightObject->getComponent<Transform>();
                gpuData.position = transform.position;
                gpuData.rotation = transform.rotation();
                gpuData.type = 1;
                gpuData.innerConeAngle = glm::radians(l.innerConeAngleDegrees);
                gpuData.outerConeAngle = glm::radians(l.outerConeAngleDegrees);
            }
            else if constexpr (std::is_same_v<T, AreaLight>)
            {
                Transform &transform = lightObject->getComponent<Transform>();
                gpuData.position = transform.position;
                gpuData.rotation = transform.rotation();
                gpuData.type = 2;
                gpuData.areaSize = l.size;
            }
            else if constexpr (std::is_same_v<T, DirectionalLight>)
            {
                Transform &transform = lightObject->getComponent<Transform>();
                gpuData.rotation = transform.rotation();
                gpuData.type = 3;
            }
            else if constexpr (std::is_same_v<T, ImageLight>)
            {
                gpuData.type = 4;
            }
        }, light.light);
    }

    m_numLights = static_cast<uint32_t>(lights.size());
    m_registry.updateBuffer(m_bufferName, data.data(), sizeof(LightGpuData) * data.size());
}

}  // namespace lr