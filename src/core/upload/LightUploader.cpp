#include "core/upload/LightUploader.hpp"

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

void LightUploader::upload(std::vector<LightVariant> &lights)
{
    std::vector<LightGpuData> data;    
    data.reserve(lights.size());

    for (const auto &light : lights)
    {
        LightGpuData &gpuData = data.emplace_back();
        
        std::visit([&gpuData](auto &&l)
        {
            using T = std::decay_t<decltype(l)>;

            gpuData.position = l.transform.position;
            gpuData.rotation = l.transform.rotation;
            gpuData.color = l.color;
            gpuData.intensity = l.intensity;
            gpuData.innerConeAngle = 0.0f; // default for point and directional
            gpuData.outerConeAngle = 0.0f; // default for point and directional
            gpuData.areaSize = glm::vec2(0.0f); // default for point and directional
            
            if constexpr (std::is_same_v<T, PointLight>)
            {
                gpuData.type = 0;
            }
            else if constexpr (std::is_same_v<T, SpotLight>)
            {
                gpuData.type = 1;
                gpuData.innerConeAngle = glm::radians(l.innerConeAngleDegrees);
                gpuData.outerConeAngle = glm::radians(l.outerConeAngleDegrees);
            }
            else if constexpr (std::is_same_v<T, AreaLight>)
            {
                gpuData.type = 2;
                gpuData.areaSize = l.size;
            }
            else if constexpr (std::is_same_v<T, DirectionalLight>)
            {
                gpuData.type = 3;
            }
        }, light);
    }

    m_numLights = static_cast<uint32_t>(lights.size());
    m_registry.updateBuffer(m_bufferName, data.data(), sizeof(LightGpuData) * data.size());
}

}  // namespace lr