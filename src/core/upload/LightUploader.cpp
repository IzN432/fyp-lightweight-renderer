#include "core/upload/LightUploader.hpp"

#include "core/scene/TransformComponent.hpp"
#include "core/scene/SceneObject.hpp"

#include <stdexcept>
#include <limits>
#include <string>

namespace lr
{

LightUploader::LightUploader(ResourceRegistry &registry, const std::string name, uint32_t maxLights)
    : m_registry(registry), m_maxLights(maxLights)
{
    m_bufferName = name + "_lb";
    registry.registerDynamicBuffer(m_bufferName, sizeof(LightGpuData) * m_maxLights,
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
}

void LightUploader::upload(std::vector<SceneObject *> &lights)
{
    if (lights.size() > m_maxLights)
    {
        throw std::length_error("LightUploader: " + std::to_string(lights.size()) + " lights, but the light buffer holds " +
                                std::to_string(m_maxLights));
    }
    std::vector<LightGpuData> data;
    data.reserve(lights.size());
    uint32_t nextSpotShadow = 0;
    uint32_t nextPointShadow = 0;
    uint32_t nextDirectionalShadow = 0;
    uint32_t nextAreaShadow = 0;

    for (const auto &lightObject : lights)
    {
        Light &light = lightObject->getComponent<Light>();

        LightGpuData &gpuData = data.emplace_back();

        std::visit(
            [&gpuData, &lightObject, &nextSpotShadow, &nextPointShadow, &nextDirectionalShadow, &nextAreaShadow](auto &&l) {
                using T = std::decay_t<decltype(l)>;

                gpuData.color          = l.color;
                gpuData.position       = glm::vec3(0.0f);                   // default for directional and area lights
                gpuData.rotation       = glm::quat(1.0f, 0.0f, 0.0f, 0.0f); // default for directional and area lights
                gpuData.intensity      = l.intensity;
                gpuData.innerConeAngle = 0.0f;            // default for point and directional
                gpuData.outerConeAngle = 0.0f;            // default for point and directional
                gpuData.areaSize       = glm::vec2(0.0f); // default for point and directional
                gpuData.flags          = 0;
                gpuData.range           = 0.0f;
                gpuData.shadowNearPlane = 0.0f;
                gpuData.shadowIndex     = std::numeric_limits<uint32_t>::max();

                if constexpr (std::is_same_v<T, PointLight>)
                {
                    TransformComponent &transform = lightObject->getComponent<TransformComponent>();
                    gpuData.position     = transform.transform().position();
                    gpuData.type         = 0;
                    gpuData.range           = l.range;
                    gpuData.shadowNearPlane = l.shadowNearPlane;
                    gpuData.shadowIndex     = nextPointShadow++;
                } else if constexpr (std::is_same_v<T, SpotLight>)
                {
                    TransformComponent &transform   = lightObject->getComponent<TransformComponent>();
                    gpuData.position       = transform.transform().position();
                    gpuData.rotation       = transform.transform().rotation();
                    gpuData.type           = 1;
                    gpuData.innerConeAngle = glm::radians(l.innerConeAngleDegrees);
                    gpuData.outerConeAngle = glm::radians(l.outerConeAngleDegrees);
                    gpuData.range           = l.range;
                    gpuData.shadowNearPlane = l.shadowNearPlane;
                    gpuData.shadowIndex     = nextSpotShadow++;
                } else if constexpr (std::is_same_v<T, AreaLight>)
                {
                    TransformComponent &transform = lightObject->getComponent<TransformComponent>();
                    gpuData.position     = transform.transform().position();
                    gpuData.rotation     = transform.transform().rotation();
                    gpuData.type         = 2;
                    gpuData.areaSize     = l.size;
                    gpuData.outerConeAngle = glm::radians(l.spreadAngleDegrees);
                    gpuData.flags        = l.twoSided ? kLightFlagTwoSided : 0u;
                    gpuData.shadowIndex  = nextAreaShadow++;
                } else if constexpr (std::is_same_v<T, DirectionalLight>)
                {
                    TransformComponent &transform = lightObject->getComponent<TransformComponent>();
                    gpuData.rotation     = transform.transform().rotation();
                    gpuData.type         = 3;
                    gpuData.shadowIndex  = nextDirectionalShadow++;
                } else if constexpr (std::is_same_v<T, ImageLight>)
                {
                    gpuData.type = 4;
                }
            },
            light.light);
    }

    m_numLights = static_cast<uint32_t>(lights.size());
    m_registry.updateBuffer(m_bufferName, data.data(), sizeof(LightGpuData) * data.size());
}

} // namespace lr
