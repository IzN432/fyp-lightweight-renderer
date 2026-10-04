#pragma once

#include "core/framegraph/ResourceRegistry.hpp"

#include "core/scene/Light.hpp"

namespace lr
{

struct alignas(16) LightGpuData
{
    glm::vec3 position; // 12 bytes
    // 0 = point, 1 = spot, 2 = area, 3 = directional
    uint32_t  type;     // 4 bytes
    glm::quat rotation; // 16 bytes

    glm::vec3 color;     // 12 bytes
    float     intensity; // 4 bytes

    float     innerConeAngle; // 4 bytes (spot lights)
    float     outerConeAngle; // 4 bytes (spot lights)
    glm::vec2 areaSize;       // 8 bytes (area lights)

    uint32_t flags; // 4 bytes: bit 0 = two-sided (area lights); padded to 16 by alignas
};

inline constexpr uint32_t kLightFlagTwoSided = 1u;

static_assert(sizeof(LightGpuData) == 80, "LightGpuData size mismatch!");

class LightUploader
{
public:
    explicit LightUploader(ResourceRegistry &registry, const std::string name = "lights", uint32_t maxLights = 16);

    void upload(std::vector<SceneObject *> &lights);

    const std::string &bufferName() const { return m_bufferName; }
    const uint32_t    &numLights() const { return m_numLights; }
    uint32_t           maxLights() const { return m_maxLights; }

private:
    ResourceRegistry &m_registry;
    std::string       m_bufferName;
    uint32_t          m_maxLights;
    uint32_t          m_numLights = 0;
};
} // namespace lr