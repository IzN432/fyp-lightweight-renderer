#pragma once

#include "core/scene/Transform.hpp"
#include <variant>

namespace lr
{

struct Light
{
    Transform transform{};
    glm::vec3 color{1.0f, 1.0f, 1.0f};
    float intensity = 1.0f;
};

struct PointLight : public Light
{
};

struct SpotLight : public Light
{
    float innerConeAngleDegrees = 15.0f;
    float outerConeAngleDegrees = 30.0f;
};

struct AreaLight : public Light
{
    glm::vec2 size{1.0f, 1.0f}; // width and height in world units
};

struct DirectionalLight : public Light
{
};

using LightVariant = std::variant<PointLight, SpotLight, AreaLight, DirectionalLight>;

} // namespace lr