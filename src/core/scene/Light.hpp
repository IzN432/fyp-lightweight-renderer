#pragma once

#include "core/scene/Transform.hpp"
#include "core/scene/Component.hpp"

#include <variant>

namespace lr
{

struct Light
{
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

struct LightGUICallbacks
{
    bool operator()(PointLight &light) const
    {
        bool changed = false;
        changed |= ImGui::SliderFloat("Light Intensity", &light.intensity, 0.0f, 1.0f);
        changed |= ImGui::ColorEdit3("Light Color", &light.color.x);
        return changed;
    }

    bool operator()(DirectionalLight &light) const
    {
        bool changed = false;
        changed |= ImGui::SliderFloat("Light Intensity", &light.intensity, 0.0f, 1.0f);
        changed |= ImGui::ColorEdit3("Light Color", &light.color.x);
        return changed;
    }

    bool operator()(SpotLight &light) const
    {
        bool changed = false;
        changed |= ImGui::SliderFloat("Light Intensity", &light.intensity, 0.0f, 1.0f);
        changed |= ImGui::ColorEdit3("Light Color", &light.color.x);
        changed |= ImGui::SliderFloat("Inner Cone Angle", &light.innerConeAngleDegrees, 0.0f, 90.0f);
        changed |= ImGui::SliderFloat("Outer Cone Angle", &light.outerConeAngleDegrees, 0.0f, 90.0f);
        return changed;
    }

    bool operator()(AreaLight &light) const
    {
        bool changed = false;
        changed |= ImGui::SliderFloat("Light Intensity", &light.intensity, 0.0f, 1.0f);
        changed |= ImGui::ColorEdit3("Light Color", &light.color.x);
        changed |= ImGui::DragFloat2("Size", &light.size.x, 0.1f);
        return changed;
    }
};

struct EditorLight : public Component
{
    LightVariant light;
    
    explicit EditorLight(const LightVariant &lightVariant)
        : light(lightVariant), Component("Light") {}

    bool onGUIImpl() override
    {
        return std::visit([&](auto &&l) -> bool {
            return LightGUICallbacks{}(l);
        }, light);
    }
    
};

} // namespace lr