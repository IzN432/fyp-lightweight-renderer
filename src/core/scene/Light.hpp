#pragma once

#include "core/scene/Transform.hpp"
#include "core/scene/Component.hpp"

#include <variant>

namespace lr
{

struct BaseLight
{
    glm::vec3 color{1.0f, 1.0f, 1.0f};
    float intensity = 1.0f;
};

struct PointLight : public BaseLight
{
};

struct SpotLight : public BaseLight
{
    float innerConeAngleDegrees = 15.0f;
    float outerConeAngleDegrees = 30.0f;
};

struct AreaLight : public BaseLight
{
    glm::vec2 size{1.0f, 1.0f}; // width and height in world units
};

struct DirectionalLight : public BaseLight
{
};

struct ImageLight : public BaseLight
{
};

using LightVariant = std::variant<PointLight, SpotLight, AreaLight, DirectionalLight, ImageLight>;

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

    bool operator()(ImageLight &light) const
    {
        bool changed = false;
        changed |= ImGui::SliderFloat("Light Intensity", &light.intensity, 0.0f, 1.0f);
        changed |= ImGui::ColorEdit3("Light Color", &light.color.x);
        return changed;
    }
};

struct Light : public Component
{
    LightVariant light;
    
    explicit Light(const LightVariant &lightVariant)
        : light(lightVariant), Component("Light") {}

    void onGUIImpl() override
    {
        if (std::visit(LightGUICallbacks{}, light))
        {
            notifyChanged();
        }
    }
    
};

} // namespace lr