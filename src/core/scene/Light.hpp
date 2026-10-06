#pragma once

#include "core/scene/TransformComponent.hpp"
#include "core/scene/Component.hpp"

#include <algorithm>
#include <variant>

namespace lr
{

struct BaseLight
{
    glm::vec3 color{1.0f, 1.0f, 1.0f};
    float     intensity = 1.0f;
};

struct PointLight : public BaseLight
{};

struct SpotLight : public BaseLight
{
    float innerConeAngleDegrees = 15.0f;
    float outerConeAngleDegrees = 30.0f;
};

struct AreaLight : public BaseLight
{
    glm::vec2 size{1.0f, 1.0f}; // width and height in world units
    // Emits from (and is visible from) both faces; otherwise only along its forward axis (local -Z).
    bool twoSided = true;
};

struct DirectionalLight : public BaseLight
{};

struct ImageLight : public BaseLight
{};

static const char *lightTypeNames[] = {"Point", "Spot", "Area", "Directional", "Image"};

using LightVariant = std::variant<PointLight, SpotLight, AreaLight, DirectionalLight, ImageLight>;

struct LightGUICallbacks
{
    bool operator()(PointLight &light) const
    {
        bool changed = false;
        changed |= ImGui::SliderFloat("Light Intensity", &light.intensity, 0.0f, 100.0f);
        changed |= ImGui::ColorEdit3("Light Color", &light.color.x);
        return changed;
    }

    bool operator()(DirectionalLight &light) const
    {
        bool changed = false;
        changed |= ImGui::SliderFloat("Light Intensity", &light.intensity, 0.0f, 100.0f);
        changed |= ImGui::ColorEdit3("Light Color", &light.color.x);
        return changed;
    }

    bool operator()(SpotLight &light) const
    {
        bool changed = false;
        changed |= ImGui::SliderFloat("Light Intensity", &light.intensity, 0.0f, 100.0f);
        changed |= ImGui::ColorEdit3("Light Color", &light.color.x);
        const bool innerChanged = ImGui::SliderFloat("Inner Cone Angle", &light.innerConeAngleDegrees, 0.0f, 90.0f);
        if (innerChanged && light.innerConeAngleDegrees > light.outerConeAngleDegrees)
            light.outerConeAngleDegrees = light.innerConeAngleDegrees;
        const bool outerChanged = ImGui::SliderFloat("Outer Cone Angle", &light.outerConeAngleDegrees, 0.0f, 90.0f);
        if (outerChanged && light.outerConeAngleDegrees < light.innerConeAngleDegrees)
            light.innerConeAngleDegrees = light.outerConeAngleDegrees;
        changed |= innerChanged || outerChanged;
        return changed;
    }

    bool operator()(AreaLight &light) const
    {
        bool changed = false;
        changed |= ImGui::SliderFloat("Light Intensity", &light.intensity, 0.0f, 100.0f);
        changed |= ImGui::ColorEdit3("Light Color", &light.color.x);
        changed |= ImGui::DragFloat2("Size", &light.size.x, 0.1f);
        changed |= ImGui::Checkbox("Two-Sided", &light.twoSided);
        return changed;
    }

    bool operator()(ImageLight &light) const
    {
        bool changed = false;
        changed |= ImGui::SliderFloat("Light Intensity", &light.intensity, 0.0f, 100.0f);
        changed |= ImGui::ColorEdit3("Light Color", &light.color.x);
        return changed;
    }
};

struct Light : public Component
{
    LightVariant light;

    explicit Light(const LightVariant &lightVariant) : light(lightVariant), Component("Light")
    {
        enforceConeAngles(light);
    }

    // Replaces the light's parameters and flags it dirty, so SceneGpu re-uploads it (see flushDirty).
    void set(const LightVariant &lightVariant)
    {
        light = lightVariant;
        enforceConeAngles(light);
        markDirty();
    }

    void onGUIImpl() override
    {
        bool changed = false;

        int currentType = static_cast<int>(light.index());

        if (ImGui::Combo("Light Type", &currentType, lightTypeNames,
                         static_cast<int>(sizeof(lightTypeNames) / sizeof(lightTypeNames[0]))))
        {
            BaseLight &baseLight = std::visit(
                [](auto &l) -> BaseLight & {
                    return static_cast<BaseLight &>(l);
                },
                light);
            switch (currentType)
            {
                case 0:
                    light = PointLight{static_cast<BaseLight>(baseLight)};
                    break;
                case 1:
                    light = SpotLight{static_cast<BaseLight>(baseLight)};
                    break;
                case 2:
                    light = AreaLight{static_cast<BaseLight>(baseLight)};
                    break;
                case 3:
                    light = DirectionalLight{static_cast<BaseLight>(baseLight)};
                    break;
                case 4:
                    light = ImageLight{static_cast<BaseLight>(baseLight)};
                    break;
            }
            changed |= true;
        }

        if (std::visit(LightGUICallbacks{}, light))
        {
            changed |= true;
        }

        if (changed)
        {
            markDirty();
        }
    }

    void onSelectGizmo(SelectionGizmoContext &context) const override;

private:
    static void enforceConeAngles(LightVariant &variant)
    {
        if (auto *spot = std::get_if<SpotLight>(&variant))
        {
            spot->outerConeAngleDegrees = std::clamp(spot->outerConeAngleDegrees, 0.0f, 90.0f);
            spot->innerConeAngleDegrees = std::clamp(spot->innerConeAngleDegrees, 0.0f,
                                                     spot->outerConeAngleDegrees);
        }
    }
};

} // namespace lr
