#pragma once

#include "core/scene/TransformComponent.hpp"
#include "core/scene/Component.hpp"

#include <algorithm>
#include <memory>
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
    float range                 = 100.0f;
    float shadowNearPlane       = 0.1f;
    // Radius of the emitter in world units. Shading treats the light as a point regardless; this only
    // widens its shadow penumbra (see shadow_sampling.glslh). 0 gives the hard-edged PCF shadow.
    float sourceRadius          = 0.0f;
};

struct AreaLight : public BaseLight
{
    glm::vec2 size{1.0f, 1.0f}; // width and height in world units
    // Emits from (and is visible from) both faces; otherwise only along its forward axis (local -Z).
    bool twoSided = true;
    // Half-angle of the emitted cone on each active face. Keeping lighting and shadow projection to
    // the same spread prevents the unshadowed grazing-angle region outside the shadow map.
    float spreadAngleDegrees = 60.0f;
};

struct DirectionalLight : public BaseLight
{
    // Half the angle the emitter subtends, as seen from the scene — 0.265 for the sun. Unlike a spot
    // light's linear radius, this makes the penumbra grow with the caster-to-receiver distance.
    float angularRadiusDegrees = 0.0f;
};

struct ImageLight : public BaseLight
{};

static const char *lightTypeNames[] = {"Point", "Spot", "Area", "Directional", "Image"};

using LightVariant = std::variant<PointLight, SpotLight, AreaLight, DirectionalLight, ImageLight>;

struct LightGUICallbacks
{
    // Everything every light type has, so each overload below only adds its own parameters.
    //
    // The intensity drag speed is per-pixel and ImGui re-reads it every frame, so a purely
    // proportional speed compounds -- 1% per pixel turns a 100-pixel drag into roughly a 2.7x
    // change, which reads as a smooth sweep across an order of magnitude. The floor keeps dim lights
    // on a linear 0.1 step (and stops a light dragged to 0 from freezing there with a zero speed).
    static bool sharedGui(BaseLight &light)
    {
        bool changed = false;
        changed |= ImGui::DragFloat("Light Intensity", &light.intensity, std::max(0.1f, light.intensity * 0.01f),
                                    0.0f, FLT_MAX);
        changed |= ImGui::ColorEdit3("Light Color", &light.color.x);
        return changed;
    }

    bool operator()(PointLight &light) const
    {
        return sharedGui(light);
    }

    bool operator()(DirectionalLight &light) const
    {
        bool changed = sharedGui(light);
        changed |= ImGui::SliderFloat("Angular Radius", &light.angularRadiusDegrees, 0.0f, 10.0f, "%.3f deg");
        return changed;
    }

    bool operator()(SpotLight &light) const
    {
        bool changed = sharedGui(light);
        const bool innerChanged = ImGui::SliderFloat("Inner Cone Angle", &light.innerConeAngleDegrees, 0.0f, 90.0f);
        if (innerChanged && light.innerConeAngleDegrees > light.outerConeAngleDegrees)
            light.outerConeAngleDegrees = light.innerConeAngleDegrees;
        const bool outerChanged = ImGui::SliderFloat("Outer Cone Angle", &light.outerConeAngleDegrees, 0.0f, 90.0f);
        if (outerChanged && light.outerConeAngleDegrees < light.innerConeAngleDegrees)
            light.innerConeAngleDegrees = light.outerConeAngleDegrees;
        changed |= innerChanged || outerChanged;
        changed |= ImGui::DragFloat("Range", &light.range, 0.25f, 0.1f, 10000.0f);
        changed |= ImGui::DragFloat("Shadow Near Plane", &light.shadowNearPlane, 0.01f, 0.01f,
                                    std::max(0.01f, light.range - 0.01f));
        changed |= ImGui::DragFloat("Source Radius", &light.sourceRadius, 0.005f, 0.0f, 10.0f);
        return changed;
    }

    bool operator()(AreaLight &light) const
    {
        bool changed = sharedGui(light);
        changed |= ImGui::DragFloat2("Size", &light.size.x, 0.1f);
        changed |= ImGui::SliderFloat("Spread Angle", &light.spreadAngleDegrees, 1.0f, 89.0f, "%.1f deg");
        changed |= ImGui::Checkbox("Two-Sided", &light.twoSided);
        return changed;
    }

    bool operator()(ImageLight &light) const
    {
        return sharedGui(light);
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

    // The variant carries the light's type as well as its parameters, so pasting turns the target
    // into the same kind of light. Applied through set(), which re-runs the cone-angle clamps.
    std::unique_ptr<ComponentValues> copyValues() const override
    {
        return std::make_unique<ComponentValueSnapshot<LightVariant>>(light);
    }

    void pasteValues(const ComponentValues &values) override
    {
        set(componentValuesAs<LightVariant>(values));
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
            spot->range = std::max(spot->range, 0.02f);
            spot->shadowNearPlane = std::clamp(spot->shadowNearPlane, 0.01f, spot->range - 0.01f);
            spot->sourceRadius    = std::max(spot->sourceRadius, 0.0f);
        }
        if (auto *directional = std::get_if<DirectionalLight>(&variant))
        {
            directional->angularRadiusDegrees = std::clamp(directional->angularRadiusDegrees, 0.0f, 45.0f);
        }
        if (auto *area = std::get_if<AreaLight>(&variant))
        {
            area->spreadAngleDegrees = std::clamp(area->spreadAngleDegrees, 1.0f, 89.0f);
        }
    }
};

} // namespace lr
