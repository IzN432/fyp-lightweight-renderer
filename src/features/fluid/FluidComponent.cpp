#include "features/fluid/FluidComponent.hpp"

#include "core/scene/SceneObject.hpp"

#include <imgui.h>

#include <algorithm>
#include <utility>

namespace lr
{

float FluidComponent::Config::smoothingRadius() const
{
    return std::max(particleSpacing, 1e-4f) * std::max(smoothingRadiusScale, 1.0f);
}

float FluidComponent::Config::particleRadius() const
{
    return std::max(particleSpacing, 1e-4f) * std::max(particleRenderScale, 0.0f);
}

FluidComponent::FluidComponent(Config config) : Component("Fluid Simulation"), m_config(std::move(config)) {}

void FluidComponent::setConfig(Config config)
{
    if (m_config == config) return;
    m_config = std::move(config);
    ++m_revision;
    markDirty();
}

std::unique_ptr<ComponentValues> FluidComponent::copyValues() const
{
    return std::make_unique<ComponentValueSnapshot<Config>>(m_config);
}

void FluidComponent::pasteValues(const ComponentValues &values)
{
    setConfig(componentValuesAs<Config>(values));
}

ComponentValuesAdder FluidComponent::valuesAdder() const
{
    return [](SceneObject &object, const ComponentValues &values) {
        object.addComponent<FluidComponent>(componentValuesAs<Config>(values));
    };
}

void FluidComponent::onGUIImpl()
{
    ImGui::TextDisabled("Select this object to edit and run it in the Fluid panel.");
}

} // namespace lr
