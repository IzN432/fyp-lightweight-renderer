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

FluidComponent::FluidComponent(Config config) : FluidComponent(Data(std::move(config))) {}

FluidComponent::FluidComponent(BakedConfig config) : FluidComponent(Data(std::move(config))) {}

FluidComponent::FluidComponent(Data data) : Component("Fluid Simulation"), m_data(std::move(data)) {}

void FluidComponent::setConfig(Config config) { setData(std::move(config)); }

void FluidComponent::setBakedConfig(BakedConfig config) { setData(std::move(config)); }

void FluidComponent::setData(Data data)
{
    if (m_data == data)
    {
        return;
    }
    m_data = std::move(data);
    ++m_revision;
    markDirty();
}

std::unique_ptr<ComponentValues> FluidComponent::copyValues() const
{
    return std::make_unique<ComponentValueSnapshot<Data>>(m_data);
}

void FluidComponent::pasteValues(const ComponentValues &values) { setData(componentValuesAs<Data>(values)); }

ComponentValuesAdder FluidComponent::valuesAdder() const
{
    return [](SceneObject &object, const ComponentValues &values) {
        object.addComponent<FluidComponent>(componentValuesAs<Data>(values));
    };
}

void FluidComponent::onGUIImpl()
{
    ImGui::TextDisabled(isLive() ? "Live fluid. Select this object to edit and run it in the Fluid panel."
                                 : "Baked fluid. Select this object to control playback in the Fluid panel.");
}

} // namespace lr
