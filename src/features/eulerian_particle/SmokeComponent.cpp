#include "features/eulerian_particle/SmokeComponent.hpp"

#include "core/scene/SceneObject.hpp"

#include <imgui.h>

#include <utility>

namespace lr::eulerian
{

SmokeComponent::SmokeComponent(Config config) : Component("Smoke Simulation"), m_config(std::move(config)) {}

void SmokeComponent::setConfig(Config config)
{
    if (m_config == config) return;
    m_config = std::move(config);
    ++m_revision;
    markDirty();
}

std::unique_ptr<ComponentValues> SmokeComponent::copyValues() const
{
    return std::make_unique<ComponentValueSnapshot<Config>>(m_config);
}

void SmokeComponent::pasteValues(const ComponentValues &values)
{
    setConfig(componentValuesAs<Config>(values));
}

ComponentValuesAdder SmokeComponent::valuesAdder() const
{
    return [](SceneObject &object, const ComponentValues &values) {
        object.addComponent<SmokeComponent>(componentValuesAs<Config>(values));
    };
}

void SmokeComponent::onGUIImpl()
{
    ImGui::TextDisabled("Select this object to edit and run it in the Smoke panel.");
}

} // namespace lr::eulerian
