#include "features/eulerian_particle/SmokeComponent.hpp"

#include "core/scene/SceneObject.hpp"

#include <imgui.h>

#include <utility>

namespace lr::eulerian
{

SmokeComponent::SmokeComponent(Config config) : Component("Smoke Simulation"), m_data(std::move(config)) {}
SmokeComponent::SmokeComponent(BakedConfig config) : Component("Baked Smoke"), m_data(std::move(config)) {}
SmokeComponent::SmokeComponent(Data data) : Component("Smoke Simulation"), m_data(std::move(data)) {}

void SmokeComponent::setConfig(Config config)
{
    setData(std::move(config));
}

void SmokeComponent::setBakedConfig(BakedConfig config)
{
    setData(std::move(config));
}

void SmokeComponent::setData(Data data)
{
    if (m_data == data) return;
    m_data = std::move(data);
    ++m_revision;
    markDirty();
}

std::unique_ptr<ComponentValues> SmokeComponent::copyValues() const
{
    return std::make_unique<ComponentValueSnapshot<Data>>(m_data);
}

void SmokeComponent::pasteValues(const ComponentValues &values)
{
    setData(componentValuesAs<Data>(values));
}

ComponentValuesAdder SmokeComponent::valuesAdder() const
{
    return [](SceneObject &object, const ComponentValues &values) {
        object.addComponent<SmokeComponent>(componentValuesAs<Data>(values));
    };
}

void SmokeComponent::onGUIImpl()
{
    ImGui::TextDisabled(isBaked() ? "Select this object to play its baked gas cache in the Smoke panel."
                                  : "Select this object to edit and run it in the Smoke panel.");
}

} // namespace lr::eulerian
