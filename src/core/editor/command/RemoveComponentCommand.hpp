#pragma once

#include "Command.hpp"

#include "core/scene/ComponentClipboard.hpp"
#include "core/scene/SceneObjectId.hpp"

#include <memory>
#include <typeindex>

namespace lr
{

class Scene;
class ComponentAddService;

// A component deleted from the Inspector's component menu. The mirror of AddComponentCommand: it
// carries the values the component held so undo can put an equivalent one back.
//
// Only a component that both allows removal and knows how to be added from its values is recorded
// this way; see SceneObject::onGUI, which deletes the rest outright rather than offering an undo it
// could not honour.
class RemoveComponentCommand : public Command
{
public:
    RemoveComponentCommand(Scene &scene, SceneObjectId object, std::type_index componentType,
                           std::unique_ptr<ComponentValues> values, ComponentValuesAdder adder,
                           ComponentAddService &addService)
        : m_scene(scene), m_object(object), m_componentType(componentType), m_values(std::move(values)),
          m_adder(std::move(adder)), m_addService(addService)
    {}

    void execute() override;
    void undo() override;

private:
    Scene                           &m_scene;
    SceneObjectId                    m_object;
    std::type_index                  m_componentType;
    std::unique_ptr<ComponentValues> m_values;
    ComponentValuesAdder             m_adder;
    ComponentAddService             &m_addService;
};

} // namespace lr
