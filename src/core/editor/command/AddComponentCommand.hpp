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

// A component the Inspector has just put on an object, from the catalog or from the clipboard.
//
// The add itself has already happened by the time this is recorded, so the values are read back off
// the new component rather than passed in: redoing then reproduces the component the user actually
// got, including whatever the catalog's factory chose for it.
class AddComponentCommand : public Command
{
public:
    AddComponentCommand(Scene &scene, SceneObjectId object, std::type_index componentType,
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
