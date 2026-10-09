#pragma once

#include "Command.hpp"

#include "core/scene/ComponentClipboard.hpp"
#include "core/scene/SceneObjectId.hpp"

#include <memory>
#include <typeindex>

namespace lr
{

class Scene;

// One completed edit of a component's values in the Inspector, whichever widget made it: the
// component hands over what its values were when the interaction started and what they are now,
// and this puts either set back (see Component::undoValues).
//
// The target is named by object id and component type rather than held by reference, because the
// component can be deleted and re-added while this command sits in the history. A command whose
// target is no longer there does nothing instead of writing through a dangling pointer.
class EditComponentValuesCommand : public Command
{
public:
    EditComponentValuesCommand(Scene &scene, SceneObjectId object, std::type_index componentType,
                               std::unique_ptr<ComponentValues> before,
                               std::unique_ptr<ComponentValues> after)
        : m_scene(scene), m_object(object), m_componentType(componentType), m_before(std::move(before)),
          m_after(std::move(after))
    {}

    void execute() override;
    void undo() override;

private:
    void restore(const ComponentValues &values);

    Scene                           &m_scene;
    SceneObjectId                    m_object;
    std::type_index                  m_componentType;
    std::unique_ptr<ComponentValues> m_before;
    std::unique_ptr<ComponentValues> m_after;
};

} // namespace lr
