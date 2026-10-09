#include "core/scene/SceneObject.hpp"

#include "core/editor/EditorContext.hpp"
#include "core/editor/command/CommandManager.hpp"
#include "core/editor/command/RemoveComponentCommand.hpp"
#include "core/scene/Scene.hpp"
#include "core/scene/TransformComponent.hpp"

#include <memory>
#include <utility>

namespace lr
{

void SceneObject::onGUI(EditorContext &context)
{
    int                            id = 0;
    std::optional<std::type_index> removalRequest;
    for (auto &[type, component] : components)
    {
        ImGui::PushID(id++);
        component->onGUI(context, removalRequest);
        ImGui::PopID();
    }
    // Performed only once the loop is over. A component's own context menu is what asks for this,
    // so erasing where it was asked would invalidate this iteration and destroy the component whose
    // method is still running.
    if (!removalRequest)
    {
        return;
    }

    Component           *component = componentOfType(*removalRequest);
    ComponentValuesAdder adder     = component ? component->valuesAdder() : nullptr;
    std::unique_ptr<ComponentValues> values = component ? component->undoValues() : nullptr;
    // A component that cannot say how to put an equivalent one back is deleted outright. Recording
    // it would offer an undo that silently did nothing, which is worse than offering none.
    if (!adder || !values)
    {
        removeComponent(*removalRequest);
        return;
    }
    context.commands.executeCommand(std::make_unique<RemoveComponentCommand>(
        *m_scene, m_id, *removalRequest, std::move(values), std::move(adder), context.componentAdds));
}

glm::mat4 SceneObject::worldMatrix() const
{
    std::vector<const SceneObject *> ancestry;
    const SceneObject               *current = this;
    while (current)
    {
        ancestry.push_back(current);
        current = current->m_parent ? &m_scene->getSceneObject(current->m_parent.value()) : nullptr;
    }

    glm::mat4 world(1.0f);
    for (auto it = ancestry.rbegin(); it != ancestry.rend(); ++it)
    {
        if ((*it)->hasComponent<TransformComponent>())
        {
            world *= (*it)->getComponent<TransformComponent>().transform().localMatrix();
        }
    }
    return world;
}

glm::quat SceneObject::worldRotation() const
{
    std::vector<const SceneObject *> ancestry;
    const SceneObject               *current = this;
    while (current)
    {
        ancestry.push_back(current);
        current = current->m_parent ? &m_scene->getSceneObject(current->m_parent.value()) : nullptr;
    }

    glm::quat world(1.0f, 0.0f, 0.0f, 0.0f);
    for (auto it = ancestry.rbegin(); it != ancestry.rend(); ++it)
    {
        if ((*it)->hasComponent<TransformComponent>())
        {
            world = glm::normalize(world * (*it)->getComponent<TransformComponent>().transform().rotation());
        }
    }
    return world;
}

} // namespace lr
