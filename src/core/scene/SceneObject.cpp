#include "core/scene/SceneObject.hpp"

#include "core/scene/Scene.hpp"
#include "core/scene/TransformComponent.hpp"

namespace lr
{

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

} // namespace lr
