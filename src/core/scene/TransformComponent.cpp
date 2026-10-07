#include "core/scene/TransformComponent.hpp"

#include "core/scene/SceneObject.hpp"

namespace lr
{

glm::mat4 TransformComponent::worldMatrix() const
{
    // A detached TransformComponent — one not owned by a SceneObject, such as the placeholder
    // SceneManager binds while no mesh is being edited — has no ancestors to accumulate, so its
    // local transform already is its world transform.
    const SceneObject *owner = findOwningObject();
    return owner ? owner->worldMatrix() : m_transform.localMatrix();
}

} // namespace lr
