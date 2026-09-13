#include "core/scene/TransformComponent.hpp"

#include "core/scene/SceneObject.hpp"

namespace lr
{

glm::mat4 TransformComponent::worldMatrix() const
{
    return getOwningObject().worldMatrix();
}

} // namespace lr
