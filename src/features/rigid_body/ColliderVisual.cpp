#include "features/rigid_body/ColliderVisual.hpp"

#include "core/scene/Scene.hpp"
#include "core/scene/SceneObject.hpp"
#include "features/rigid_body/ColliderComponent.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <type_traits>
#include <variant>

namespace lr
{
namespace
{

OverlayLineStyle lineStyle(const ColliderComponent &component)
{
    return {
        .color            = component.visualizationColor(),
        .visibleOpacity   = component.visualizationOpacity(),
        .occludedOpacity = component.occludedOpacity(),
    };
}

glm::mat4 colliderLocalMatrix(const Collider &collider)
{
    return glm::translate(glm::mat4(1.0f), collider.localPosition) * glm::mat4_cast(collider.localRotation);
}

void appendColliderLines(OverlayLineBuilder &builder, const SceneObject &object,
                         const ColliderComponent &component)
{
    const OverlayLineStyle style = lineStyle(component);
    for (const Collider &collider : component.colliders())
    {
        const glm::mat4 base = object.worldMatrix() * colliderLocalMatrix(collider);
        std::visit(
            [&](const auto &shape) {
                using Shape = std::decay_t<decltype(shape)>;
                if constexpr (std::is_same_v<Shape, SphereCollider>)
                    builder.addPrimitive(primitives::sphere.outline,
                                         glm::scale(base, glm::vec3(shape.radius)), style);
                else if constexpr (std::is_same_v<Shape, BoxCollider>)
                    builder.addPrimitive(primitives::cube.outline,
                                         glm::scale(base, shape.halfExtents * 2.0f), style);
                else
                {
                    glm::mat4 model = glm::translate(base, glm::vec3(0.0f, shape.offset, 0.0f));
                    const glm::vec2 size = shape.halfExtents * 2.0f;
                    builder.addPrimitive(primitives::plane.outline,
                                         glm::scale(model, glm::vec3(size.x, 1.0f, size.y)), style);
                }
            }, collider.shape);
    }
}

} // namespace

std::vector<OverlayLine> buildColliderOverlayLines(const Scene &scene)
{
    OverlayLineBuilder builder;
    for (const auto &object : scene.sceneObjects())
    {
        if (!object->hasComponent<ColliderComponent>())
        {
            continue;
        }

        const ColliderComponent &component = object->getComponent<ColliderComponent>();
        if (component.visible())
        {
            appendColliderLines(builder, *object, component);
        }
    }
    return builder.takeLines();
}

} // namespace lr
