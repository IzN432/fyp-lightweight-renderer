#pragma once

#include <cstdint>
#include <vector>
#include <unordered_set>
#include "TranslateDragHandler.hpp"
#include "core/scene/TransformComponent.hpp"

#include <glm/gtc/matrix_inverse.hpp>

namespace lr
{

// What a translate gizmo drives. Lets the same gizmo (ray/axis-projection math, drag-origin
// tracking) serve different "what happens with the drag delta" behaviors — plain vertex
// translation by default, something more involved (e.g. an ARAP solve) via a swapped-in handler.
class VertexDragHandler : public TranslateDragHandler
{
public:
    virtual ~VertexDragHandler() = default;

    // Vertex positions are stored in object-local space, while the shared translate gizmo emits
    // world-space deltas. The editable target is rebound alongside VertexManager so every concrete
    // vertex tool applies the same conversion.
    void setTargetTransform(const TransformComponent *transform) { m_transform = transform; }

    // Vertices this handler currently drives — used generically to position/scale a gizmo at their
    // centroid, regardless of which concrete handler is active.
    virtual const std::unordered_set<uint32_t> &indices() const = 0;

protected:
    glm::vec3 worldDeltaToLocal(const glm::vec3 &worldDelta) const
    {
        if (!m_transform)
        {
            return worldDelta;
        }
        return glm::vec3(glm::inverse(m_transform->worldMatrix()) * glm::vec4(worldDelta, 0.0f));
    }

private:
    const TransformComponent *m_transform = nullptr;
};

} // namespace lr
