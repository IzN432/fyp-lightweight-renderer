#pragma once

#include <cstdint>
#include <vector>
#include <unordered_set>
#include <glm/vec3.hpp>

namespace lr
{

// What a translate gizmo drives. Lets the same gizmo (ray/axis-projection math, drag-origin
// tracking) serve different "what happens with the drag delta" behaviors — plain vertex
// translation by default, something more involved (e.g. an ARAP solve) via a swapped-in handler.
class VertexDragHandler
{
public:
    virtual ~VertexDragHandler() = default;

    // Vertices this handler currently drives — used generically to position/scale a gizmo at their
    // centroid, regardless of which concrete handler is active.
    virtual const std::unordered_set<uint32_t> &indices() const = 0;

    // Drag gesture started, before any translate() calls this gesture.
    virtual void beginDrag() = 0;

    // Called every drag frame with this frame's incremental delta.
    virtual void translate(const glm::vec3 &frameDelta) = 0;

    // Drag gesture ended. totalDelta is the accumulated delta since beginDrag(). Implementations
    // push their own undo Command here — translate() already applied the live visual effect.
    virtual void endDrag(const glm::vec3 &totalDelta) = 0;
};

} // namespace lr
