#pragma once

#include <cstdint>
#include <vector>
#include <unordered_set>
#include "TranslateDragHandler.hpp"

namespace lr
{

// What a translate gizmo drives. Lets the same gizmo (ray/axis-projection math, drag-origin
// tracking) serve different "what happens with the drag delta" behaviors — plain vertex
// translation by default, something more involved (e.g. an ARAP solve) via a swapped-in handler.
class VertexDragHandler : public TranslateDragHandler
{
public:
    virtual ~VertexDragHandler() = default;

    // Vertices this handler currently drives — used generically to position/scale a gizmo at their
    // centroid, regardless of which concrete handler is active.
    virtual const std::unordered_set<uint32_t> &indices() const = 0;

};

} // namespace lr
