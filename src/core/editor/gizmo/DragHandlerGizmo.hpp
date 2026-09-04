#pragma once

#include "core/editor/VertexDragHandler.hpp"

namespace lr
{

// Mixin for gizmos that drive a VertexDragHandler (currently the translate gizmos). Kept separate
// from the generic Gizmo base rather than added there, since not every gizmo type has a handler
// to reassign.
class DragHandlerGizmo
{
public:
    virtual ~DragHandlerGizmo()                                                 = default;
    virtual void                     setDragHandler(VertexDragHandler &handler) = 0;
    virtual const VertexDragHandler &dragHandler() const                        = 0;
};

} // namespace lr
