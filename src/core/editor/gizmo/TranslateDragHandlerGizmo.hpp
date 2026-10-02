#pragma once

#include "core/editor/TranslateDragHandler.hpp"

namespace lr
{

// Mixin for translation gizmos whose target behavior can be reassigned at runtime.
class TranslateDragHandlerGizmo
{
public:
    virtual ~TranslateDragHandlerGizmo() = default;
    virtual void setDragHandler(TranslateDragHandler &handler) = 0;
    virtual TranslateDragHandler &dragHandler() const = 0;
};

} // namespace lr
