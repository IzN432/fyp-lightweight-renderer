#pragma once

#include <glm/vec3.hpp>

namespace lr
{

// Behavior driven by a translation gizmo. Implementations decide what the world-space drag
// delta affects (vertices, a scene object, or another translatable editor target).
class TranslateDragHandler
{
public:
    virtual ~TranslateDragHandler() = default;

    virtual void beginDrag() = 0;
    virtual void translate(const glm::vec3 &frameDelta) = 0;
    virtual void endDrag(const glm::vec3 &totalDelta) = 0;
};

} // namespace lr
