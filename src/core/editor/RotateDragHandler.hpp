#pragma once

#include <glm/mat4x4.hpp>

namespace lr
{

class RotateDragHandler
{
public:
    virtual ~RotateDragHandler() = default;

    virtual void beginDrag() = 0;
    virtual void rotateToWorld(const glm::mat4 &worldMatrix) = 0;
    virtual void endDrag() = 0;
};

} // namespace lr
