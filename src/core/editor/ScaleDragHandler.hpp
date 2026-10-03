#pragma once

#include <glm/mat4x4.hpp>

namespace lr
{

class ScaleDragHandler
{
public:
    virtual ~ScaleDragHandler() = default;

    virtual void beginDrag() = 0;
    virtual void scaleToWorld(const glm::mat4 &worldMatrix) = 0;
    virtual void endDrag() = 0;
};

} // namespace lr
