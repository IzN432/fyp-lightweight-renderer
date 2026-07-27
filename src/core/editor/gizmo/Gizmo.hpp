#pragma once

#include "core/passes/overlaygeometry/OverlayInstance.hpp"

namespace lr
{

class Gizmo
{
public:
    Gizmo(OverlayInstance instance)
        : m_instance(instance) {};
    virtual ~Gizmo() = default;

    // Returns the OverlayInstance to be used in overlay geometry pass for rendering
    const OverlayInstance &getInstance() const { return m_instance; }

    // Mouse interaction callbacks: override these in derived classes to implement gizmo behavior
    virtual void onMouseDown(double ndcX, double ndcY, double aspect) {}
    virtual void onMouseUp(double ndcX, double ndcY, double aspect) {}
    virtual void dragCallback(double ndcX, double ndcY, double dNdcX, double dNdcY, double aspect) {}

    void setPosition(const glm::vec3 &pos) { m_instance.position = pos; }
    void setScale(const glm::vec3 &scale) { m_instance.scale = scale; }
    void setPickingId(uint32_t id) { m_instance.pickingId = id; }
protected:
    OverlayInstance m_instance;
};
} // namespace lr
