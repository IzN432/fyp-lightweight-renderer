#pragma once

#include "core/scene/Component.hpp"
#include "features/rigid_body/Collider.hpp"

#include <glm/glm.hpp>

#include <utility>

namespace lr
{

class ColliderComponent : public Component
{
public:
    explicit ColliderComponent(Collider collider = {})
        : Component("ColliderComponent"), m_collider(std::move(collider))
    {}

    Collider       &collider() { return m_collider; }
    const Collider &collider() const { return m_collider; }

    bool visible() const { return m_visible; }
    const glm::vec3 &visualizationColor() const { return m_visualizationColor; }
    float visualizationOpacity() const { return m_visualizationOpacity; }
    float occludedOpacity() const { return m_occludedOpacity; }

    void onGUIImpl() override;

private:
    Collider  m_collider;
    bool      m_visible              = false;
    glm::vec3 m_visualizationColor   = {0.15f, 0.85f, 0.35f};
    float     m_visualizationOpacity = 1.0f;
    float     m_occludedOpacity      = 0.15f;
};

} // namespace lr
