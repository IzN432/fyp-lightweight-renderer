#pragma once

#include "core/scene/Component.hpp"
#include "features/rigid_body/Collider.hpp"

#include <glm/glm.hpp>

#include <utility>
#include <vector>

namespace lr
{

class ColliderComponent : public Component
{
public:
    explicit ColliderComponent(Collider collider = {})
        : Component("ColliderComponent"), m_colliders{std::move(collider)}
    {}

    explicit ColliderComponent(std::vector<Collider> colliders)
        : Component("ColliderComponent"), m_colliders(std::move(colliders))
    {}

    // Convenience access for existing single-collider callers.
    Collider       &collider() { return m_colliders.front(); }
    const Collider &collider() const { return m_colliders.front(); }
    std::vector<Collider>       &colliders() { return m_colliders; }
    const std::vector<Collider> &colliders() const { return m_colliders; }

    Collider &addCollider(Collider collider = {})
    {
        m_colliders.push_back(std::move(collider));
        markDirty();
        return m_colliders.back();
    }

    bool visible() const { return m_visible; }
    void setVisible(bool visible)
    {
        m_visible = visible;
        markDirty();
    }
    const glm::vec3 &visualizationColor() const { return m_visualizationColor; }
    float visualizationOpacity() const { return m_visualizationOpacity; }
    float occludedOpacity() const { return m_occludedOpacity; }

    void onGUIImpl() override;

private:
    std::vector<Collider> m_colliders;
    bool      m_visible              = false;
    glm::vec3 m_visualizationColor   = {0.15f, 0.85f, 0.35f};
    float     m_visualizationOpacity = 1.0f;
    float     m_occludedOpacity      = 0.15f;
};

} // namespace lr
