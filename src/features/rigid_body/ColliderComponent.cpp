#include "features/rigid_body/ColliderComponent.hpp"

#include <imgui.h>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <array>
#include <type_traits>
#include <variant>

namespace lr
{
namespace
{

constexpr std::array<const char *, 3> kColliderTypeNames = {"Sphere", "Plane", "Box"};
constexpr float                       kMinimumDimension  = 0.001f;

struct ColliderGUICallbacks
{
    void operator()(SphereCollider &sphere) const
    {
        ImGui::DragFloat("Radius", &sphere.radius, 0.01f, kMinimumDimension, 10000.0f);
        sphere.radius = std::max(sphere.radius, kMinimumDimension);
    }

    void operator()(PlaneCollider &plane) const
    {
        ImGui::DragFloat("Offset", &plane.offset, 0.01f);
        ImGui::DragFloat2("Half Extents", &plane.halfExtents.x, 0.1f, kMinimumDimension, 10000.0f);
        plane.halfExtents = glm::max(plane.halfExtents, glm::vec2(kMinimumDimension));
    }

    void operator()(BoxCollider &box) const
    {
        ImGui::DragFloat3("Half Extents", &box.halfExtents.x, 0.01f, kMinimumDimension, 10000.0f);
        box.halfExtents = glm::max(box.halfExtents, glm::vec3(kMinimumDimension));
    }
};

} // namespace

void ColliderComponent::onGUIImpl()
{
    int currentType = static_cast<int>(m_collider.shape.index());
    if (ImGui::Combo("Collider Type", &currentType, kColliderTypeNames.data(),
                     static_cast<int>(kColliderTypeNames.size())))
    {
        switch (currentType)
        {
            case 0:
                m_collider.shape = SphereCollider{};
                break;
            case 1:
                m_collider.shape = PlaneCollider{};
                break;
            case 2:
                m_collider.shape = BoxCollider{};
                break;
        }
    }

    std::visit(ColliderGUICallbacks{}, m_collider.shape);

    glm::vec3 localEulerDegrees = glm::degrees(glm::eulerAngles(m_collider.localRotation));
    ImGui::DragFloat3("Local Position", &m_collider.localPosition.x, 0.01f);
    if (ImGui::DragFloat3("Local Rotation", &localEulerDegrees.x, 0.1f))
    {
        m_collider.localRotation = glm::quat(glm::radians(localEulerDegrees));
    }

    ImGui::SliderFloat("Restitution", &m_collider.restitution, 0.0f, 1.0f);
    ImGui::SliderFloat("Friction", &m_collider.friction, 0.0f, 1.0f);

    ImGui::Text("Visualization");
    ImGui::Checkbox("Show Collider", &m_visible);
    ImGui::ColorEdit3("Collider Color", &m_visualizationColor.x);
    ImGui::SliderFloat("Visible Opacity", &m_visualizationOpacity, 0.0f, 1.0f);
    ImGui::SliderFloat("Occluded Opacity", &m_occludedOpacity, 0.0f, 1.0f);
}

} // namespace lr
