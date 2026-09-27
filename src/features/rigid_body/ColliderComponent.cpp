#include "features/rigid_body/ColliderComponent.hpp"

#include <imgui.h>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>

namespace lr
{
namespace
{

constexpr float                       kMinimumDimension  = 0.001f;

const char *colliderShapeTypeName(ColliderShapeType type)
{
    switch (type)
    {
        case ColliderShapeType::Sphere: return "Sphere";
        case ColliderShapeType::Plane: return "Plane";
        case ColliderShapeType::Box: return "Box";
    }
    return "Unknown";
}

struct ColliderGUICallbacks
{
    bool &changed;

    void operator()(SphereCollider &sphere) const
    {
        changed |= ImGui::DragFloat("Radius", &sphere.radius, 0.01f, kMinimumDimension, 10000.0f);
        sphere.radius = std::max(sphere.radius, kMinimumDimension);
    }

    void operator()(PlaneCollider &plane) const
    {
        changed |= ImGui::DragFloat("Offset", &plane.offset, 0.01f);
        changed |= ImGui::DragFloat2("Half Extents", &plane.halfExtents.x, 0.1f, kMinimumDimension, 10000.0f);
        plane.halfExtents = glm::max(plane.halfExtents, glm::vec2(kMinimumDimension));
    }

    void operator()(BoxCollider &box) const
    {
        changed |= ImGui::DragFloat3("Half Extents", &box.halfExtents.x, 0.01f, kMinimumDimension, 10000.0f);
        box.halfExtents = glm::max(box.halfExtents, glm::vec3(kMinimumDimension));
    }
};

} // namespace

void ColliderComponent::onGUIImpl()
{
    bool changed = false;
    std::optional<std::size_t> removeIndex;
    for (std::size_t index = 0; index < m_colliders.size(); ++index)
    {
        Collider &collider = m_colliders[index];
        ImGui::PushID(static_cast<int>(index));
        const std::string label = "Collider " + std::to_string(index + 1);
        if (ImGui::TreeNodeEx(label.c_str(), ImGuiTreeNodeFlags_DefaultOpen))
        {
            const ColliderShapeType currentType = colliderShapeType(collider.shape);
            if (ImGui::BeginCombo("Type", colliderShapeTypeName(currentType)))
            {
                for (ColliderShapeType candidate : {ColliderShapeType::Sphere,
                                                    ColliderShapeType::Plane,
                                                    ColliderShapeType::Box})
                {
                    const bool selected = candidate == currentType;
                    if (ImGui::Selectable(colliderShapeTypeName(candidate), selected) && !selected)
                    {
                        collider.shape = makeColliderShape(candidate);
                        changed = true;
                    }
                    if (selected) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }

            std::visit(ColliderGUICallbacks{changed}, collider.shape);
            glm::vec3 localEulerDegrees = glm::degrees(glm::eulerAngles(collider.localRotation));
            changed |= ImGui::DragFloat3("Local Position", &collider.localPosition.x, 0.01f);
            if (ImGui::DragFloat3("Local Rotation", &localEulerDegrees.x, 0.1f))
            {
                collider.localRotation = glm::quat(glm::radians(localEulerDegrees));
                changed = true;
            }
            ImGui::Text("Physics Material");
            changed |= ImGui::SliderFloat("Restitution", &collider.material.restitution, 0.0f, 1.0f);
            changed |= ImGui::SliderFloat("Friction", &collider.material.friction, 0.0f, 2.0f);
            if (m_colliders.size() > 1 && ImGui::Button("Remove Collider")) removeIndex = index;
            ImGui::TreePop();
        }
        ImGui::PopID();
    }

    if (removeIndex)
    {
        m_colliders.erase(m_colliders.begin() + static_cast<std::ptrdiff_t>(*removeIndex));
        changed = true;
    }
    if (ImGui::Button("Add Collider")) addCollider();

    ImGui::Text("Visualization");
    ImGui::Checkbox("Show Collider", &m_visible);
    ImGui::ColorEdit3("Collider Color", &m_visualizationColor.x);
    ImGui::SliderFloat("Visible Opacity", &m_visualizationOpacity, 0.0f, 1.0f);
    ImGui::SliderFloat("Occluded Opacity", &m_occludedOpacity, 0.0f, 1.0f);

    if (changed) markDirty();
}

} // namespace lr
