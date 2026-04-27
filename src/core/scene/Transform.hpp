#pragma once

#include "core/scene/Component.hpp"

#include <imgui.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace lr
{

// Shared spatial transform for scene objects (camera, mesh instances, lights).
struct Transform : public Component
{
private:
    glm::quat m_rotation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 m_eulerDegrees{0.0f, 0.0f, 0.0f};
public:
    glm::vec3 position{0.0f, 0.0f, 0.0f};
    glm::vec3 scale{1.0f, 1.0f, 1.0f};

    explicit Transform(glm::vec3 position = glm::vec3(0.0f), 
                    glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f), 
                    glm::vec3 scale = glm::vec3(1.0f))
        : position(position), m_rotation(rotation), scale(scale)
        , m_eulerDegrees(glm::degrees(glm::eulerAngles(rotation)))
        , Component("Transform")
    {}

    const glm::quat& rotation() const { return m_rotation; }
    const glm::vec3& eulerDegrees() const { return m_eulerDegrees; }

    void setRotation(const glm::quat& q)
    {
        m_rotation = q;
        m_eulerDegrees = glm::degrees(glm::eulerAngles(q));
    }

    void setEulerDegrees(const glm::vec3& degrees)
    {
        m_eulerDegrees = degrees;
        m_rotation = glm::quat(glm::radians(degrees));
    }

    bool onGUIImpl() override
    {
        bool changed = false;
        changed |= ImGui::DragFloat3("Position", &position.x, 0.1f);
        if (ImGui::DragFloat3("Rotation (Degrees)", &m_eulerDegrees.x, 0.1f))
        {
            m_rotation = glm::quat(glm::radians(m_eulerDegrees));
            changed = true;
        }
        changed |= ImGui::DragFloat3("Scale", &scale.x, 0.1f);
        return changed;
    }

    [[nodiscard]] glm::mat4 localMatrix() const
    {
        const glm::mat4 t = glm::translate(glm::mat4(1.0f), position);
        const glm::mat4 r = glm::mat4_cast(m_rotation);
        const glm::mat4 s = glm::scale(glm::mat4(1.0f), scale);
        return t * r * s;
    }

    [[nodiscard]] glm::mat4 worldMatrix(const glm::mat4 &parentWorld) const
    {
        return parentWorld * localMatrix();
    }

    [[nodiscard]] glm::vec3 forward() const
    {
        return glm::normalize(m_rotation * glm::vec3(0.0f, 0.0f, -1.0f));
    }

    [[nodiscard]] glm::vec3 right() const
    {
        return glm::normalize(m_rotation * glm::vec3(1.0f, 0.0f, 0.0f));
    }

    [[nodiscard]] glm::vec3 up() const
    {
        return glm::normalize(m_rotation * glm::vec3(0.0f, 1.0f, 0.0f));
    }

};

} // namespace lr
