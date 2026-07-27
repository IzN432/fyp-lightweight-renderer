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
    glm::vec3 m_position{0.0f, 0.0f, 0.0f};
    glm::vec3 m_scale{1.0f, 1.0f, 1.0f};
public:
    explicit Transform(glm::vec3 position = glm::vec3(0.0f), 
                    glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f), 
                    glm::vec3 scale = glm::vec3(1.0f))
        : m_position(position), m_rotation(rotation), m_scale(scale)
        , m_eulerDegrees(glm::degrees(glm::eulerAngles(rotation)))
        , Component("Transform")
    {}

    const glm::quat& rotation() const { return m_rotation; }
    const glm::vec3& eulerDegrees() const { return m_eulerDegrees; }
    const glm::vec3& position() const { return m_position; }
    const glm::vec3& scale() const { return m_scale; }

    void setRotation(const glm::quat& q)
    {
        m_rotation = q;
        m_eulerDegrees = glm::degrees(glm::eulerAngles(q));
        notifyChanged();
    }

    void setEulerDegrees(const glm::vec3& degrees)
    {
        m_eulerDegrees = degrees;
        m_rotation = glm::quat(glm::radians(degrees));
        notifyChanged();
    }

    void setPosition(const glm::vec3& pos)
    {
        m_position = pos;
        notifyChanged();
    }

    void setScale(const glm::vec3& s)
    {
        m_scale = s;
        notifyChanged();
    }

    void onGUIImpl() override
    {
        bool changed = false;
        changed |= ImGui::DragFloat3("Position", &m_position.x, 0.1f);
        if (ImGui::DragFloat3("Rotation (Degrees)", &m_eulerDegrees.x, 0.1f))
        {
            m_rotation = glm::quat(glm::radians(m_eulerDegrees));
            changed = true;
        }
        changed |= ImGui::DragFloat3("Scale", &m_scale.x, 0.1f);
        if (changed)
        {
            notifyChanged();
        }
    }

    [[nodiscard]] glm::mat4 localMatrix() const
    {
        const glm::mat4 t = glm::translate(glm::mat4(1.0f), m_position);
        const glm::mat4 r = glm::mat4_cast(m_rotation);
        const glm::mat4 s = glm::scale(glm::mat4(1.0f), m_scale);
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
