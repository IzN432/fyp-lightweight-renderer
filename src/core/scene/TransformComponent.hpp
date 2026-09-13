#pragma once

#include "core/scene/Component.hpp"
#include "core/scene/Transform.hpp"

#include <imgui.h>

#include <utility>

namespace lr
{

// Scene component wrapper around a reusable Transform. 
// Implements explicit update functions for accurate dirty 
// tracking.
struct TransformComponent : public Component
{
public:
    explicit TransformComponent(glm::vec3 position = glm::vec3(0.0f),
                                glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
                                glm::vec3 scale = glm::vec3(1.0f))
        : Component("TransformComponent"), m_transform(position, rotation, scale)
    {}

    explicit TransformComponent(Transform transform)
        : Component("TransformComponent"), m_transform(std::move(transform))
    {}

    const Transform &transform() const { return m_transform; }

    // Includes the transforms of this object's scene ancestors.
    glm::mat4 worldMatrix() const;

    void setRotation(const glm::quat &rotation)
    {
        m_transform.setRotation(rotation);
        markDirty();
    }

    void setEulerDegrees(const glm::vec3 &degrees)
    {
        m_transform.setEulerDegrees(degrees);
        markDirty();
    }

    void setPosition(const glm::vec3 &position)
    {
        m_transform.setPosition(position);
        markDirty();
    }

    void setScale(const glm::vec3 &scale)
    {
        m_transform.setScale(scale);
        markDirty();
    }

    void onGUIImpl() override
    {
        glm::vec3 position     = m_transform.position();
        glm::vec3 eulerDegrees = m_transform.eulerDegrees();
        glm::vec3 scale        = m_transform.scale();

        bool changed = false;
        if (ImGui::DragFloat3("Position", &position.x, 0.1f))
        {
            m_transform.setPosition(position);
            changed = true;
        }
        if (ImGui::DragFloat3("Rotation (Degrees)", &eulerDegrees.x, 0.1f))
        {
            m_transform.setEulerDegrees(eulerDegrees);
            changed = true;
        }
        if (ImGui::DragFloat3("Scale", &scale.x, 0.1f))
        {
            m_transform.setScale(scale);
            changed = true;
        }
        if (changed)
        {
            markDirty();
        }
    }

private:
    Transform m_transform;
};

} // namespace lr
