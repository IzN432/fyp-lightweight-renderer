#pragma once

#include "core/scene/Component.hpp"
#include "core/scene/SceneObject.hpp"
#include "core/scene/Transform.hpp"

#include <imgui.h>

#include <memory>
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

    // Position, rotation and scale are all a transform owns, so its snapshot is the whole
    // Transform. Carrying the Transform itself also carries its euler angles rather than
    // re-deriving them from the quaternion, so a paste reproduces the numbers that were copied.
    std::unique_ptr<ComponentValues> copyValues() const override
    {
        return std::make_unique<ComponentValueSnapshot<Transform>>(m_transform);
    }

    void pasteValues(const ComponentValues &values) override
    {
        m_transform = componentValuesAs<Transform>(values);
        markDirty();
    }

    // The mirror of the rule the Inspector applies when adding one: every other component reads
    // its object's transform, so this may only go while there is nothing left to read it.
    bool allowsRemoval() const override
    {
        const SceneObject *owner = findOwningObject();
        return owner != nullptr && owner->componentCount() == 1;
    }

    // The one component with no prerequisite of its own, so it can be given to a bare object.
    ComponentValuesAdder valuesAdder() const override
    {
        return [](SceneObject &object, const ComponentValues &values) {
            object.addComponent<TransformComponent>(componentValuesAs<Transform>(values));
        };
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
