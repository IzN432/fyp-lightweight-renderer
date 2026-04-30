#pragma once

#include "core/scene/Transform.hpp"
#include "core/scene/Component.hpp"
#include "core/scene/SceneObject.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>

namespace lr
{

enum class ProjectionType
{
    Perspective,
    Orthographic,
};

// Scene camera data. Input/control systems can mutate this state,
// and render systems can consume view/projection matrices from it.
struct Camera : public Component
{
    ProjectionType projectionType = ProjectionType::Perspective;

    // Perspective settings
    float fovYDegrees = 60.0f;
    float nearPlane   = 0.1f;
    float farPlane    = 1000.0f;

    // Orthographic settings (vertical size in world units)
    float orthoHeight = 10.0f;

    explicit Camera() : Component("Camera") {}

    [[nodiscard]] glm::mat4 viewMatrix() const
    {
        const lr::Transform& transform = getOwningObject().getComponent<Transform>();
        const glm::vec3 eye = transform.position;
        return glm::lookAt(eye, eye + transform.forward(), transform.up());
    }

    [[nodiscard]] glm::mat4 projectionMatrix(float aspectRatio) const
    {
        const float safeAspect = std::max(aspectRatio, 0.0001f);

        glm::mat4 projection(1.0f);
        if (projectionType == ProjectionType::Perspective)
        {
            projection = glm::perspective(glm::radians(fovYDegrees), safeAspect,
                                          nearPlane, farPlane);
            projection[1][1] *= -1.0f;
        }
        else
        {
            const float halfH = orthoHeight * 0.5f;
            const float halfW = halfH * safeAspect;
            projection = glm::ortho(-halfW, halfW, -halfH, halfH, nearPlane, farPlane);
            projection[1][1] *= -1.0f;
        }

        // Vulkan clip-space fixup for default GLM matrices:
        // remap depth from [-1, 1] to [0, 1]
        glm::mat4 clip(1.0f);
        clip[2][2] = 0.5f;
        clip[3][2] = 0.5f;
        return clip * projection;
    }

    [[nodiscard]] glm::mat4 viewProjectionMatrix(float aspectRatio) const
    {
        return projectionMatrix(aspectRatio) * viewMatrix();
    }

    void onGUIImpl() override
    {
        bool changed = false;

        const char* projectionTypes[] = { "Perspective", "Orthographic" };
        int currentProjection = static_cast<int>(projectionType);
        if (ImGui::Combo("Projection Type", &currentProjection, projectionTypes, IM_ARRAYSIZE(projectionTypes)))
        {
            projectionType = static_cast<ProjectionType>(currentProjection);
            changed = true;
        }

        if (projectionType == ProjectionType::Perspective)
        {
            changed |= ImGui::SliderFloat("FOV Y (Degrees)", &fovYDegrees, 1.0f, 179.0f);
        }
        else
        {
            changed |= ImGui::SliderFloat("Orthographic Height", &orthoHeight, 0.1f, 100.0f);
        }

        changed |= ImGui::SliderFloat("Near Plane", &nearPlane, 0.01f, farPlane - 0.01f);
        changed |= ImGui::SliderFloat("Far Plane", &farPlane, nearPlane + 0.01f, 10000.0f);

        if (changed)
        {
            notifyChanged();
        }
    }
};

} // namespace lr
