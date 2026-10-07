#pragma once

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <vulkan/vulkan.h>

namespace lr
{

// Per-frame view information handed to the active editor state's behavior hooks. States receive
// this instead of the camera object so that state behavior stays independent of how the host
// resolves the active camera, and so the derived matrices are computed once per frame.
struct EditorFrameContext
{
    VkExtent2D extent{0, 0};
    float      aspect = 1.0f;

    glm::mat4 view{1.0f};
    glm::mat4 projection{1.0f};
    glm::mat4 viewProjection{1.0f};

    glm::vec3 cameraPosition{0.0f};
    glm::vec3 cameraForward{0.0f, 0.0f, -1.0f};
    bool      orthographic = false;
};

} // namespace lr
