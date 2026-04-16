#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace lr
{

// Shared spatial transform for scene objects (camera, mesh instances, lights).
struct Transform
{
    glm::vec3 position{0.0f, 0.0f, 0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f}; // identity quaternion: w, x, y, z
    glm::vec3 scale{1.0f, 1.0f, 1.0f};

    [[nodiscard]] glm::mat4 localMatrix() const
    {
        const glm::mat4 t = glm::translate(glm::mat4(1.0f), position);
        const glm::mat4 r = glm::mat4_cast(rotation);
        const glm::mat4 s = glm::scale(glm::mat4(1.0f), scale);
        return t * r * s;
    }

    [[nodiscard]] glm::mat4 worldMatrix(const glm::mat4 &parentWorld) const
    {
        return parentWorld * localMatrix();
    }

    [[nodiscard]] glm::vec3 forward() const
    {
        return glm::normalize(rotation * glm::vec3(0.0f, 0.0f, -1.0f));
    }

    [[nodiscard]] glm::vec3 right() const
    {
        return glm::normalize(rotation * glm::vec3(1.0f, 0.0f, 0.0f));
    }

    [[nodiscard]] glm::vec3 up() const
    {
        return glm::normalize(rotation * glm::vec3(0.0f, 1.0f, 0.0f));
    }
};

} // namespace lr
