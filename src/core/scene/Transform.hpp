#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

namespace lr
{

// Reusable local spatial transform, independent of the scene component system.
class Transform
{
public:
    explicit Transform(glm::vec3 position = glm::vec3(0.0f),
                       glm::quat rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f),
                       glm::vec3 scale = glm::vec3(1.0f))
        : m_rotation(rotation), m_eulerDegrees(glm::degrees(glm::eulerAngles(rotation))), m_position(position),
          m_scale(scale)
    {}

    const glm::quat &rotation() const { return m_rotation; }
    const glm::vec3 &eulerDegrees() const { return m_eulerDegrees; }
    const glm::vec3 &position() const { return m_position; }
    const glm::vec3 &scale() const { return m_scale; }

    void setRotation(const glm::quat &rotation)
    {
        m_rotation     = rotation;
        m_eulerDegrees = glm::degrees(glm::eulerAngles(rotation));
    }

    void setEulerDegrees(const glm::vec3 &degrees)
    {
        m_eulerDegrees = degrees;
        m_rotation     = glm::quat(glm::radians(degrees));
    }

    void setPosition(const glm::vec3 &position) { m_position = position; }
    void setScale(const glm::vec3 &scale) { m_scale = scale; }

    [[nodiscard]] glm::mat4 localMatrix() const
    {
        const glm::mat4 translation = glm::translate(glm::mat4(1.0f), m_position);
        const glm::mat4 rotation    = glm::mat4_cast(m_rotation);
        const glm::mat4 scale       = glm::scale(glm::mat4(1.0f), m_scale);
        return translation * rotation * scale;
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

    // Member-wise, euler angles included: two transforms that agree on a rotation but disagree on
    // the angles the inspector shows for it are not the same edit state.
    bool operator==(const Transform &) const = default;

private:
    glm::quat m_rotation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 m_eulerDegrees{0.0f, 0.0f, 0.0f};
    glm::vec3 m_position{0.0f, 0.0f, 0.0f};
    glm::vec3 m_scale{1.0f, 1.0f, 1.0f};
};

} // namespace lr
