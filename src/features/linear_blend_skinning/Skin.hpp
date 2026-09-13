#pragma once

#include "features/linear_blend_skinning/Joint.hpp"

#include <vector>

namespace lr
{

class Scene;
class SceneObject;

class Skin
{
public:
    Skin(Scene &scene, std::vector<Joint> joints);

    const std::vector<Joint> &joints() const { return m_joints; }
    const SceneObject        &jointObject(JointIndex index) const;

    void evaluate(const glm::mat4 &meshWorldMatrix);

    const std::vector<glm::mat4> &jointMatrices() const { return m_jointMatrices; }

private:
    void validate() const;

    Scene                  *m_scene;
    std::vector<Joint>      m_joints;
    std::vector<glm::mat4> m_jointMatrices;
};

} // namespace lr
