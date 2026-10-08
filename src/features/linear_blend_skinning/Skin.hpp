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

    // Per-skin override of vertex skinning, read live by every geometry-replaying pass through
    // SkinDrawInfo. Disabling it draws the mesh's stored bind-pose positions through its model
    // matrix, the same way a mesh with no skin at all is drawn. The joint palette keeps being
    // evaluated either way, so re-enabling needs no rebuild.
    bool skinningEnabled() const { return m_skinningEnabled; }
    void setSkinningEnabled(bool enabled) { m_skinningEnabled = enabled; }

private:
    void validate() const;

    Scene                  *m_scene;
    std::vector<Joint>      m_joints;
    std::vector<glm::mat4> m_jointMatrices;
    bool                   m_skinningEnabled = true;
};

} // namespace lr
