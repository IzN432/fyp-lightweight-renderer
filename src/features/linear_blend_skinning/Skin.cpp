#include "features/linear_blend_skinning/Skin.hpp"

#include "core/scene/Scene.hpp"

#include <stdexcept>
#include <utility>

namespace lr
{

Skin::Skin(Scene &scene, std::vector<Joint> joints) : m_scene(&scene), m_joints(std::move(joints))
{
    validate();
    m_jointMatrices.resize(m_joints.size(), glm::mat4(1.0f));
}

const SceneObject &Skin::jointObject(JointIndex index) const
{
    if (index >= m_joints.size())
    {
        throw std::out_of_range("Skin joint index is out of range");
    }
    return m_scene->getSceneObject(m_joints[index].sceneObject);
}

void Skin::evaluate(const glm::mat4 &meshWorldMatrix)
{
    const glm::mat4 inverseMeshWorld = glm::inverse(meshWorldMatrix);
    for (JointIndex jointIndex = 0; jointIndex < m_joints.size(); ++jointIndex)
    {
        const Joint &joint = m_joints[jointIndex];
        m_jointMatrices[jointIndex] =
            inverseMeshWorld * jointObject(jointIndex).worldMatrix() * joint.inverseBindMatrix;
    }
}

void Skin::validate() const
{
    for (const Joint &joint : m_joints)
    {
        try
        {
            m_scene->getSceneObject(joint.sceneObject);
        }
        catch (const std::out_of_range &)
        {
            throw std::invalid_argument("Skin joint references an invalid scene object");
        }
    }
}

} // namespace lr
