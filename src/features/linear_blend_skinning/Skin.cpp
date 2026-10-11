#include "features/linear_blend_skinning/Skin.hpp"
#include "features/linear_blend_skinning/SkinStore.hpp"

#include "core/scene/Scene.hpp"

#include <stdexcept>
#include <utility>

namespace lr
{

SkinBinding::SkinBinding(Scene &scene, std::vector<Joint> joints, SkinId id)
    : m_scene(&scene), m_id(id), m_joints(std::move(joints))
{
    if (m_id.is_nil())
    {
        throw std::invalid_argument("Skin definition ID cannot be nil");
    }
    validate();
}

const SceneObject &SkinBinding::jointObject(JointIndex index) const
{
    if (index >= m_joints.size())
    {
        throw std::out_of_range("Skin joint index is out of range");
    }
    return m_scene->getSceneObject(m_joints[index].sceneObject);
}

Skin::Skin(SkinStore &store, SkinHandle handle) : m_store(&store), m_handle(handle)
{
    m_jointMatrices.resize(binding().joints().size(), glm::mat4(1.0f));
}

SkinId Skin::id() const { return m_store->idOf(m_handle); }
const SkinBinding &Skin::binding() const { return m_store->get(m_handle); }
const std::vector<Joint> &Skin::joints() const { return binding().joints(); }
const SceneObject &Skin::jointObject(JointIndex index) const { return binding().jointObject(index); }

void Skin::evaluate(const glm::mat4 &meshWorldMatrix)
{
    const glm::mat4 inverseMeshWorld = glm::inverse(meshWorldMatrix);
    for (JointIndex jointIndex = 0; jointIndex < joints().size(); ++jointIndex)
    {
        const Joint &joint = joints()[jointIndex];
        m_jointMatrices[jointIndex] =
            inverseMeshWorld * jointObject(jointIndex).worldMatrix() * joint.inverseBindMatrix;
    }
}

void SkinBinding::validate() const
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
