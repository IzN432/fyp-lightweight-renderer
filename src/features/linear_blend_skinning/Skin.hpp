#pragma once

#include "features/linear_blend_skinning/Joint.hpp"
#include "core/utility/Uuid.hpp"

#include <vector>

namespace lr
{

class Scene;
class SceneObject;
class SkinStore;

using SkinId = Uuid;
using SkinHandle = uint32_t;

// Immutable binding shared by every mesh instance that references the same authored skin. Joint
// palette order and inverse-bind matrices are binding data; evaluated matrices are not.
class SkinBinding
{
public:
    SkinBinding(Scene &scene, std::vector<Joint> joints, SkinId id = generateUuid());

    SkinId id() const { return m_id; }
    const std::vector<Joint> &joints() const { return m_joints; }
    const SceneObject &jointObject(JointIndex index) const;

private:
    // Checks that all scene objects referenced by the joints are still in the scene
    void validate() const;

    Scene              *m_scene;
    SkinId              m_id;
    std::vector<Joint>  m_joints;
};

// Per-mesh skin instance. It shares immutable binding data while retaining the evaluated palette
// and enabled flag that depend on the particular mesh object using it.
class Skin
{
public:
    Skin(SkinStore &store, SkinHandle handle);

    SkinHandle handle() const { return m_handle; }
    SkinId id() const;
    const SkinBinding &binding() const;
    const std::vector<Joint> &joints() const;
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
    SkinStore             *m_store;
    SkinHandle             m_handle;
    std::vector<glm::mat4> m_jointMatrices;
    bool                   m_skinningEnabled = true;
};

} // namespace lr
