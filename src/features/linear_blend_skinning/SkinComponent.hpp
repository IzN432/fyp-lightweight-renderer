#pragma once

#include "core/scene/Component.hpp"
#include "features/linear_blend_skinning/Skin.hpp"

#include <utility>

namespace lr
{

class SkinComponent : public Component
{
public:
    explicit SkinComponent(Skin skin) : Component("SkinComponent"), m_skin(std::move(skin)) {}

    const Skin &skin() const { return m_skin; }

    void setNodeTransform(SkeletonNodeIndex index, Transform transform)
    {
        m_skin.setNodeTransform(index, std::move(transform));
        markDirty();
    }

    void resetPose()
    {
        m_skin.resetPose();
        markDirty();
    }

    void evaluate(const glm::mat4 &meshWorldMatrix) { m_skin.evaluate(meshWorldMatrix); }

private:
    Skin m_skin;
};

} // namespace lr
