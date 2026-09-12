#pragma once

#include "features/linear_blend_skinning/Joint.hpp"

#include <vector>

namespace lr
{

class Skin
{
public:
    Skin(std::vector<SkeletonNode> nodes, std::vector<Joint> joints);

    const std::vector<SkeletonNode> &nodes() const { return m_nodes; }
    const std::vector<Joint>        &joints() const { return m_joints; }

    const SkeletonNode &node(SkeletonNodeIndex index) const;
    void                setNodeTransform(SkeletonNodeIndex index, Transform transform);

private:
    void validate() const;

    std::vector<SkeletonNode> m_nodes;
    std::vector<Joint>        m_joints;
};

} // namespace lr
