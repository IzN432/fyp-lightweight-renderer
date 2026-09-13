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
    void                resetPose();
    void                evaluate(const glm::mat4 &meshWorldMatrix);

    const std::vector<glm::mat4> &nodeWorldMatrices() const { return m_nodeWorldMatrices; }
    const std::vector<glm::mat4> &jointMatrices() const { return m_jointMatrices; }

private:
    void validate() const;
    void buildEvaluationOrder();

    std::vector<SkeletonNode>      m_nodes;
    std::vector<Joint>             m_joints;
    std::vector<Transform>         m_defaultLocalTransforms;
    std::vector<SkeletonNodeIndex> m_evaluationOrder;
    std::vector<glm::mat4>         m_nodeWorldMatrices;
    std::vector<glm::mat4>         m_jointMatrices;
};

} // namespace lr
