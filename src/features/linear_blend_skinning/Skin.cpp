#include "features/linear_blend_skinning/Skin.hpp"

#include <stdexcept>
#include <utility>

namespace lr
{

Skin::Skin(std::vector<SkeletonNode> nodes, std::vector<Joint> joints)
    : m_nodes(std::move(nodes)), m_joints(std::move(joints))
{
    validate();
    buildEvaluationOrder();

    m_defaultLocalTransforms.reserve(m_nodes.size());
    for (const SkeletonNode &node : m_nodes)
    {
        m_defaultLocalTransforms.push_back(node.localTransform);
    }

    m_nodeWorldMatrices.resize(m_nodes.size(), glm::mat4(1.0f));
    m_jointMatrices.resize(m_joints.size(), glm::mat4(1.0f));
}

const SkeletonNode &Skin::node(SkeletonNodeIndex index) const
{
    if (index >= m_nodes.size())
    {
        throw std::out_of_range("Skin node index is out of range");
    }
    return m_nodes[index];
}

void Skin::setNodeTransform(SkeletonNodeIndex index, Transform transform)
{
    if (index >= m_nodes.size())
    {
        throw std::out_of_range("Skin node index is out of range");
    }
    m_nodes[index].localTransform = std::move(transform);
}

void Skin::resetPose()
{
    for (SkeletonNodeIndex index = 0; index < m_nodes.size(); ++index)
    {
        m_nodes[index].localTransform = m_defaultLocalTransforms[index];
    }
}

void Skin::evaluate(const glm::mat4 &meshWorldMatrix)
{
    for (SkeletonNodeIndex nodeIndex : m_evaluationOrder)
    {
        const SkeletonNode &currentNode = m_nodes[nodeIndex];
        const glm::mat4 localMatrix = currentNode.localTransform.localMatrix();

        if (currentNode.parent)
        {
            m_nodeWorldMatrices[nodeIndex] =
                m_nodeWorldMatrices[currentNode.parent.value()] * localMatrix;
        }
        else
        {
            m_nodeWorldMatrices[nodeIndex] = localMatrix;
        }
    }

    const glm::mat4 inverseMeshWorld = glm::inverse(meshWorldMatrix);
    for (JointIndex jointIndex = 0; jointIndex < m_joints.size(); ++jointIndex)
    {
        const Joint &joint = m_joints[jointIndex];
        m_jointMatrices[jointIndex] =
            inverseMeshWorld * m_nodeWorldMatrices[joint.node] * joint.inverseBindMatrix;
    }
}

void Skin::buildEvaluationOrder()
{
    m_evaluationOrder.clear();
    m_evaluationOrder.reserve(m_nodes.size());

    std::vector<bool> added(m_nodes.size(), false);
    for (SkeletonNodeIndex nodeIndex = 0; nodeIndex < m_nodes.size(); ++nodeIndex)
    {
        std::vector<SkeletonNodeIndex> chain;
        SkeletonNodeIndex              current = nodeIndex;

        while (!added[current])
        {
            chain.push_back(current);
            if (!m_nodes[current].parent)
            {
                break;
            }
            current = m_nodes[current].parent.value();
        }

        for (auto it = chain.rbegin(); it != chain.rend(); ++it)
        {
            m_evaluationOrder.push_back(*it);
            added[*it] = true;
            
        }
    }
}

void Skin::validate() const
{
    for (const Joint &joint : m_joints)
    {
        if (joint.node >= m_nodes.size())
        {
            throw std::invalid_argument("Skin joint references an invalid node");
        }
    }

    for (SkeletonNodeIndex nodeIndex = 0; nodeIndex < m_nodes.size(); ++nodeIndex)
    {
        auto   ancestor  = m_nodes[nodeIndex].parent;
        size_t remaining = m_nodes.size();

        while (ancestor)
        {
            if (ancestor.value() >= m_nodes.size())
            {
                throw std::invalid_argument("Skin node references an invalid parent");
            }
            if (remaining-- == 0)
            {
                throw std::invalid_argument("Skin node hierarchy contains a cycle");
            }
            ancestor = m_nodes[ancestor.value()].parent;
        }
    }
}

} // namespace lr
