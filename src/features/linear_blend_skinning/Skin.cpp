#include "features/linear_blend_skinning/Skin.hpp"

#include <stdexcept>
#include <utility>

namespace lr
{

Skin::Skin(std::vector<SkeletonNode> nodes, std::vector<Joint> joints)
    : m_nodes(std::move(nodes)), m_joints(std::move(joints))
{
    validate();
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
