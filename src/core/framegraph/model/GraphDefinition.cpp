#include "GraphDefinition.hpp"

#include <stdexcept>

namespace lr::framegraph
{

PassId GraphDefinition::addPass(std::string name, PassKind kind)
{
    const PassId id{static_cast<uint32_t>(m_passes.size())};
    const auto [it, inserted] = m_passIds.emplace(name, id);
    if (!inserted)
    {
        throw std::runtime_error("FrameGraph: duplicate pass name '" + name + "'");
    }

    m_passes.push_back({.id = id, .name = std::move(name), .kind = kind});
    return id;
}

ResourceId GraphDefinition::addResource(std::string name, ResourceKind kind)
{
    const auto existing = m_resourceIds.find(name);
    if (existing != m_resourceIds.end())
    {
        ResourceNode &node = m_resources[existing->second.value];
        if (node.kind == ResourceKind::Unknown)
        {
            node.kind = kind;
        } else if (kind != ResourceKind::Unknown && node.kind != kind)
        {
            throw std::runtime_error("FrameGraph: resource '" + name + "' is declared as both an image and a buffer");
        }
        return existing->second;
    }

    const ResourceId id{static_cast<uint32_t>(m_resources.size())};
    m_resources.push_back({.id = id, .name = std::move(name), .kind = kind});
    m_resourceIds.emplace(m_resources.back().name, id);
    return id;
}

void GraphDefinition::addAccess(PassId passId, ResourceAccess access)
{
    if (access.resource.value >= m_resources.size())
    {
        throw std::out_of_range("FrameGraph IR: invalid resource id");
    }
    pass(passId);
    m_passes[passId.value].accesses.push_back(access);
}

void GraphDefinition::addDependency(PassId passId, PassId dependency)
{
    pass(passId);
    pass(dependency);
    m_passes[passId.value].explicitDependencies.push_back(dependency);
}

const ResourceNode &GraphDefinition::resource(ResourceId id) const
{
    if (id.value >= m_resources.size())
    {
        throw std::out_of_range("FrameGraph IR: invalid resource id");
    }
    return m_resources[id.value];
}

const PassNode &GraphDefinition::pass(PassId id) const
{
    if (id.value >= m_passes.size())
    {
        throw std::out_of_range("FrameGraph IR: invalid pass id");
    }
    return m_passes[id.value];
}

ResourceId GraphDefinition::findResource(std::string_view name) const
{
    const auto it = m_resourceIds.find(std::string(name));
    return it == m_resourceIds.end() ? ResourceId{} : it->second;
}

PassId GraphDefinition::findPass(std::string_view name) const
{
    const auto it = m_passIds.find(std::string(name));
    return it == m_passIds.end() ? PassId{} : it->second;
}

} // namespace lr::framegraph
