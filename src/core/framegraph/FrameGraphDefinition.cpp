#include "FrameGraphDefinition.hpp"

#include <stdexcept>

namespace lr
{

PassHandle FrameGraphDefinition::addPass(std::string name)
{
    if (name.empty())
    {
        throw std::invalid_argument("FrameGraph: pass name cannot be empty");
    }
    for (const PassDesc &pass : m_passes)
    {
        if (pass.name == name)
        {
            throw std::runtime_error("FrameGraph: duplicate pass name '" + name + "'");
        }
    }

    const PassHandle handle{static_cast<uint32_t>(m_passes.size()), owner()};
    m_passes.push_back({.name = std::move(name), .handle = handle});
    return handle;
}

PassDesc &FrameGraphDefinition::pass(PassHandle handle)
{
    if (handle.owner != owner())
    {
        throw std::invalid_argument("FrameGraph: pass handle belongs to another graph");
    }
    if (handle.index >= m_passes.size())
    {
        throw std::out_of_range("FrameGraph: invalid pass handle");
    }
    return m_passes[handle.index];
}

const PassDesc &FrameGraphDefinition::pass(PassHandle handle) const
{
    return const_cast<FrameGraphDefinition &>(*this).pass(handle);
}

std::vector<PassHandle> FrameGraphDefinition::passHandles() const
{
    std::vector<PassHandle> handles;
    handles.reserve(m_passes.size());
    for (const PassDesc &pass : m_passes)
    {
        handles.push_back(pass.handle);
    }
    return handles;
}

} // namespace lr
