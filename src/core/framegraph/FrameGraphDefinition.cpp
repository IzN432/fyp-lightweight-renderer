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
    ++m_revision;
    return handle;
}

ImageHandle FrameGraphDefinition::importBackbuffer(std::string_view name, VkFormat format)
{
    if (format == VK_FORMAT_UNDEFINED)
    {
        throw std::invalid_argument("FrameGraph: backbuffer format cannot be undefined");
    }
    const ImageHandle handle = image(name);
    for (const ExternalImageDesc &external : m_externalImages)
    {
        if (external.image == handle)
        {
            throw std::runtime_error("FrameGraph: external image '" + this->name(handle) + "' is already declared");
        }
    }
    m_externalImages.push_back({.image = handle, .format = format});
    ++m_revision;
    return handle;
}

size_t FrameGraphDefinition::checkedIndex(PassHandle handle) const
{
    if (handle.owner != owner())
    {
        throw std::invalid_argument("FrameGraph: pass handle belongs to another graph");
    }
    if (handle.index >= m_passes.size())
    {
        throw std::out_of_range("FrameGraph: invalid pass handle");
    }
    return handle.index;
}

PassDesc &FrameGraphDefinition::pass(PassHandle handle)
{
    const size_t index = checkedIndex(handle);
    ++m_revision;
    return m_passes[index];
}

const PassDesc &FrameGraphDefinition::pass(PassHandle handle) const { return m_passes[checkedIndex(handle)]; }

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
