#pragma once

#include "PassDefinition.hpp"

#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace lr
{

// Owns the declarative frontend of a frame graph. Pass and resource handles are
// valid only within this definition; runtime Vulkan resources live elsewhere.
class FrameGraphDefinition
{
public:
    struct ExternalImageDesc
    {
        ImageHandle image;
        VkFormat    format = VK_FORMAT_UNDEFINED;
        ExtentSpec  extent = ExtentSpec::swapchain();
    };

    FrameGraphDefinition() = default;

    // Definitions are value-like so compilation can retain an immutable
    // snapshot independent of subsequent frontend edits.
    FrameGraphDefinition(const FrameGraphDefinition &)            = default;
    FrameGraphDefinition &operator=(const FrameGraphDefinition &) = default;
    FrameGraphDefinition(FrameGraphDefinition &&)                 = default;
    FrameGraphDefinition &operator=(FrameGraphDefinition &&)      = default;

    PassHandle  addPass(std::string name);
    ImageHandle importBackbuffer(std::string_view name, VkFormat format);

    ImageHandle        image(std::string_view name) { return m_resources.image(name); }
    BufferHandle       buffer(std::string_view name) { return m_resources.buffer(name); }
    const std::string &name(ImageHandle handle) const { return m_resources.name(handle); }
    const std::string &name(BufferHandle handle) const { return m_resources.name(handle); }

    // Mutable access counts as a change (see revision()); the const overloads don't.
    PassDesc       &pass(PassHandle handle);
    const PassDesc &pass(PassHandle handle) const;

    std::span<PassDesc> passes()
    {
        ++m_revision;
        return m_passes;
    }
    std::span<const PassDesc> passes() const { return m_passes; }

    std::vector<PassHandle>            passHandles() const;
    std::span<const ExternalImageDesc> externalImages() const { return m_externalImages; }

    uint64_t owner() const { return m_resources.owner(); }

    // Increases whenever passes may have changed (added, or handed out for modification), so a
    // compiled graph can tell it is out of date.
    uint64_t                      revision() const { return m_revision; }
    const ResourceHandleRegistry &resources() const { return m_resources; }

private:
    size_t checkedIndex(PassHandle handle) const;

    ResourceHandleRegistry         m_resources;
    std::vector<PassDesc>          m_passes;
    std::vector<ExternalImageDesc> m_externalImages;
    uint64_t                       m_revision = 0;
};

} // namespace lr
