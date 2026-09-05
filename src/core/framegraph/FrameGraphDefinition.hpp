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
    FrameGraphDefinition() = default;

    // Definitions are value-like so compilation can retain an immutable
    // snapshot independent of subsequent frontend edits.
    FrameGraphDefinition(const FrameGraphDefinition &)            = default;
    FrameGraphDefinition &operator=(const FrameGraphDefinition &) = default;
    FrameGraphDefinition(FrameGraphDefinition &&)                 = default;
    FrameGraphDefinition &operator=(FrameGraphDefinition &&)      = default;

    PassHandle addPass(std::string name);

    ImageHandle        image(std::string_view name) { return m_resources.image(name); }
    BufferHandle       buffer(std::string_view name) { return m_resources.buffer(name); }
    const std::string &name(ImageHandle handle) const { return m_resources.name(handle); }
    const std::string &name(BufferHandle handle) const { return m_resources.name(handle); }

    PassDesc       &pass(PassHandle handle);
    const PassDesc &pass(PassHandle handle) const;

    std::span<PassDesc>       passes() { return m_passes; }
    std::span<const PassDesc> passes() const { return m_passes; }

    std::vector<PassHandle> passHandles() const;

    uint64_t                      owner() const { return m_resources.owner(); }
    const ResourceHandleRegistry &resources() const { return m_resources; }

private:
    ResourceHandleRegistry m_resources;
    std::vector<PassDesc>  m_passes;
};

} // namespace lr
