#pragma once

#include "PassBuilder.hpp"

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

    FrameGraphDefinition(const FrameGraphDefinition &)            = delete;
    FrameGraphDefinition &operator=(const FrameGraphDefinition &) = delete;

    PassHandle addPass(std::string name);

    ImageHandle  image(std::string_view name) { return m_resources.image(name); }
    BufferHandle buffer(std::string_view name) { return m_resources.buffer(name); }

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
