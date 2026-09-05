#pragma once

#include "Handles.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <stdexcept>
#include <unordered_map>

namespace lr
{

struct ExternalImageBinding
{
    VkImage     image = VK_NULL_HANDLE;
    VkImageView view  = VK_NULL_HANDLE;
};

// Physical images supplied for one execution of a compiled graph. Handles are
// graph-owned; native Vulkan objects remain private to the C++ execution layer.
class ExternalImageBindings
{
public:
    void bind(ImageHandle handle, VkImage image, VkImageView view)
    {
        if (!handle)
        {
            throw std::invalid_argument("FrameGraph: cannot bind an invalid external image handle");
        }
        if (image == VK_NULL_HANDLE || view == VK_NULL_HANDLE)
        {
            throw std::invalid_argument("FrameGraph: external image binding requires an image and image view");
        }
        if (m_owner == 0)
        {
            m_owner = handle.owner;
        } else if (m_owner != handle.owner)
        {
            throw std::invalid_argument("FrameGraph: external image bindings mix handles from different graphs");
        }
        m_images[handle.index] = {image, view};
    }

    const ExternalImageBinding *find(ImageHandle handle) const
    {
        if (m_owner != 0 && m_owner != handle.owner)
        {
            throw std::invalid_argument("FrameGraph: external image handle belongs to another graph");
        }
        const auto found = m_images.find(handle.index);
        return found == m_images.end() ? nullptr : &found->second;
    }

private:
    uint64_t                                           m_owner = 0;
    std::unordered_map<uint32_t, ExternalImageBinding> m_images;
};

} // namespace lr
