#pragma once

#include "ResourceRegistry.hpp"
#include "core/vulkan/Allocator.hpp"
#include "core/vulkan/VulkanContext.hpp"

#include <vulkan/vulkan.h>

#include <string>
#include <vector>

namespace lr
{

class ImageReadback
{
public:
    ImageReadback(const VulkanContext &ctx, Allocator &alloc);
    ~ImageReadback();

    ImageReadback(const ImageReadback &) = delete;
    ImageReadback &operator=(const ImageReadback &) = delete;

    static constexpr uint32_t kNoData = 0xFFFFFFFFu;

    // Read one uint32_t pixel. Convenience wrapper around readRect.
    uint32_t readPixel(const ResourceRegistry &resources,
                       const std::string &imageName,
                       uint32_t x, uint32_t y);

    // Read a rectangular region in one GPU submission (row-major, uint32_t per pixel).
    // Returns an empty vector if the image is not registered.
    std::vector<uint32_t> readRect(const ResourceRegistry &resources,
                                   const std::string &imageName,
                                   uint32_t x, uint32_t y,
                                   uint32_t width, uint32_t height);

private:
    void ensureStagingCapacity(uint32_t pixelCount);

    const VulkanContext &m_ctx;
    Allocator           &m_alloc;
    VkCommandPool        m_commandPool  = VK_NULL_HANDLE;
    AllocatedBuffer      m_staging{};
    uint32_t             m_stagingPixels = 0;
};

}  // namespace lr
