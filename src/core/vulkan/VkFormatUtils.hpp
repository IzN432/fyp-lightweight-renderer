#pragma once

#include <vulkan/vulkan.h>

#include <spdlog/spdlog.h>
#include <cstdint>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <spdlog/spdlog.h>
#include <string>

#include <spdlog/spdlog.h>
namespace lr
{

// Byte size of one texel/element for common vertex attribute VkFormat values.
inline uint32_t vkFormatByteSize(VkFormat fmt)
{
    switch (fmt)
    {
    case VK_FORMAT_R32_SFLOAT:
    case VK_FORMAT_R32_UINT:
    case VK_FORMAT_R32_SINT:             return 4;
    case VK_FORMAT_R32G32_SFLOAT:
    case VK_FORMAT_R32G32_UINT:
    case VK_FORMAT_R32G32_SINT:          return 8;
    case VK_FORMAT_R32G32B32_SFLOAT:
    case VK_FORMAT_R32G32B32_UINT:
    case VK_FORMAT_R32G32B32_SINT:       return 12;
    case VK_FORMAT_R32G32B32A32_SFLOAT:
    case VK_FORMAT_R32G32B32A32_UINT:
    case VK_FORMAT_R32G32B32A32_SINT:    return 16;
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R8G8B8A8_UINT:        return 4;
    case VK_FORMAT_R16G16_SFLOAT:        return 4;
    case VK_FORMAT_R16G16B16A16_SFLOAT:  return 8;
    default:
        throw std::runtime_error(
            "vkFormatByteSize: unsupported format " + std::to_string(static_cast<int>(fmt)));
    }
}

} // namespace lr
