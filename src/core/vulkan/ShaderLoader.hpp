#pragma once

#include <vulkan/vulkan.h>

#include <filesystem>

namespace lr
{

class ShaderModule
{
public:
    ShaderModule(VkDevice device, const std::filesystem::path &spvPath);
    ~ShaderModule();

    ShaderModule(const ShaderModule &) = delete;
    ShaderModule &operator=(const ShaderModule &) = delete;
    ShaderModule(ShaderModule &&) = delete;
    ShaderModule &operator=(ShaderModule &&) = delete;

    VkShaderModule get() const { return m_module; }

private:
    VkDevice m_device = VK_NULL_HANDLE;
    VkShaderModule m_module = VK_NULL_HANDLE;
};

}  // namespace lr
