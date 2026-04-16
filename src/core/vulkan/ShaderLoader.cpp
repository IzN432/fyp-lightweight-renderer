#include "ShaderLoader.hpp"

#include <spdlog/spdlog.h>

#include <fstream>
#include <stdexcept>
#include <vector>

namespace lr
{

ShaderModule::ShaderModule(VkDevice device, const std::filesystem::path &spvPath)
    : m_device(device)
{
    std::ifstream file(spvPath, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        spdlog::error("Runtime error: throwing std::runtime_error");
        throw std::runtime_error("ShaderModule: could not open " + spvPath.string());
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    if (fileSize % sizeof(uint32_t) != 0)
    {
        spdlog::error("Runtime error: throwing std::runtime_error");
        throw std::runtime_error("ShaderModule: .spv size is not a multiple of 4: " + spvPath.string());
    }

    std::vector<uint32_t> code(fileSize / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char *>(code.data()), static_cast<std::streamsize>(fileSize));

    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = fileSize;
    ci.pCode = code.data();

    if (vkCreateShaderModule(m_device, &ci, nullptr, &m_module) != VK_SUCCESS)
    {
        spdlog::error("Runtime error: throwing std::runtime_error");
        throw std::runtime_error("ShaderModule: failed to create shader module for " + spvPath.string());
    }

    spdlog::debug("ShaderModule: loaded {}", spvPath.string());
}

ShaderModule::~ShaderModule()
{
    vkDestroyShaderModule(m_device, m_module, nullptr);
}

}  // namespace lr
