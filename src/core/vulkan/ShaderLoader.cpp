#include "ShaderLoader.hpp"
#include "VkResultUtils.hpp"

#include <spdlog/spdlog.h>

#include <fstream>
#include <stdexcept>

namespace lr
{

std::vector<uint32_t> readSpirvFile(const std::filesystem::path &spvPath)
{
    std::ifstream file(spvPath, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        throw std::runtime_error("ShaderModule: could not open " + spvPath.string());
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    if (fileSize % sizeof(uint32_t) != 0)
    {
        throw std::runtime_error("ShaderModule: .spv size is not a multiple of 4: " + spvPath.string());
    }

    std::vector<uint32_t> code(fileSize / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char *>(code.data()), static_cast<std::streamsize>(fileSize));
    return code;
}

ShaderModule::ShaderModule(VkDevice device, const ShaderCode &code) : m_device(device)
{
    if (code.empty())
    {
        throw std::runtime_error("ShaderModule: no shader code given");
    }

    const std::vector<uint32_t>  fileWords = code.spirv.empty() ? readSpirvFile(code.path) : std::vector<uint32_t>{};
    const std::vector<uint32_t> &words     = code.spirv.empty() ? fileWords : code.spirv;

    VkShaderModuleCreateInfo ci{};
    ci.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = words.size() * sizeof(uint32_t);
    ci.pCode    = words.data();

    checkVk(vkCreateShaderModule(m_device, &ci, nullptr, &m_module),
            "ShaderModule: vkCreateShaderModule for " + code.label());

    spdlog::debug("ShaderModule: loaded {}", code.label());
}

ShaderModule::~ShaderModule() { vkDestroyShaderModule(m_device, m_module, nullptr); }

} // namespace lr
