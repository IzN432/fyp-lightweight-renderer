#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace lr
{

// One shader stage's SPIR-V — either a .spv file read when the pipeline is created, or words already
// held in memory (e.g. produced at runtime by compileGlslFile/compileGlslSource, see ShaderCompiler.hpp).
// At most one of the two is set; an empty ShaderCode means "no shader for this stage".
struct ShaderCode
{
    std::string           path;
    std::vector<uint32_t> spirv;

    static ShaderCode fromFile(std::string spvPath) { return {.path = std::move(spvPath)}; }
    static ShaderCode fromSpirv(std::vector<uint32_t> words) { return {.spirv = std::move(words)}; }

    bool        empty() const { return path.empty() && spirv.empty(); }
    std::string label() const { return path.empty() ? std::string("<in-memory SPIR-V>") : path; }
};

std::vector<uint32_t> readSpirvFile(const std::filesystem::path &spvPath);

class ShaderModule
{
public:
    ShaderModule(VkDevice device, const ShaderCode &code);
    ~ShaderModule();

    ShaderModule(const ShaderModule &)            = delete;
    ShaderModule &operator=(const ShaderModule &) = delete;
    ShaderModule(ShaderModule &&)                 = delete;
    ShaderModule &operator=(ShaderModule &&)      = delete;

    VkShaderModule get() const { return m_module; }

private:
    VkDevice       m_device = VK_NULL_HANDLE;
    VkShaderModule m_module = VK_NULL_HANDLE;
};

} // namespace lr
