#include "ShaderInterface.hpp"

#include "core/vulkan/ShaderLoader.hpp"

#include <spirv-reflect/spirv_reflect.h>

#include <algorithm>
#include <map>
#include <string>
#include <vector>

namespace lr
{

namespace
{

// RAII wrapper over SpvReflectShaderModule.
class ReflectedShader
{
public:
    ReflectedShader(const std::vector<uint32_t> &words, const std::string &label)
    {
        if (spvReflectCreateShaderModule(words.size() * sizeof(uint32_t), words.data(), &m_module) !=
            SPV_REFLECT_RESULT_SUCCESS)
        {
            throw ShaderInterfaceError("FrameGraph: could not reflect the SPIR-V of " + label);
        }
    }
    ~ReflectedShader() { spvReflectDestroyShaderModule(&m_module); }

    ReflectedShader(const ReflectedShader &)            = delete;
    ReflectedShader &operator=(const ReflectedShader &) = delete;

    template <typename T, typename Enumerate> std::vector<T *> enumerate(Enumerate enumerateFn) const
    {
        uint32_t count = 0;
        enumerateFn(&m_module, &count, nullptr);
        std::vector<T *> items(count);
        enumerateFn(&m_module, &count, items.data());
        return items;
    }

private:
    SpvReflectShaderModule m_module{};
};

std::string stageNames(VkShaderStageFlags stages)
{
    std::string names;
    const auto  add = [&](VkShaderStageFlags bit, const char *name) {
        if (stages & bit)
        {
            names += (names.empty() ? "" : "+") + std::string(name);
        }
    };
    add(VK_SHADER_STAGE_VERTEX_BIT, "vertex");
    add(VK_SHADER_STAGE_FRAGMENT_BIT, "fragment");
    add(VK_SHADER_STAGE_COMPUTE_BIT, "compute");
    return names.empty() ? "no stages" : names;
}

std::string descriptorName(VkDescriptorType type)
{
    switch (type)
    {
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
            return "sampled image";
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            return "storage image";
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            return "uniform buffer";
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            return "storage buffer";
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            return "separate texture (unsupported: use a combined sampler2D)";
        case VK_DESCRIPTOR_TYPE_SAMPLER:
            return "separate sampler (unsupported: use a combined sampler2D)";
        default:
            return "descriptor type " + std::to_string(static_cast<int>(type));
    }
}

struct DeclaredDescriptor
{
    VkDescriptorType   type;
    VkShaderStageFlags stages;
};

// Same mapping FrameGraphCompiler::buildDescriptorSets uses to build the set layout.
std::map<uint32_t, DeclaredDescriptor> declaredDescriptors(const PassDesc &pass)
{
    std::map<uint32_t, DeclaredDescriptor> declared;
    for (const ImageUse &use : pass.imageUses)
    {
        if (use.isDescriptor())
        {
            declared[use.binding] = {use.usage == ImageUsage::Storage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                                                      : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                     use.stages};
        }
    }
    for (const BufferUse &use : pass.bufferUses)
    {
        if (use.isDescriptor())
        {
            declared[use.binding] = {use.usage == BufferUsage::Uniform ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                                                       : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                     use.stages};
        }
    }
    return declared;
}

struct UsedDescriptor
{
    VkDescriptorType   type;
    VkShaderStageFlags stages = 0;
    std::string        name;
};

// End of the bytes a push-constant block covers. Member offsets are absolute — a stage's block can
// start past 0 (`layout(offset = 64)`) to share one range with another stage — while the block's own
// `offset` is just its lowest member's, so offset + size would count that gap twice.
uint32_t pushConstantEnd(const SpvReflectBlockVariable &block)
{
    uint32_t end = 0;
    for (uint32_t i = 0; i < block.member_count; ++i)
    {
        end = std::max(end, block.members[i].offset + block.members[i].size);
    }
    return block.member_count > 0 ? end : block.size;
}

} // namespace

void validateShaderInterface(const PassDesc &pass)
{
    if (pass.type == PassType::Custom)
    {
        return;
    }

    struct Stage
    {
        const ShaderCode     &code;
        VkShaderStageFlagBits bit;
    };
    std::vector<Stage> stages;
    if (pass.type == PassType::Compute)
    {
        stages.push_back({pass.computeShader, VK_SHADER_STAGE_COMPUTE_BIT});
    } else
    {
        stages.push_back({pass.vertShader, VK_SHADER_STAGE_VERTEX_BIT});
        stages.push_back({pass.fragShader, VK_SHADER_STAGE_FRAGMENT_BIT});
    }

    std::vector<std::string>           problems;
    std::map<uint32_t, UsedDescriptor> used;
    uint32_t                           pushEnd    = 0;
    VkShaderStageFlags                 pushStages = 0;

    for (const Stage &stage : stages)
    {
        if (stage.code.empty())
        {
            continue; // FrameGraphCompiler reports missing shaders itself.
        }
        const std::string     label = stage.code.label();
        const ReflectedShader shader(stage.code.spirv.empty() ? readSpirvFile(stage.code.path) : stage.code.spirv,
                                     label);

        for (const SpvReflectDescriptorBinding *binding :
             shader.enumerate<SpvReflectDescriptorBinding>(spvReflectEnumerateDescriptorBindings))
        {
            if (!binding->accessed)
            {
                continue;
            }
            const std::string name = binding->name ? binding->name : "";
            if (binding->set != 0)
            {
                problems.push_back(label + " uses set " + std::to_string(binding->set) + " (binding " +
                                   std::to_string(binding->binding) + " '" + name +
                                   "'), but passes have a single descriptor set: use set = 0");
                continue;
            }
            const auto type         = static_cast<VkDescriptorType>(binding->descriptor_type);
            auto [entry, firstSeen] = used.try_emplace(binding->binding, UsedDescriptor{type, 0, name});
            if (!firstSeen && entry->second.type != type)
            {
                problems.push_back("binding " + std::to_string(binding->binding) +
                                   ": the shaders disagree on its type (" + descriptorName(entry->second.type) +
                                   " vs " + descriptorName(type) + ")");
            }
            entry->second.stages |= stage.bit;
        }

        for (const SpvReflectBlockVariable *block :
             shader.enumerate<SpvReflectBlockVariable>(spvReflectEnumeratePushConstantBlocks))
        {
            if (block->flags & SPV_REFLECT_VARIABLE_FLAGS_UNUSED)
            {
                continue;
            }
            pushEnd = std::max(pushEnd, pushConstantEnd(*block));
            pushStages |= stage.bit;
        }

        if (stage.bit == VK_SHADER_STAGE_VERTEX_BIT)
        {
            for (const SpvReflectInterfaceVariable *input :
                 shader.enumerate<SpvReflectInterfaceVariable>(spvReflectEnumerateInputVariables))
            {
                if (input->built_in != -1)
                {
                    continue;
                }
                const std::string where = label + " reads vertex input location " + std::to_string(input->location) +
                                          (input->name ? std::string(" '") + input->name + "'" : std::string());
                if (pass.type == PassType::Fullscreen)
                {
                    problems.push_back(where + ", but fullscreen passes have no vertex input");
                } else if (std::ranges::none_of(pass.vertexAttributes, [&](const VkVertexInputAttributeDescription &a) {
                               return a.location == input->location;
                           }))
                {
                    problems.push_back(where + ", but the vertex layout has no attribute at that location");
                }
            }
        }
    }

    const std::map<uint32_t, DeclaredDescriptor> declared = declaredDescriptors(pass);
    for (const auto &[binding, use] : used)
    {
        const std::string what = "binding " + std::to_string(binding) + " ('" + use.name + "', " +
                                 descriptorName(use.type) + ", " + stageNames(use.stages) + ")";
        const auto        decl = declared.find(binding);
        if (decl == declared.end())
        {
            problems.push_back(what + " is used by the shaders but not declared by the pass");
        } else if (decl->second.type != use.type)
        {
            problems.push_back(what + " is declared as a " + descriptorName(decl->second.type));
        } else if ((use.stages & ~decl->second.stages) != 0)
        {
            problems.push_back(what + " is declared only for " + stageNames(decl->second.stages));
        }
    }

    if (pushEnd > pass.pushConstantSize)
    {
        problems.push_back("push constants: the shaders use " + std::to_string(pushEnd) +
                           " bytes, but push_constant_size is " + std::to_string(pass.pushConstantSize));
    } else if (pushEnd > 0 && (pushStages & ~pass.pushConstantStages) != 0)
    {
        problems.push_back("push constants are used by " + stageNames(pushStages) + " but declared only for " +
                           stageNames(pass.pushConstantStages));
    }

    if (problems.empty())
    {
        return;
    }

    // Declarations no shader uses are harmless alone, but next to a mismatch they're usually the
    // other half of it (e.g. a binding number off by one).
    std::string unused;
    for (const auto &[binding, decl] : declared)
    {
        if (!used.contains(binding))
        {
            unused += (unused.empty() ? "" : ", ") + std::to_string(binding) + " (" + descriptorName(decl.type) + ")";
        }
    }

    std::string message = "FrameGraph: pass '" + pass.name + "' doesn't match its shaders:";
    for (const std::string &problem : problems)
    {
        message += "\n  - " + problem;
    }
    if (!unused.empty())
    {
        message += "\n  note: the pass also declares bindings no shader uses: " + unused;
    }
    throw ShaderInterfaceError(message);
}

} // namespace lr
