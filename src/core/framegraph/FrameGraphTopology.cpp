#include "FrameGraphTopology.hpp"
#include "PassDescAdapter.hpp"
#include "compiler/GraphCompiler.hpp"

#include <sstream>
#include <stdexcept>

namespace
{

bool isDepthFormat(VkFormat format)
{
    return format == VK_FORMAT_D32_SFLOAT || format == VK_FORMAT_D24_UNORM_S8_UINT ||
           format == VK_FORMAT_D16_UNORM || format == VK_FORMAT_D32_SFLOAT_S8_UINT;
}

const char *passTypeName(lr::PassType type)
{
    switch (type)
    {
        case lr::PassType::Geometry:
            return "geometry";
        case lr::PassType::Fullscreen:
            return "fullscreen";
        case lr::PassType::Compute:
            return "compute";
        case lr::PassType::Custom:
            return "custom";
    }
    return "unknown";
}

const char *accessName(lr::BindingAccess access)
{
    switch (access)
    {
        case lr::BindingAccess::Read:
            return "read";
        case lr::BindingAccess::Write:
            return "write";
        case lr::BindingAccess::ReadWrite:
            return "read-write";
    }
    return "unknown";
}

template <typename T>
void writeHex(std::ostringstream &out, T value)
{
    out << "0x" << std::hex << static_cast<uint64_t>(value) << std::dec;
}

} // namespace

namespace lr::framegraph
{

PassDesc &appendPass(std::vector<PassDesc> &passes, std::string name)
{
    for (const auto &pass : passes)
    {
        if (pass.name == name)
        {
            throw std::runtime_error("FrameGraph: duplicate pass name '" + name + "'");
        }
    }

    passes.push_back(PassDesc{.name = std::move(name)});
    return passes.back();
}

std::vector<size_t> sortPasses(std::span<const PassDesc> passes)
{
    const GraphDefinition graph = translatePassDescriptions(passes);
    const ExecutionPlan plan = buildLegacyExecutionPlan(graph);

    std::vector<size_t> sorted;
    sorted.reserve(plan.orderedPasses.size());
    for (PassId pass : plan.orderedPasses)
    {
        sorted.push_back(pass.value);
    }
    return sorted;
}

std::vector<PlannedImage> planAttachmentImages(
    std::span<const PassDesc> passes,
    VkExtent2D defaultExtent,
    const std::unordered_set<std::string> &existingImages)
{
    std::unordered_set<std::string> knownImages = existingImages;
    std::vector<PlannedImage> planned;

    for (const auto &pass : passes)
    {
        for (const auto &write : pass.writes)
        {
            if (!knownImages.insert(write.name).second)
            {
                continue;
            }

            const bool depth = isDepthFormat(write.format);
            planned.push_back({
                .name   = write.name,
                .format = write.format,
                .extent = (write.extent.width == 0 || write.extent.height == 0)
                              ? defaultExtent
                              : write.extent,
                .usage = static_cast<VkImageUsageFlags>(
                    depth
                        ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                        : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT),
                .aspect = static_cast<VkImageAspectFlags>(
                    depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT),
            });
        }
    }

    return planned;
}

std::string dumpTopology(
    std::span<const PassDesc> passes,
    std::span<const size_t> sortedPassIndices,
    std::span<const std::vector<BarrierDebugInfo>> barriersByPass)
{
    std::ostringstream out;
    out << "framegraph\n";
    out << "declaration-order:\n";

    for (size_t i = 0; i < passes.size(); ++i)
    {
        const auto &pass = passes[i];
        out << "  [" << i << "] " << pass.name << " type=" << passTypeName(pass.type) << '\n';

        for (const auto &dependency : pass.explicitDeps)
        {
            out << "    depends-on " << dependency << '\n';
        }
        for (const auto &binding : pass.bindings)
        {
            out << "    binding " << binding.binding << " resource=" << binding.resourceName
                << " access=" << accessName(binding.access) << " descriptor="
                << static_cast<int>(binding.type) << " stages=";
            writeHex(out, binding.stages);
            out << " layout=" << static_cast<int>(binding.imageLayout)
                << " count=" << binding.descriptorCount << " mip=" << binding.mipLevel << '\n';
        }
        for (const auto &write : pass.writes)
        {
            out << "    attachment resource=" << write.name << " format="
                << static_cast<int>(write.format) << " extent=" << write.extent.width << 'x'
                << write.extent.height << " load=" << static_cast<int>(write.loadOp) << '\n';
        }
    }

    out << "execution-order:";
    for (size_t index : sortedPassIndices)
    {
        out << ' ' << passes[index].name;
    }
    out << '\n';

    out << "barriers:\n";
    for (size_t passIndex = 0; passIndex < barriersByPass.size(); ++passIndex)
    {
        for (const auto &barrier : barriersByPass[passIndex])
        {
            out << "  before=" << passes[passIndex].name << " resource=" << barrier.resourceName
                << " layout=" << static_cast<int>(barrier.oldLayout) << "->"
                << static_cast<int>(barrier.newLayout) << " src-stage=";
            writeHex(out, barrier.srcStage);
            out << " src-access=";
            writeHex(out, barrier.srcAccess);
            out << " dst-stage=";
            writeHex(out, barrier.dstStage);
            out << " dst-access=";
            writeHex(out, barrier.dstAccess);
            out << '\n';
        }
    }

    return out.str();
}

} // namespace lr::framegraph
