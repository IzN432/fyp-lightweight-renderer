#include "FrameGraphTopology.hpp"
#include "PassDescAdapter.hpp"
#include "compiler/GraphCompiler.hpp"
#include "core/vulkan/VkFormatUtils.hpp"

#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace
{

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

const char *accessName(lr::AccessMode access)
{
    switch (access)
    {
        case lr::AccessMode::Read:
            return "read";
        case lr::AccessMode::Write:
            return "write";
        case lr::AccessMode::ReadWrite:
            return "read-write";
    }
    return "unknown";
}

template <typename T> void writeHex(std::ostringstream &out, T value)
{
    out << "0x" << std::hex << static_cast<uint64_t>(value) << std::dec;
}

} // namespace

namespace lr::framegraph
{

std::vector<size_t> sortPasses(std::span<const PassDesc> passes, const ResourceHandleRegistry &resources,
                               uint64_t passOwner)
{
    const GraphDefinition graph = translatePassDescriptions(passes, resources, passOwner);
    const ExecutionPlan   plan  = buildExecutionPlan(graph);

    std::vector<size_t> sorted;
    sorted.reserve(plan.orderedPasses.size());
    for (PassId pass : plan.orderedPasses)
    {
        sorted.push_back(pass.value);
    }
    return sorted;
}

std::vector<PlannedImage> planAttachmentImages(std::span<const PassDesc>              passes,
                                               const ResourceHandleRegistry          &resources,
                                               const std::unordered_set<std::string> &existingImages)
{
    struct AggregatedImage
    {
        PlannedImage image;
        bool         hasAttachment = false;
    };

    std::vector<AggregatedImage>            aggregated;
    std::unordered_map<std::string, size_t> indices;

    for (const auto &pass : passes)
    {
        for (const ImageUse &use : pass.imageUses)
        {
            const std::string &name   = resources.name(use.image);
            const auto [it, inserted] = indices.emplace(name, aggregated.size());
            if (inserted)
            {
                aggregated.push_back({.image = {.name = name}});
            }
            AggregatedImage &entry = aggregated[it->second];

            switch (use.usage)
            {
                case ImageUsage::Sampled:
                case ImageUsage::SampledDepth:
                case ImageUsage::SampledMultisample:
                case ImageUsage::SampledArray:
                    entry.image.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
                    break;
                case ImageUsage::Storage:
                    entry.image.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
                    break;
                case ImageUsage::ColorAttachment:
                    entry.image.usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
                    break;
                case ImageUsage::DepthAttachment:
                    entry.image.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
                    break;
            }

            if (!use.isAttachment())
            {
                continue;
            }
            if (use.format == VK_FORMAT_UNDEFINED)
            {
                throw std::runtime_error("FrameGraph: attachment image '" + name + "' has no format");
            }

            const VkImageAspectFlags aspect =
                use.usage == ImageUsage::DepthAttachment ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;

            if (!entry.hasAttachment)
            {
                entry.image.format  = use.format;
                entry.image.extent  = use.extent;
                entry.image.aspect  = aspect;
                entry.hasAttachment = true;
                continue;
            }
            if (entry.image.format != use.format)
            {
                throw std::runtime_error("FrameGraph: attachment image '" + name + "' has conflicting formats");
            }
            if (entry.image.extent != use.extent)
            {
                throw std::runtime_error("FrameGraph: attachment image '" + name + "' has conflicting extents");
            }
            if (entry.image.aspect != aspect)
            {
                throw std::runtime_error("FrameGraph: attachment image '" + name +
                                         "' is declared as both color and depth");
            }
        }
    }

    std::vector<PlannedImage> planned;
    for (AggregatedImage &entry : aggregated)
    {
        if (entry.hasAttachment && !existingImages.contains(entry.image.name))
        {
            planned.push_back(std::move(entry.image));
        }
    }
    return planned;
}

std::vector<VkExtent2D> planRenderingExtents(std::span<const PassDesc> passes, VkExtent2D defaultExtent)
{
    std::vector<VkExtent2D> extents(passes.size(), defaultExtent);
    for (size_t passIndex = 0; passIndex < passes.size(); ++passIndex)
    {
        bool       hasAttachment = false;
        ExtentSpec extentSpec    = ExtentSpec::swapchain();
        for (const ImageUse &use : passes[passIndex].imageUses)
        {
            if (!use.isAttachment())
            {
                continue;
            }
            if (!hasAttachment)
            {
                extentSpec    = use.extent;
                hasAttachment = true;
            } else if (extentSpec != use.extent)
            {
                throw std::runtime_error("FrameGraph: pass '" + passes[passIndex].name +
                                         "' has attachments with different extents");
            }
        }
        if (hasAttachment)
        {
            extents[passIndex] = extentSpec.resolve(defaultExtent);
        }
    }
    return extents;
}

std::string dumpTopology(std::span<const PassDesc> passes, const ResourceHandleRegistry &resources,
                         std::span<const size_t>                        sortedPassIndices,
                         std::span<const std::vector<BarrierDebugInfo>> barriersByPass)
{
    std::ostringstream out;
    out << "framegraph\n";
    out << "declaration-order:\n";

    for (size_t i = 0; i < passes.size(); ++i)
    {
        const auto &pass = passes[i];
        out << "  [" << i << "] " << pass.name << " type=" << passTypeName(pass.type) << '\n';

        for (PassHandle dependency : pass.explicitDependencies)
        {
            out << "    depends-on " << passes[dependency.index].name << '\n';
        }
        for (const ImageUse &use : pass.imageUses)
        {
            out << "    image resource=" << resources.name(use.image) << " usage=" << static_cast<int>(use.usage)
                << " access=" << accessName(use.access);
            if (use.isDescriptor())
            {
                out << " binding=" << use.binding << " count=" << use.descriptorCount;
            }
            out << " stages=";
            writeHex(out, use.stages);
            out << " mip=" << use.boundMip;
            if (use.isAttachment())
            {
                out << " format=" << static_cast<int>(use.format) << " extent=" << use.extent.describe()
                    << " load=" << static_cast<int>(use.loadOp);
            }
            out << '\n';
        }
        for (const BufferUse &use : pass.bufferUses)
        {
            out << "    buffer resource=" << resources.name(use.buffer) << " usage=" << static_cast<int>(use.usage)
                << " access=" << accessName(use.access) << " binding=" << use.binding << " stages=";
            writeHex(out, use.stages);
            out << '\n';
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
                << " layout=" << static_cast<int>(barrier.oldLayout) << "->" << static_cast<int>(barrier.newLayout)
                << " src-stage=";
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
