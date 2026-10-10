// Dumps the GPU-integrated environment BRDF so it can be diffed against a reference
// implementation of learnopengl's IntegrateBRDF. Runs the real IBLPass against the real
// brdflut.comp.spv — this verifies the shipped shader, not a transcription of it.
//
// Build and run:
//   cmake --build build --config Debug --target brdf_lut_gpu_smoke
//   build/Debug/brdf_lut_gpu_smoke.exe <out.bin>
// Output: lutRes * lutRes pairs of little-endian float32 (A, B), row-major, v increasing.

#include "core/framegraph/FrameGraph.hpp"
#include "core/framegraph/ResourceRegistry.hpp"
#include "core/passes/ibl/IblPass.hpp"
#include "core/vulkan/Allocator.hpp"
#include "core/vulkan/VkResultUtils.hpp"
#include "core/vulkan/VulkanContext.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lr;

int main(int argc, char **argv)
try
{
    const std::string outPath = argc > 1 ? argv[1] : "brdf_lut_gpu.bin";
    constexpr uint32_t kRes = 256;

    VulkanContext    context({.appName = "BRDF LUT smoke", .enableValidation = true});
    Allocator        allocator(context);
    ResourceRegistry registry(context, allocator, {1, 1});

    // No HDRI: IBLPass falls back to a black 1x1 environment. The cubemap resolutions are shrunk to
    // keep the run quick — only the LUT matters here, and it does not depend on the environment.
    const IBLPass::Config cfg{
        .envRes = 8,
        .irrRes = 8,
        .pfRes  = 8,
        .pfMips = 2,
        .lutRes = kRes,
    };

    IBLPass pass(cfg);
    pass.uploadResources(registry);

    // The real pass does not need TRANSFER_SRC; this harness does, to copy the result back.
    const AllocatedImage *registered = registry.getImage("ibl_brdf_lut");
    if (registered == nullptr) throw std::runtime_error("ibl_brdf_lut was not registered");

    FrameGraph graph(context, registry);
    pass.build(graph);
    graph.executeAndWait({
        {"ibl_brdf_lut", VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL},
    });

    const AllocatedImage *lut = registry.getImage("ibl_brdf_lut");
    const VkDeviceSize    bytes = VkDeviceSize(kRes) * kRes * 2 * sizeof(float);

    AllocatedBuffer host = allocator.createBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);

    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = context.getGraphicsQueueFamily();
    VkCommandPool pool;
    checkVk(vkCreateCommandPool(context.getDevice(), &poolInfo, nullptr, &pool), "create pool");

    VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    alloc.commandPool        = pool;
    alloc.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = 1;
    VkCommandBuffer cmd;
    checkVk(vkAllocateCommandBuffers(context.getDevice(), &alloc, &cmd), "allocate command");

    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    checkVk(vkBeginCommandBuffer(cmd, &begin), "begin");

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent      = {kRes, kRes, 1};
    vkCmdCopyImageToBuffer(cmd, lut->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, host.buffer, 1, &region);

    VkBufferMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
    barrier.srcStageMask        = VK_PIPELINE_STAGE_2_COPY_BIT;
    barrier.srcAccessMask       = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.dstStageMask        = VK_PIPELINE_STAGE_2_HOST_BIT;
    barrier.dstAccessMask       = VK_ACCESS_2_HOST_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer              = host.buffer;
    barrier.size                = bytes;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.bufferMemoryBarrierCount = 1;
    dependency.pBufferMemoryBarriers    = &barrier;
    vkCmdPipelineBarrier2(cmd, &dependency);

    checkVk(vkEndCommandBuffer(cmd), "end");

    VkCommandBufferSubmitInfo commands{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    commands.commandBuffer = cmd;
    VkSubmitInfo2 submit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos    = &commands;
    checkVk(vkQueueSubmit2(context.getGraphicsQueue(), 1, &submit, VK_NULL_HANDLE), "submit");
    context.waitIdle();
    checkVk(vmaInvalidateAllocation(allocator.getHandle(), host.allocation, 0, bytes), "invalidate");

    std::vector<float> values(size_t(kRes) * kRes * 2);
    std::memcpy(values.data(), host.info.pMappedData, bytes);

    FILE *out = std::fopen(outPath.c_str(), "wb");
    if (out == nullptr) throw std::runtime_error("cannot open " + outPath);
    std::fwrite(values.data(), sizeof(float), values.size(), out);
    std::fclose(out);
    std::printf("wrote %zu floats (%ux%u RG32F) to %s\n", values.size(), kRes, kRes, outPath.c_str());

    vkFreeCommandBuffers(context.getDevice(), pool, 1, &cmd);
    vkDestroyCommandPool(context.getDevice(), pool, nullptr);
    allocator.destroy(host);
    return 0;
} catch (const std::exception &e)
{
    std::fprintf(stderr, "brdf_lut_gpu_smoke failed: %s\n", e.what());
    return 1;
}
