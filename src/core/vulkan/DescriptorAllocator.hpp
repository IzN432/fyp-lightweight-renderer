#pragma once

#include <vulkan/vulkan.h>

#include <deque>
#include <span>
#include <vector>

namespace lr
{

class DescriptorAllocator
{
public:
    explicit DescriptorAllocator(VkDevice device);
    ~DescriptorAllocator();

    DescriptorAllocator(const DescriptorAllocator &)            = delete;
    DescriptorAllocator &operator=(const DescriptorAllocator &) = delete;

    // Create a descriptor set layout — ownership retained here, destroyed in destructor.
    VkDescriptorSetLayout createLayout(std::span<const VkDescriptorSetLayoutBinding> bindings);

    // Create a pipeline layout wrapping one descriptor set layout.
    // Optionally appends a single push constant range.
    VkPipelineLayout createPipelineLayout(VkDescriptorSetLayout layout, uint32_t pushConstantSize = 0,
                                          VkShaderStageFlags pushStages = 0);

    // Free all allocated descriptor sets and destroy all cached layouts/pipeline
    // layouts. Call before recompiling passes so the pool doesn't exhaust.
    void reset();

    // Allocate a descriptor set; when the current pool is exhausted another is created, so the number
    // of sets (e.g. one per frame in flight for some passes) isn't capped by a fixed pool size.
    VkDescriptorSet allocate(VkDescriptorSetLayout layout);

    // Accumulate descriptor writes.
    void writeImage(VkDescriptorSet set, uint32_t binding, VkImageView view, VkSampler sampler,
                    VkImageLayout imageLayout, VkDescriptorType type);
    void writeImageArray(VkDescriptorSet set, uint32_t binding, std::span<const VkImageView> views, VkSampler sampler,
                         VkImageLayout imageLayout, VkDescriptorType type);
    void writeBuffer(VkDescriptorSet set, uint32_t binding, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range,
                     VkDescriptorType type);

    // Flush all accumulated writes via vkUpdateDescriptorSets.
    void commit();

private:
    VkDescriptorPool createPool();

    VkDevice                      m_device;
    std::vector<VkDescriptorPool> m_pools; // the last one is allocated from

    std::vector<VkDescriptorSetLayout> m_layouts;
    std::vector<VkPipelineLayout>      m_pipelineLayouts;

    // Stable backing storage for VkWriteDescriptorSet pointers
    std::deque<VkDescriptorImageInfo>              m_imageInfos;
    std::deque<std::vector<VkDescriptorImageInfo>> m_imageInfoArrays;
    std::deque<VkDescriptorBufferInfo>             m_bufferInfos;
    std::vector<VkWriteDescriptorSet>              m_writes;
};

} // namespace lr
