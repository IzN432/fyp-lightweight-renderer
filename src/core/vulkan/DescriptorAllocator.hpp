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

    DescriptorAllocator(const DescriptorAllocator &) = delete;
    DescriptorAllocator &operator=(const DescriptorAllocator &) = delete;

    // Create a descriptor set layout — ownership retained here, destroyed in destructor.
    VkDescriptorSetLayout createLayout(std::span<const VkDescriptorSetLayoutBinding> bindings);

    // Create a pipeline layout wrapping one descriptor set layout.
    // Optionally appends a single push constant range.
    VkPipelineLayout createPipelineLayout(VkDescriptorSetLayout layout,
                                           uint32_t pushConstantSize = 0,
                                           VkShaderStageFlags pushStages = 0);

    // Allocate a descriptor set from the internal pool.
    VkDescriptorSet allocate(VkDescriptorSetLayout layout);

    // Accumulate descriptor writes.
    void writeImage(VkDescriptorSet set, uint32_t binding,
                    VkImageView view, VkSampler sampler, VkImageLayout imageLayout,
                    VkDescriptorType type);
    void writeImageArray(VkDescriptorSet set, uint32_t binding,
                         std::span<const VkImageView> views,
                         VkSampler sampler, VkImageLayout imageLayout,
                         VkDescriptorType type);
    void writeBuffer(VkDescriptorSet set, uint32_t binding,
                     VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range,
                     VkDescriptorType type);

    // Flush all accumulated writes via vkUpdateDescriptorSets.
    void commit();

private:
    VkDevice m_device;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;

    std::vector<VkDescriptorSetLayout> m_layouts;
    std::vector<VkPipelineLayout>      m_pipelineLayouts;

    // Stable backing storage for VkWriteDescriptorSet pointers
    std::deque<VkDescriptorImageInfo>               m_imageInfos;
    std::deque<std::vector<VkDescriptorImageInfo>>  m_imageInfoArrays;
    std::deque<VkDescriptorBufferInfo>              m_bufferInfos;
    std::vector<VkWriteDescriptorSet>               m_writes;
};

}  // namespace lr
