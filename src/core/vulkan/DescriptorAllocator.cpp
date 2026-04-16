#include "DescriptorAllocator.hpp"

#include <spdlog/spdlog.h>

#include <stdexcept>

namespace lr
{

DescriptorAllocator::DescriptorAllocator(VkDevice device) : m_device(device)
{
    VkDescriptorPoolSize sizes[] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 64},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          32},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         32},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         32},
    };

    VkDescriptorPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    ci.maxSets = 64;
    ci.poolSizeCount = static_cast<uint32_t>(std::size(sizes));
    ci.pPoolSizes = sizes;

    if (vkCreateDescriptorPool(m_device, &ci, nullptr, &m_pool) != VK_SUCCESS)
    {
        spdlog::error("Runtime error: throwing std::runtime_error");
        throw std::runtime_error("DescriptorAllocator: failed to create pool");
    }

    spdlog::debug("DescriptorAllocator: created");
}

DescriptorAllocator::~DescriptorAllocator()
{
    for (auto layout : m_layouts)
        vkDestroyDescriptorSetLayout(m_device, layout, nullptr);
    for (auto layout : m_pipelineLayouts)
        vkDestroyPipelineLayout(m_device, layout, nullptr);
    vkDestroyDescriptorPool(m_device, m_pool, nullptr);
}

VkDescriptorSetLayout DescriptorAllocator::createLayout(
    std::span<const VkDescriptorSetLayoutBinding> bindings)
{
    VkDescriptorSetLayoutCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ci.bindingCount = static_cast<uint32_t>(bindings.size());
    ci.pBindings = bindings.data();

    VkDescriptorSetLayout layout;
    if (vkCreateDescriptorSetLayout(m_device, &ci, nullptr, &layout) != VK_SUCCESS)
    {
        spdlog::error("Runtime error: throwing std::runtime_error");
        throw std::runtime_error("DescriptorAllocator: failed to create layout");
    }

    m_layouts.push_back(layout);
    return layout;
}

VkPipelineLayout DescriptorAllocator::createPipelineLayout(VkDescriptorSetLayout layout,
                                                             uint32_t pushConstantSize,
                                                             VkShaderStageFlags pushStages)
{
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = pushStages;
    pushRange.offset = 0;
    pushRange.size = pushConstantSize;

    VkPipelineLayoutCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    ci.setLayoutCount = 1;
    ci.pSetLayouts = &layout;
    if (pushConstantSize > 0)
    {
        ci.pushConstantRangeCount = 1;
        ci.pPushConstantRanges = &pushRange;
    }

    VkPipelineLayout pipelineLayout;
    if (vkCreatePipelineLayout(m_device, &ci, nullptr, &pipelineLayout) != VK_SUCCESS)
    {
        spdlog::error("Runtime error: throwing std::runtime_error");
        throw std::runtime_error("DescriptorAllocator: failed to create pipeline layout");
    }

    m_pipelineLayouts.push_back(pipelineLayout);
    return pipelineLayout;
}

VkDescriptorSet DescriptorAllocator::allocate(VkDescriptorSetLayout layout)
{
    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = m_pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &layout;

    VkDescriptorSet set;
    if (vkAllocateDescriptorSets(m_device, &ai, &set) != VK_SUCCESS)
    {
        spdlog::error("Runtime error: throwing std::runtime_error");
        throw std::runtime_error("DescriptorAllocator: failed to allocate descriptor set");
    }

    return set;
}

void DescriptorAllocator::writeImage(VkDescriptorSet set, uint32_t binding,
                                      VkImageView view, VkSampler sampler,
                                      VkImageLayout imageLayout, VkDescriptorType type)
{
    // Push info first — pointer must remain stable until commit()
    m_imageInfos.push_back({sampler, view, imageLayout});

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = binding;
    write.descriptorCount = 1;
    write.descriptorType = type;
    write.pImageInfo = &m_imageInfos.back();
    m_writes.push_back(write);
}

void DescriptorAllocator::writeImageArray(VkDescriptorSet set, uint32_t binding,
                                           std::span<const VkImageView> views,
                                           VkSampler sampler,
                                           VkImageLayout imageLayout,
                                           VkDescriptorType type)
{
    if (views.empty())
    {
        spdlog::error("Runtime error: throwing std::runtime_error");
        throw std::runtime_error("DescriptorAllocator: writeImageArray called with empty views");
    }

    std::vector<VkDescriptorImageInfo> infos;
    infos.reserve(views.size());
    for (VkImageView view : views)
        infos.push_back({sampler, view, imageLayout});

    m_imageInfoArrays.push_back(std::move(infos));

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = binding;
    write.descriptorCount = static_cast<uint32_t>(m_imageInfoArrays.back().size());
    write.descriptorType = type;
    write.pImageInfo = m_imageInfoArrays.back().data();
    m_writes.push_back(write);
}

void DescriptorAllocator::writeBuffer(VkDescriptorSet set, uint32_t binding,
                                       VkBuffer buffer, VkDeviceSize offset,
                                       VkDeviceSize range, VkDescriptorType type)
{
    m_bufferInfos.push_back({buffer, offset, range});

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = binding;
    write.descriptorCount = 1;
    write.descriptorType = type;
    write.pBufferInfo = &m_bufferInfos.back();
    m_writes.push_back(write);
}

void DescriptorAllocator::commit()
{
    if (m_writes.empty())
        return;

    vkUpdateDescriptorSets(m_device,
                           static_cast<uint32_t>(m_writes.size()), m_writes.data(),
                           0, nullptr);

    m_writes.clear();
    m_imageInfos.clear();
    m_imageInfoArrays.clear();
    m_bufferInfos.clear();
}

}  // namespace lr
