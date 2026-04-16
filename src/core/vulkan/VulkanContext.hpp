#pragma once

#include <vulkan/vulkan.h>

#include <array>
#include <string>
#include <string_view>
#include <vector>

namespace lr
{

class VulkanContext
{
public:
    struct Config
    {
        std::string appName = "App";
        bool enableValidation = true;
        bool enableRayTracing = false;
        bool enableDebugNames = true;
        int numGraphicsQueues = 1;
        bool enableTransferQueue = true;
        std::vector<const char *> extraInstanceExtensions;
        std::vector<const char *> extraDeviceExtensions;
        std::vector<const char *> extraLayers;
    };

    struct QueueFamilies
    {
        int graphics = -1;
        int transfer = -1;
        int present = -1;
    };

public:
    // surface is optional — VK_NULL_HANDLE for headless
    // surface is NOT stored; only used during physical device selection
    explicit VulkanContext(const Config &config, VkSurfaceKHR surface = VK_NULL_HANDLE);
    ~VulkanContext();

    VulkanContext(const VulkanContext &) = delete;
    VulkanContext &operator=(const VulkanContext &) = delete;
    VulkanContext(VulkanContext &&) = delete;
    VulkanContext &operator=(VulkanContext &&) = delete;

    // Core handles
    VkInstance getInstance() const { return m_instance; }
    VkPhysicalDevice getPhysicalDevice() const { return m_physicalDevice; }
    VkDevice getDevice() const { return m_device; }

    // Queues
    VkQueue getGraphicsQueue(int idx = 0) const;
    int getGraphicsQueueFamily() const { return m_graphicsQueueFamilies[0]; }

    VkQueue getTransferQueue() const;
    int getTransferQueueFamily() const;
    bool hasTransferQueue() const { return m_hasDedicatedTransfer; }

    // Device queries
    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags flags) const;
    VkFormat findSupportedFormat(const std::vector<VkFormat> &candidates,
                                  VkImageTiling tiling,
                                  VkFormatFeatureFlags features) const;

    // Properties
    const VkPhysicalDeviceProperties2 &getDeviceProperties() const { return m_deviceProperties2; }
    const VkPhysicalDeviceMemoryProperties &getMemoryProperties() const { return m_memProperties; }
    uint32_t getApiVersion() const { return m_apiVersion; }

    // Sync
    void waitIdle() const;

    // Debug — no-op if enableDebugNames is false or the extension isn't loaded.
    void setDebugName(VkObjectType type, uint64_t handle, std::string_view name) const;
    void beginDebugLabel(VkCommandBuffer cmd,
                         std::string_view name,
                         const std::array<float, 4> &color = {1.0f, 1.0f, 1.0f, 1.0f}) const;
    void endDebugLabel(VkCommandBuffer cmd) const;
    bool debugNamesEnabled() const { return m_config.enableDebugNames; }

private:
    void createInstance();
    void createDebugMessenger();
    void pickPhysicalDevice(VkSurfaceKHR surface);
    void createDevice();
    void destroy();

    QueueFamilies findQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface) const;
    int scoreDevice(VkPhysicalDevice device, VkSurfaceKHR surface) const;

    bool checkLayerSupport(const std::vector<const char *> &layers) const;
    bool checkInstanceExtensionSupport(const std::vector<const char *> &extensions) const;

private:
    VkInstance m_instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT m_debugMessenger = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;

    std::vector<VkQueue> m_graphicsQueues;
    std::vector<int> m_graphicsQueueFamilies;

    VkQueue m_transferQueue = VK_NULL_HANDLE;
    int m_transferQueueFamily = -1;
    bool m_hasDedicatedTransfer = false;

    VkPhysicalDeviceMemoryProperties m_memProperties{};
    VkPhysicalDeviceProperties2 m_deviceProperties2{};
    VkPhysicalDeviceVulkan11Properties m_deviceProperties11{};
    VkPhysicalDeviceVulkan12Properties m_deviceProperties12{};
    VkPhysicalDeviceFeatures2 m_deviceFeatures2{};
    VkPhysicalDeviceVulkan12Features m_deviceFeatures12{};
    VkPhysicalDeviceVulkan13Features m_deviceFeatures13{};

    std::vector<const char *> m_instanceExtensions;
    std::vector<const char *> m_deviceExtensions;
    std::vector<const char *> m_layers;

    Config m_config;
    uint32_t m_apiVersion = VK_API_VERSION_1_0;

    PFN_vkSetDebugUtilsObjectNameEXT m_vkSetDebugUtilsObjectName = nullptr;
    PFN_vkCmdBeginDebugUtilsLabelEXT m_vkCmdBeginDebugUtilsLabel = nullptr;
    PFN_vkCmdEndDebugUtilsLabelEXT m_vkCmdEndDebugUtilsLabel = nullptr;
};

}  // namespace lr
