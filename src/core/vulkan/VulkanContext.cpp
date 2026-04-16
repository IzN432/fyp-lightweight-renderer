#include "VulkanContext.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <stdexcept>

namespace lr
{

// ---------------------------------------------------------------------------
// Debug messenger callback
// ---------------------------------------------------------------------------

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT /*type*/,
    const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
    void * /*pUserData*/)
{
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
        spdlog::error("[Vulkan] {}", pCallbackData->pMessage);
    else if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        spdlog::warn("[Vulkan] {}", pCallbackData->pMessage);

    return VK_FALSE;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

VulkanContext::VulkanContext(const Config &config, VkSurfaceKHR surface)
{
    m_config = config;
    createInstance();
    pickPhysicalDevice(surface);
    createDevice();
}

VulkanContext::~VulkanContext()
{
    destroy();
}

void VulkanContext::destroy()
{
    if (m_device != VK_NULL_HANDLE)
    {
        vkDestroyDevice(m_device, nullptr);
        m_device = VK_NULL_HANDLE;
    }

    if (m_debugMessenger != VK_NULL_HANDLE)
    {
        auto fn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(m_instance, "vkDestroyDebugUtilsMessengerEXT"));
        if (fn)
            fn(m_instance, m_debugMessenger, nullptr);
        m_debugMessenger = VK_NULL_HANDLE;
    }

    if (m_instance != VK_NULL_HANDLE)
    {
        vkDestroyInstance(m_instance, nullptr);
        m_instance = VK_NULL_HANDLE;
    }
}

VkQueue VulkanContext::getGraphicsQueue(int idx) const
{
    return m_graphicsQueues.at(static_cast<size_t>(idx));
}

VkQueue VulkanContext::getTransferQueue() const
{
    if (m_hasDedicatedTransfer)
        return m_transferQueue;
    return m_graphicsQueues[0];
}

int VulkanContext::getTransferQueueFamily() const
{
    if (m_hasDedicatedTransfer)
        return m_transferQueueFamily;
    return m_graphicsQueueFamilies[0];
}

void VulkanContext::waitIdle() const
{
    vkDeviceWaitIdle(m_device);
}

void VulkanContext::setDebugName(VkObjectType type, uint64_t handle, std::string_view name) const
{
    if (!m_vkSetDebugUtilsObjectName || handle == 0)
        return;

    VkDebugUtilsObjectNameInfoEXT info{};
    info.sType        = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    info.objectType   = type;
    info.objectHandle = handle;
    info.pObjectName  = name.data();
    m_vkSetDebugUtilsObjectName(m_device, &info);
}

void VulkanContext::beginDebugLabel(VkCommandBuffer cmd,
                                    std::string_view name,
                                    const std::array<float, 4> &color) const
{
    if (!m_vkCmdBeginDebugUtilsLabel || cmd == VK_NULL_HANDLE)
        return;

    VkDebugUtilsLabelEXT label{};
    label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    label.pLabelName = name.data();
    label.color[0] = color[0];
    label.color[1] = color[1];
    label.color[2] = color[2];
    label.color[3] = color[3];
    m_vkCmdBeginDebugUtilsLabel(cmd, &label);
}

void VulkanContext::endDebugLabel(VkCommandBuffer cmd) const
{
    if (!m_vkCmdEndDebugUtilsLabel || cmd == VK_NULL_HANDLE)
        return;

    m_vkCmdEndDebugUtilsLabel(cmd);
}

uint32_t VulkanContext::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags flags) const
{
    for (uint32_t i = 0; i < m_memProperties.memoryTypeCount; ++i)
    {
        bool typeMatch = (typeBits & (1u << i)) != 0;
        bool propMatch = (m_memProperties.memoryTypes[i].propertyFlags & flags) == flags;
        if (typeMatch && propMatch)
            return i;
    }
    spdlog::error("Runtime error: throwing std::runtime_error");
    throw std::runtime_error("VulkanContext: no suitable memory type found");
}

VkFormat VulkanContext::findSupportedFormat(const std::vector<VkFormat> &candidates,
                                             VkImageTiling tiling,
                                             VkFormatFeatureFlags features) const
{
    for (VkFormat fmt : candidates)
    {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(m_physicalDevice, fmt, &props);

        bool supported = (tiling == VK_IMAGE_TILING_LINEAR)
                             ? (props.linearTilingFeatures & features) == features
                             : (props.optimalTilingFeatures & features) == features;

        if (supported)
            return fmt;
    }
    spdlog::error("Runtime error: throwing std::runtime_error");
    throw std::runtime_error("VulkanContext: no supported format found in candidates");
}

// ---------------------------------------------------------------------------
// Private — instance
// ---------------------------------------------------------------------------

void VulkanContext::createInstance()
{
    vkEnumerateInstanceVersion(&m_apiVersion);

    // Required layers
    if (m_config.enableValidation)
        m_layers.push_back("VK_LAYER_KHRONOS_validation");

    for (const char *layer : m_config.extraLayers)
        m_layers.push_back(layer);

    if (!checkLayerSupport(m_layers))
    {
        spdlog::error("Runtime error: throwing std::runtime_error");
        throw std::runtime_error("VulkanContext: requested Vulkan layers not available");
    }

    // Required instance extensions
    if (m_config.enableValidation || m_config.enableDebugNames)
        m_instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    for (const char *ext : m_config.extraInstanceExtensions)
        m_instanceExtensions.push_back(ext);

    if (!checkInstanceExtensionSupport(m_instanceExtensions))
    {
        spdlog::error("Runtime error: throwing std::runtime_error");
        throw std::runtime_error("VulkanContext: requested instance extensions not available");
    }

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = m_config.appName.c_str();
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "lr";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = m_apiVersion;

    VkInstanceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ci.pApplicationInfo = &appInfo;
    ci.enabledLayerCount = static_cast<uint32_t>(m_layers.size());
    ci.ppEnabledLayerNames = m_layers.data();
    ci.enabledExtensionCount = static_cast<uint32_t>(m_instanceExtensions.size());
    ci.ppEnabledExtensionNames = m_instanceExtensions.data();

    if (vkCreateInstance(&ci, nullptr, &m_instance) != VK_SUCCESS)
    {
        spdlog::error("Runtime error: throwing std::runtime_error");
        throw std::runtime_error("VulkanContext: failed to create VkInstance");
    }

    spdlog::info("VulkanContext: instance created (API {}.{}.{})",
                 VK_VERSION_MAJOR(m_apiVersion),
                 VK_VERSION_MINOR(m_apiVersion),
                 VK_VERSION_PATCH(m_apiVersion));

    if (m_config.enableValidation)
        createDebugMessenger();
}

void VulkanContext::createDebugMessenger()
{
    VkDebugUtilsMessengerCreateInfoEXT ci{};
    ci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    ci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    ci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    ci.pfnUserCallback = debugCallback;

    auto fn = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(m_instance, "vkCreateDebugUtilsMessengerEXT"));

    if (!fn || fn(m_instance, &ci, nullptr, &m_debugMessenger) != VK_SUCCESS)
        spdlog::warn("VulkanContext: could not create debug messenger");
}

// ---------------------------------------------------------------------------
// Private — physical device
// ---------------------------------------------------------------------------

VulkanContext::QueueFamilies VulkanContext::findQueueFamilies(VkPhysicalDevice device,
                                                               VkSurfaceKHR surface) const
{
    QueueFamilies result;

    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());

    // Find graphics queue
    for (uint32_t i = 0; i < count; ++i)
    {
        if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
        {
            result.graphics = static_cast<int>(i);
            break;
        }
    }

    // Find dedicated transfer queue (no graphics bit preferred)
    for (uint32_t i = 0; i < count; ++i)
    {
        bool hasTransfer = families[i].queueFlags & VK_QUEUE_TRANSFER_BIT;
        bool noGraphics = !(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT);
        if (hasTransfer && noGraphics)
        {
            result.transfer = static_cast<int>(i);
            break;
        }
    }

    // Check present support if a surface is provided
    if (surface != VK_NULL_HANDLE)
    {
        for (uint32_t i = 0; i < count; ++i)
        {
            VkBool32 presentSupport = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
            if (presentSupport)
            {
                result.present = static_cast<int>(i);
                break;
            }
        }
    }

    return result;
}

int VulkanContext::scoreDevice(VkPhysicalDevice device, VkSurfaceKHR surface) const
{
    QueueFamilies families = findQueueFamilies(device, surface);

    // Must have a graphics queue
    if (families.graphics < 0)
        return -1;

    // If a surface was provided, must support present
    if (surface != VK_NULL_HANDLE && families.present < 0)
        return -1;

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(device, &props);

    int score = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) ? 1000 : 100;

    spdlog::info("VulkanContext: found device '{}' (score {})", props.deviceName, score);
    return score;
}

void VulkanContext::pickPhysicalDevice(VkSurfaceKHR surface)
{
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(m_instance, &count, nullptr);
    if (count == 0)
    {
        spdlog::error("Runtime error: throwing std::runtime_error");
        throw std::runtime_error("VulkanContext: no Vulkan-capable GPUs found");
    }

    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(m_instance, &count, devices.data());

    int bestScore = -1;
    for (VkPhysicalDevice dev : devices)
    {
        int score = scoreDevice(dev, surface);
        if (score > bestScore)
        {
            bestScore = score;
            m_physicalDevice = dev;
        }
    }

    if (bestScore < 0)
    {
        spdlog::error("Runtime error: throwing std::runtime_error");
        throw std::runtime_error("VulkanContext: no suitable GPU found");
    }

    // Query properties via pNext chain
    m_deviceProperties11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES;
    m_deviceProperties12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES;
    m_deviceProperties12.pNext = &m_deviceProperties11;
    m_deviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    m_deviceProperties2.pNext = &m_deviceProperties12;
    vkGetPhysicalDeviceProperties2(m_physicalDevice, &m_deviceProperties2);

    // Query features via pNext chain
    m_deviceFeatures13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    m_deviceFeatures12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    m_deviceFeatures12.pNext = &m_deviceFeatures13;
    m_deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    m_deviceFeatures2.pNext = &m_deviceFeatures12;
    vkGetPhysicalDeviceFeatures2(m_physicalDevice, &m_deviceFeatures2);

    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &m_memProperties);

    spdlog::info("VulkanContext: selected GPU '{}'",
                 m_deviceProperties2.properties.deviceName);
}

// ---------------------------------------------------------------------------
// Private — logical device
// ---------------------------------------------------------------------------

void VulkanContext::createDevice()
{
    QueueFamilies families = findQueueFamilies(m_physicalDevice, VK_NULL_HANDLE);

    // Collect unique queue families we need
    std::vector<int> uniqueFamilies = {families.graphics};
    if (m_config.enableTransferQueue && families.transfer >= 0)
    {
        uniqueFamilies.push_back(families.transfer);
        m_hasDedicatedTransfer = true;
        m_transferQueueFamily = families.transfer;
    }

    float queuePriority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queueCIs;
    queueCIs.reserve(uniqueFamilies.size());

    for (int family : uniqueFamilies)
    {
        VkDeviceQueueCreateInfo qci{};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = static_cast<uint32_t>(family);
        qci.queueCount = 1;
        qci.pQueuePriorities = &queuePriority;
        queueCIs.push_back(qci);
    }

    // Enable device extensions
    for (const char *ext : m_config.extraDeviceExtensions)
        m_deviceExtensions.push_back(ext);

    // Enable features — chain 1.2 and 1.3 structs
    VkPhysicalDeviceVulkan13Features features13{};
    features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    features13.dynamicRendering = VK_TRUE;  // useful for framegraph
    features13.synchronization2 = VK_TRUE;

    VkPhysicalDeviceVulkan12Features features12{};
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features12.pNext = &features13;
    features12.bufferDeviceAddress = VK_TRUE;
    features12.descriptorIndexing = VK_TRUE;
    features12.runtimeDescriptorArray = VK_TRUE;
    features12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;

    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &features12;
    features2.features.samplerAnisotropy = VK_TRUE;
    features2.features.geometryShader = VK_TRUE;

    VkDeviceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    ci.pNext = &features2;
    ci.queueCreateInfoCount = static_cast<uint32_t>(queueCIs.size());
    ci.pQueueCreateInfos = queueCIs.data();
    ci.enabledExtensionCount = static_cast<uint32_t>(m_deviceExtensions.size());
    ci.ppEnabledExtensionNames = m_deviceExtensions.data();
    // Note: pEnabledFeatures must be null when using VkPhysicalDeviceFeatures2 in pNext

    if (vkCreateDevice(m_physicalDevice, &ci, nullptr, &m_device) != VK_SUCCESS)
    {
        spdlog::error("Runtime error: throwing std::runtime_error");
        throw std::runtime_error("VulkanContext: failed to create logical device");
    }

    // Retrieve queue handles
    m_graphicsQueues.resize(1);
    m_graphicsQueueFamilies.resize(1);
    m_graphicsQueueFamilies[0] = families.graphics;
    vkGetDeviceQueue(m_device, static_cast<uint32_t>(families.graphics), 0, &m_graphicsQueues[0]);

    if (m_hasDedicatedTransfer)
        vkGetDeviceQueue(m_device, static_cast<uint32_t>(m_transferQueueFamily), 0, &m_transferQueue);

    if (m_config.enableDebugNames)
    {
        m_vkSetDebugUtilsObjectName =
            reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
                vkGetInstanceProcAddr(m_instance, "vkSetDebugUtilsObjectNameEXT"));
        if (!m_vkSetDebugUtilsObjectName)
            spdlog::warn("VulkanContext: vkSetDebugUtilsObjectNameEXT not available — debug names disabled");

        m_vkCmdBeginDebugUtilsLabel =
            reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(
                vkGetDeviceProcAddr(m_device, "vkCmdBeginDebugUtilsLabelEXT"));
        m_vkCmdEndDebugUtilsLabel =
            reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(
                vkGetDeviceProcAddr(m_device, "vkCmdEndDebugUtilsLabelEXT"));
        if (!m_vkCmdBeginDebugUtilsLabel || !m_vkCmdEndDebugUtilsLabel)
            spdlog::warn("VulkanContext: debug label commands unavailable — pass labels disabled");
    }

    spdlog::info("VulkanContext: logical device created");
}

// ---------------------------------------------------------------------------
// Private — validation helpers
// ---------------------------------------------------------------------------

bool VulkanContext::checkLayerSupport(const std::vector<const char *> &layers) const
{
    uint32_t count = 0;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> available(count);
    vkEnumerateInstanceLayerProperties(&count, available.data());

    for (const char *name : layers)
    {
        bool found = std::any_of(available.begin(), available.end(),
                                 [name](const VkLayerProperties &p)
                                 { return std::string(p.layerName) == name; });
        if (!found)
        {
            spdlog::error("VulkanContext: layer '{}' not available", name);
            return false;
        }
    }
    return true;
}

bool VulkanContext::checkInstanceExtensionSupport(const std::vector<const char *> &extensions) const
{
    uint32_t count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> available(count);
    vkEnumerateInstanceExtensionProperties(nullptr, &count, available.data());

    for (const char *name : extensions)
    {
        bool found = std::any_of(available.begin(), available.end(),
                                 [name](const VkExtensionProperties &p)
                                 { return std::string(p.extensionName) == name; });
        if (!found)
        {
            spdlog::error("VulkanContext: instance extension '{}' not available", name);
            return false;
        }
    }
    return true;
}

}  // namespace lr
