#include "CudaInteropSemaphore.hpp"

#include "core/vulkan/VulkanContext.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>

#if defined(_WIN32)
#include <vulkan/vulkan_win32.h>
#endif

namespace lr::eulerian
{
namespace
{

void checkCuda(cudaError_t result, const char *operation)
{
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("CudaInteropSemaphore: ") + operation + ": " +
                                 cudaGetErrorString(result));
}

#if defined(_WIN32)
constexpr VkExternalSemaphoreHandleTypeFlagBits kHandleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#elif defined(__linux__)
constexpr VkExternalSemaphoreHandleTypeFlagBits kHandleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#else
#error CUDA/Vulkan external-semaphore interop is only implemented for Windows and Linux
#endif

// A semaphore whose handle CUDA will import has to be created exportable, and the driver has to
// agree it can be exported before creation (VUID-VkSemaphoreCreateInfo-pNext-None equivalents for
// external types). Querying first turns an obscure import failure into a named one.
void requireExportableSemaphore(const VulkanContext &context)
{
    VkPhysicalDeviceExternalSemaphoreInfo query{};
    query.sType      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO;
    query.handleType = kHandleType;

    VkExternalSemaphoreProperties properties{};
    properties.sType = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES;
    vkGetPhysicalDeviceExternalSemaphoreProperties(context.getPhysicalDevice(), &query, &properties);

    if ((properties.externalSemaphoreFeatures & VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT) == 0 ||
        (properties.externalSemaphoreFeatures & VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT) == 0)
        throw std::runtime_error("CudaInteropSemaphore: the device cannot export an importable binary semaphore of "
                                 "the platform's opaque handle type");
    if ((properties.compatibleHandleTypes & kHandleType) == 0)
        throw std::runtime_error("CudaInteropSemaphore: the platform's opaque semaphore handle type is not "
                                 "self-compatible on this device");
}

} // namespace

CudaInteropSemaphore::CudaInteropSemaphore(const VulkanContext &context, int cudaDevice)
    : m_context(&context), m_device(cudaDevice)
{
    requireExportableSemaphore(context);

    VkExportSemaphoreCreateInfo exportInfo{};
    exportInfo.sType       = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
    exportInfo.handleTypes = kHandleType;

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphoreInfo.pNext = &exportInfo;
    if (vkCreateSemaphore(context.getDevice(), &semaphoreInfo, nullptr, &m_semaphore) != VK_SUCCESS)
        throw std::runtime_error("CudaInteropSemaphore: vkCreateSemaphore failed");

    checkCuda(cudaSetDevice(m_device), "cudaSetDevice");

    cudaExternalSemaphoreHandleDesc desc{};

    try
    {
#if defined(_WIN32)
        const auto getHandle = reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(
            vkGetDeviceProcAddr(context.getDevice(), "vkGetSemaphoreWin32HandleKHR"));
        if (getHandle == nullptr)
            throw std::runtime_error("CudaInteropSemaphore: vkGetSemaphoreWin32HandleKHR is unavailable; enable "
                                     "VK_KHR_external_semaphore_win32");

        VkSemaphoreGetWin32HandleInfoKHR handleInfo{};
        handleInfo.sType      = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
        handleInfo.semaphore  = m_semaphore;
        handleInfo.handleType = kHandleType;
        HANDLE handle = nullptr;
        if (getHandle(context.getDevice(), &handleInfo, &handle) != VK_SUCCESS)
            throw std::runtime_error("CudaInteropSemaphore: vkGetSemaphoreWin32HandleKHR failed");

        desc.type                = cudaExternalSemaphoreHandleTypeOpaqueWin32;
        desc.handle.win32.handle = handle;
        // CUDA duplicates a Win32 handle on import, so this process still owns its copy either way.
        const cudaError_t importResult = cudaImportExternalSemaphore(&m_external, &desc);
        CloseHandle(handle);
        checkCuda(importResult, "cudaImportExternalSemaphore");
#else
        const auto getHandle = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(
            vkGetDeviceProcAddr(context.getDevice(), "vkGetSemaphoreFdKHR"));
        if (getHandle == nullptr)
            throw std::runtime_error("CudaInteropSemaphore: vkGetSemaphoreFdKHR is unavailable; enable "
                                     "VK_KHR_external_semaphore_fd");

        VkSemaphoreGetFdInfoKHR handleInfo{};
        handleInfo.sType      = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
        handleInfo.semaphore  = m_semaphore;
        handleInfo.handleType = kHandleType;
        int fd = -1;
        if (getHandle(context.getDevice(), &handleInfo, &fd) != VK_SUCCESS)
            throw std::runtime_error("CudaInteropSemaphore: vkGetSemaphoreFdKHR failed");

        desc.type      = cudaExternalSemaphoreHandleTypeOpaqueFd;
        desc.handle.fd = fd;
        // CUDA takes ownership of an imported file descriptor, including on import failure, so
        // there is nothing to close here on either path.
        checkCuda(cudaImportExternalSemaphore(&m_external, &desc), "cudaImportExternalSemaphore");
#endif
    } catch (...)
    {
        vkDestroySemaphore(context.getDevice(), m_semaphore, nullptr);
        m_semaphore = VK_NULL_HANDLE;
        throw;
    }
}

CudaInteropSemaphore::~CudaInteropSemaphore()
{
    if (m_external != nullptr)
    {
        (void)cudaSetDevice(m_device);
        (void)cudaDestroyExternalSemaphore(m_external);
    }
    if (m_semaphore != VK_NULL_HANDLE && m_context != nullptr)
        vkDestroySemaphore(m_context->getDevice(), m_semaphore, nullptr);
}

CudaInteropSemaphore::CudaInteropSemaphore(CudaInteropSemaphore &&other) noexcept
    : m_context(other.m_context), m_semaphore(other.m_semaphore), m_external(other.m_external),
      m_device(other.m_device)
{
    other.m_semaphore = VK_NULL_HANDLE;
    other.m_external  = nullptr;
}

CudaInteropSemaphore &CudaInteropSemaphore::operator=(CudaInteropSemaphore &&other) noexcept
{
    if (this != &other)
    {
        std::swap(m_context, other.m_context);
        std::swap(m_semaphore, other.m_semaphore);
        std::swap(m_external, other.m_external);
        std::swap(m_device, other.m_device);
    }
    return *this;
}

void CudaInteropSemaphore::waitInCuda(cudaStream_t stream) const
{
    cudaExternalSemaphoreWaitParams params{};
    checkCuda(cudaWaitExternalSemaphoresAsync(&m_external, &params, 1, stream),
              "cudaWaitExternalSemaphoresAsync");
}

void CudaInteropSemaphore::signalInCuda(cudaStream_t stream) const
{
    cudaExternalSemaphoreSignalParams params{};
    checkCuda(cudaSignalExternalSemaphoresAsync(&m_external, &params, 1, stream),
              "cudaSignalExternalSemaphoresAsync");
}

} // namespace lr::eulerian
