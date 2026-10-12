#include "CudaVolume.hpp"

#include "CudaVolumeKernels.cuh"
#include "core/framegraph/ResourceRegistry.hpp"
#include "core/vulkan/VulkanContext.hpp"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <utility>

#if defined(_WIN32)
#include <vulkan/vulkan_win32.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

namespace lr::eulerian
{
namespace
{

void checkCuda(cudaError_t result, const char *operation)
{
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("CudaVolume: ") + operation + ": " + cudaGetErrorString(result));
}

int selectCudaDevice(const VulkanContext &context)
{
    VkPhysicalDeviceIDProperties id{};
    id.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &id;
    vkGetPhysicalDeviceProperties2(context.getPhysicalDevice(), &properties);

    int count = 0;
    checkCuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    for (int device = 0; device < count; ++device)
    {
        cudaDeviceProp cudaProperties{};
        checkCuda(cudaGetDeviceProperties(&cudaProperties, device), "cudaGetDeviceProperties");
        if (std::memcmp(cudaProperties.uuid.bytes, id.deviceUUID, VK_UUID_SIZE) == 0)
            return device;
    }
    throw std::runtime_error("CudaVolume: no CUDA device matches the selected Vulkan physical device UUID");
}

cudaChannelFormatDesc channelDescription(VkFormat format)
{
    if (format == VK_FORMAT_R32_SFLOAT)
        return cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    if (format == VK_FORMAT_R32G32B32A32_SFLOAT)
        return cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    throw std::invalid_argument(
        "CudaVolume: only VK_FORMAT_R32_SFLOAT and VK_FORMAT_R32G32B32A32_SFLOAT volumes can be imported");
}

int channelCount(VkFormat format)
{
    return format == VK_FORMAT_R32G32B32A32_SFLOAT ? 4 : 1;
}

} // namespace

struct CudaVolume::Impl
{
    int                       device = 0;
    cudaExternalMemory_t      externalMemory = nullptr;
    cudaMipmappedArray_t      mipmappedArray = nullptr;
    cudaArray_t               array = nullptr;
    cudaSurfaceObject_t       surface = 0;
    cudaExtent                extent{};
    int                       channels = 1;

    ~Impl()
    {
        (void)cudaSetDevice(device);
        if (surface != 0)
            (void)cudaDestroySurfaceObject(surface);
        if (mipmappedArray != nullptr)
            (void)cudaFreeMipmappedArray(mipmappedArray);
        if (externalMemory != nullptr)
            (void)cudaDestroyExternalMemory(externalMemory);
    }
};

CudaVolume::CudaVolume(ResourceRegistry &resources, const VulkanContext &context, std::string resourceName)
    : m_resources(&resources), m_context(&context), m_resourceName(std::move(resourceName)), m_impl(std::make_unique<Impl>())
{
    resources.prepareImageForExternalWrite(m_resourceName);
    const AllocatedImage *image = resources.getImage(m_resourceName);
    if (image == nullptr || !image->isExternal() || image->deviceMemory == VK_NULL_HANDLE)
        throw std::invalid_argument("CudaVolume: resource '" + m_resourceName + "' is not an exportable image");
    if (image->extent.depth <= 1)
        throw std::invalid_argument("CudaVolume: resource '" + m_resourceName + "' is not a 3D volume");

    m_impl->device = selectCudaDevice(context);
    checkCuda(cudaSetDevice(m_impl->device), "cudaSetDevice");

    cudaExternalMemoryHandleDesc memoryDesc{};
    memoryDesc.size  = image->allocationSize;
    memoryDesc.flags = cudaExternalMemoryDedicated;

#if defined(_WIN32)
    const auto getMemoryHandle = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
        vkGetDeviceProcAddr(context.getDevice(), "vkGetMemoryWin32HandleKHR"));
    if (getMemoryHandle == nullptr)
        throw std::runtime_error("CudaVolume: vkGetMemoryWin32HandleKHR is unavailable");

    VkMemoryGetWin32HandleInfoKHR handleInfo{};
    handleInfo.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    handleInfo.memory     = image->deviceMemory;
    handleInfo.handleType = image->externalMemoryHandleType;
    HANDLE handle = nullptr;
    const VkResult handleResult = getMemoryHandle(context.getDevice(), &handleInfo, &handle);
    if (handleResult != VK_SUCCESS)
        throw std::runtime_error("CudaVolume: vkGetMemoryWin32HandleKHR failed");
    memoryDesc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
    memoryDesc.handle.win32.handle = handle;
    const cudaError_t importResult = cudaImportExternalMemory(&m_impl->externalMemory, &memoryDesc);
    CloseHandle(handle);
    checkCuda(importResult, "cudaImportExternalMemory");
#elif defined(__linux__)
    const auto getMemoryHandle = reinterpret_cast<PFN_vkGetMemoryFdKHR>(
        vkGetDeviceProcAddr(context.getDevice(), "vkGetMemoryFdKHR"));
    if (getMemoryHandle == nullptr)
        throw std::runtime_error("CudaVolume: vkGetMemoryFdKHR is unavailable");
    VkMemoryGetFdInfoKHR handleInfo{};
    handleInfo.sType      = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    handleInfo.memory     = image->deviceMemory;
    handleInfo.handleType = image->externalMemoryHandleType;
    int fd = -1;
    if (getMemoryHandle(context.getDevice(), &handleInfo, &fd) != VK_SUCCESS)
        throw std::runtime_error("CudaVolume: vkGetMemoryFdKHR failed");
    memoryDesc.type      = cudaExternalMemoryHandleTypeOpaqueFd;
    memoryDesc.handle.fd = fd;
    // CUDA takes ownership of an imported file descriptor, including on import failure.
    checkCuda(cudaImportExternalMemory(&m_impl->externalMemory, &memoryDesc), "cudaImportExternalMemory");
#else
#error CUDA/Vulkan external-memory import is only implemented for Windows and Linux
#endif

    cudaExternalMemoryMipmappedArrayDesc arrayDesc{};
    arrayDesc.formatDesc = channelDescription(image->format);
    arrayDesc.extent.width  = image->extent.width;
    arrayDesc.extent.height = image->extent.height;
    arrayDesc.extent.depth  = image->extent.depth;
    arrayDesc.flags      = cudaArraySurfaceLoadStore;
    arrayDesc.numLevels  = image->mipLevels;
    checkCuda(cudaExternalMemoryGetMappedMipmappedArray(&m_impl->mipmappedArray, m_impl->externalMemory, &arrayDesc),
              "cudaExternalMemoryGetMappedMipmappedArray");
    checkCuda(cudaGetMipmappedArrayLevel(&m_impl->array, m_impl->mipmappedArray, 0),
              "cudaGetMipmappedArrayLevel");

    cudaResourceDesc surfaceDesc{};
    surfaceDesc.resType         = cudaResourceTypeArray;
    surfaceDesc.res.array.array = m_impl->array;
    checkCuda(cudaCreateSurfaceObject(&m_impl->surface, &surfaceDesc), "cudaCreateSurfaceObject");
    m_impl->extent   = arrayDesc.extent;
    m_impl->channels = channelCount(image->format);
}

CudaVolume::~CudaVolume() = default;
CudaVolume::CudaVolume(CudaVolume &&) noexcept = default;
CudaVolume &CudaVolume::operator=(CudaVolume &&) noexcept = default;

namespace
{
void requireLive(const void *impl)
{
    if (impl == nullptr)
        throw std::logic_error("CudaVolume: cannot use a moved-from volume");
}
} // namespace

void CudaVolume::writeTestPattern(float timeSeconds)
{
    requireLive(m_impl.get());
    prepareForExternalWrite();
    checkCuda(cudaSetDevice(m_impl->device), "cudaSetDevice");
    checkCuda(launchDensityTestPattern(m_impl->surface, m_impl->extent, timeSeconds, nullptr),
              "launchDensityTestPattern");
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

void CudaVolume::clear()
{
    requireLive(m_impl.get());
    prepareForExternalWrite();
    checkCuda(cudaSetDevice(m_impl->device), "cudaSetDevice");
    checkCuda(launchClearVolume(m_impl->surface, m_impl->extent, m_impl->channels, nullptr), "launchClearVolume");
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

void CudaVolume::synchronize() const
{
    requireLive(m_impl.get());
    checkCuda(cudaSetDevice(m_impl->device), "cudaSetDevice");
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

void CudaVolume::prepareForExternalWrite()
{
    requireLive(m_impl.get());
    m_resources->prepareImageForExternalWrite(m_resourceName);
}

cudaSurfaceObject_t CudaVolume::surface() const
{
    requireLive(m_impl.get());
    return m_impl->surface;
}

cudaExtent CudaVolume::extent() const
{
    requireLive(m_impl.get());
    return m_impl->extent;
}

int CudaVolume::channels() const
{
    requireLive(m_impl.get());
    return m_impl->channels;
}

std::vector<float> CudaVolume::download() const
{
    requireLive(m_impl.get());
    checkCuda(cudaSetDevice(m_impl->device), "cudaSetDevice");
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    const std::size_t cells = m_impl->extent.width * m_impl->extent.height * m_impl->extent.depth;
    std::vector<float> host(cells * static_cast<std::size_t>(m_impl->channels));

    cudaMemcpy3DParms copy{};
    copy.srcArray = m_impl->array;
    copy.dstPtr   = make_cudaPitchedPtr(host.data(),
                                        m_impl->extent.width * sizeof(float) * static_cast<std::size_t>(m_impl->channels),
                                        m_impl->extent.width, m_impl->extent.height);
    copy.extent   = m_impl->extent;
    copy.kind     = cudaMemcpyDeviceToHost;
    checkCuda(cudaMemcpy3D(&copy), "cudaMemcpy3D");
    return host;
}

int CudaVolume::device() const
{
    requireLive(m_impl.get());
    return m_impl->device;
}

} // namespace lr::eulerian
