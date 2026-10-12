#pragma once

#include <cuda_runtime_api.h>
#include <vulkan/vulkan.h>

namespace lr
{
class VulkanContext;
}

namespace lr::eulerian
{

// One exportable binary VkSemaphore imported into CUDA as an external semaphore, so a Vulkan
// submission and a CUDA stream can hand work to each other without either side blocking the host.
//
// Binary (not timeline) semaphores are used deliberately: CUDA's external-semaphore wait and
// signal are one-shot operations with exactly binary semantics, and importing a timeline
// semaphore additionally requires cudaExternalSemaphoreHandleTypeTimelineSemaphore* support that
// an OPAQUE_WIN32 import cannot assume. The cost is that each semaphore carries one in-flight
// signal at a time, which is what a one-step-at-a-time handshake needs anyway.
//
// A binary semaphore must be signalled exactly once per wait. Both sides therefore have to stay
// paired: every signalFromVulkan() is matched by one waitInCuda(), and every signalFromCuda() by
// one waitInVulkan(). Dropping one half deadlocks the next step, so the owning solver tracks
// whether a signal is outstanding rather than letting callers interleave freely.
class CudaInteropSemaphore
{
public:
    // Creates the semaphore on `context`'s device, exports a platform handle and imports it into
    // CUDA on `cudaDevice`. The context must outlive this object.
    CudaInteropSemaphore(const VulkanContext &context, int cudaDevice);
    ~CudaInteropSemaphore();

    CudaInteropSemaphore(const CudaInteropSemaphore &)            = delete;
    CudaInteropSemaphore &operator=(const CudaInteropSemaphore &) = delete;
    CudaInteropSemaphore(CudaInteropSemaphore &&) noexcept;
    CudaInteropSemaphore &operator=(CudaInteropSemaphore &&) noexcept;

    // Enqueues a wait on `stream` that completes once the Vulkan side signals. Returns without
    // blocking the host: the stream stalls on the device instead.
    void waitInCuda(cudaStream_t stream) const;

    // Enqueues a signal on `stream`, to be consumed by a Vulkan wait. Does not block the host.
    void signalInCuda(cudaStream_t stream) const;

    // The raw handle, for building VkSemaphoreSubmitInfo in a caller's own submission.
    VkSemaphore handle() const { return m_semaphore; }

private:
    const VulkanContext     *m_context  = nullptr;
    VkSemaphore              m_semaphore = VK_NULL_HANDLE;
    cudaExternalSemaphore_t  m_external  = nullptr;
    int                      m_device    = 0;
};

} // namespace lr::eulerian
