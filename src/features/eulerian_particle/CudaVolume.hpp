#pragma once

#include <cuda_runtime_api.h>

#include <memory>
#include <string>
#include <vector>

namespace lr
{
class ResourceRegistry;
class VulkanContext;
}

namespace lr::eulerian
{

// Imports one exportable ResourceRegistry volume into CUDA as a surface. The registry and Vulkan
// context must outlive this object.
//
// The convenience writers here (writeTestPattern, clear) and download() synchronize on the host,
// which suits one-off setup and tests. A solver driving many volumes per frame should not: it
// orders CUDA against Vulkan with a CudaInteropSemaphore pair instead, calling the registry's
// releaseImagesToExternalWrite()/acquireImagesFromExternalWrite() around its own kernel launches
// and leaving the host free. See SmokeSolver for that pattern.
class CudaVolume
{
public:
    CudaVolume(ResourceRegistry &resources, const VulkanContext &context, std::string resourceName);
    ~CudaVolume();

    CudaVolume(const CudaVolume &)            = delete;
    CudaVolume &operator=(const CudaVolume &) = delete;
    CudaVolume(CudaVolume &&) noexcept;
    CudaVolume &operator=(CudaVolume &&) noexcept;

    void writeTestPattern(float timeSeconds = 0.0f);

    // Fills the volume with zeroes. Imported Vulkan memory starts undefined, so a solver must
    // clear every field it accumulates into before the first step.
    void clear();

    // Blocks until queued CUDA work on this volume's device has finished. A host wait, so it is
    // for setup and debugging; per-frame code uses the semaphore handshake described above.
    void synchronize() const;

    // Handles the kernels need. The surface stays valid for this object's lifetime.
    cudaSurfaceObject_t surface() const;
    cudaExtent          extent() const;
    int                 channels() const;
    int                 device() const;

    // Copies the whole volume back to the host, channels() floats per cell in x-major order.
    // Intended for tests and debugging; it synchronizes and is not cheap.
    std::vector<float> download() const;

    const std::string &resourceName() const { return m_resourceName; }

    // Re-transitions the Vulkan image so CUDA may write it again, waiting on the host for the
    // barrier. Each public write above already does this. A solver driving the kernels directly
    // uses the registry's semaphore handshake instead, which transitions every volume in one
    // submission without a host wait.
    void prepareForExternalWrite();

private:
    struct Impl;

    ResourceRegistry   *m_resources = nullptr;
    const VulkanContext *m_context   = nullptr;
    std::string          m_resourceName;
    std::unique_ptr<Impl> m_impl;
};

} // namespace lr::eulerian
