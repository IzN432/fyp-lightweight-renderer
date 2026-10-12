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
// context must outlive this object. For now writeTestPattern uses blocking synchronization; the
// surface/import lifetime is already suitable for later external-semaphore integration.
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

    // Flushes any queued CUDA work on this volume's device. Call before letting Vulkan read the
    // image again; proper external-semaphore signalling replaces this later.
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

    // Re-transitions the Vulkan image so CUDA may write it again. Each public write already does
    // this; a solver driving the kernels directly calls it once per step instead.
    void prepareForExternalWrite();

private:
    struct Impl;

    ResourceRegistry   *m_resources = nullptr;
    const VulkanContext *m_context   = nullptr;
    std::string          m_resourceName;
    std::unique_ptr<Impl> m_impl;
};

} // namespace lr::eulerian
