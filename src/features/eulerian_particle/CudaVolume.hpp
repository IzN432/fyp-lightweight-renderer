#pragma once

#include <memory>
#include <string>

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

    const std::string &resourceName() const { return m_resourceName; }

private:
    struct Impl;

    ResourceRegistry   *m_resources = nullptr;
    const VulkanContext *m_context   = nullptr;
    std::string          m_resourceName;
    std::unique_ptr<Impl> m_impl;
};

} // namespace lr::eulerian
