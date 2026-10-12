#include "core/framegraph/ResourceRegistry.hpp"
#include "core/vulkan/Allocator.hpp"
#include "core/vulkan/VulkanContext.hpp"
#include "features/eulerian_particle/CudaVolume.hpp"

#include <vulkan/vulkan.h>

#include <exception>
#include <iostream>
#include <utility>
#include <vector>

int main()
{
    try
    {
        std::vector<const char *> externalMemoryExtensions = {VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME};
#if defined(_WIN32)
        externalMemoryExtensions.push_back("VK_KHR_external_memory_win32");
#elif defined(__linux__)
        externalMemoryExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
#endif
        lr::VulkanContext context(lr::VulkanContext::Config{
            .appName               = "CUDA volume smoke",
            .enableValidation      = true,
            .extraDeviceExtensions = std::move(externalMemoryExtensions),
        });
        lr::Allocator allocator(context);
        lr::ResourceRegistry resources(context, allocator, {1, 1});
        resources.registerPersistentVolume("density", VK_FORMAT_R32_SFLOAT,
                                           VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
                                           VkExtent3D{32, 24, 16}, true);

        lr::eulerian::CudaVolume density(resources, context, "density");
        density.writeTestPattern(0.25f);
        std::cout << "CUDA wrote the exportable Vulkan density volume\n";
        return 0;
    } catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
