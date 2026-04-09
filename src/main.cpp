#include "core/vulkan/VulkanContext.hpp"
#include "core/vulkan/Allocator.hpp"
#include "core/vulkan/ShaderLoader.hpp"
#include "core/window/Window.hpp"

#include <spdlog/spdlog.h>

#include <filesystem>

int main()
{
    lr::GlfwContext glfw;

    lr::Window window({.title = "lr-test"});

    auto extensions = lr::GlfwContext::getRequiredInstanceExtensions();

    lr::VulkanContext ctx({
        .appName = "lr-test",
        .enableValidation = true,
        .extraInstanceExtensions = extensions,
    });

    spdlog::info("Device: {}", ctx.getDeviceProperties().properties.deviceName);

    lr::Allocator allocator(ctx);

    // Vertex buffer on GPU, writable from CPU (staging use case)
    auto stagingBuffer = allocator.createBuffer(
        1024,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );
    spdlog::info("Staging buffer created, mapped at {}", stagingBuffer.info.pMappedData);
    allocator.destroy(stagingBuffer);

    {
        lr::ShaderModule testShader(ctx.getDevice(),
            std::filesystem::path(LR_SHADER_DIR) / "test.vert.spv" ); // test.vert → test.vert.spv
        spdlog::info("Shader loaded OK");
    }

    while (!window.shouldClose())
        window.pollEvents();

    return 0;
}
