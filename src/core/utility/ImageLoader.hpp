#pragma once

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>

#include <vulkan/vulkan.h>

namespace lr
{

// Decoded image from disk. Pixel data is always 4 channels (RGBA), 8 bits per channel.
// Call free() when done uploading, or rely on the destructor.
struct LoadedImage
{
    uint8_t  *pixels = nullptr;
    uint32_t   width  = 0;
    uint32_t   height = 0;

    // Always VK_FORMAT_R8G8B8A8_SRGB — stb forces 4 channels.
    static constexpr VkFormat format = VK_FORMAT_R8G8B8A8_SRGB;

    LoadedImage() = default;

    LoadedImage(const LoadedImage &)            = delete;
    LoadedImage &operator=(const LoadedImage &) = delete;

    LoadedImage(LoadedImage &&other) noexcept
        : pixels(other.pixels), width(other.width), height(other.height)
    {
        other.pixels = nullptr;
        other.width  = 0;
        other.height = 0;
    }

    LoadedImage &operator=(LoadedImage &&other) noexcept
    {
        if (this != &other)
        {
            free();
            pixels       = other.pixels;
            width        = other.width;
            height       = other.height;
            other.pixels = nullptr;
            other.width  = 0;
            other.height = 0;
        }
        return *this;
    }

    ~LoadedImage() { free(); }

    void free()
    {
        if (pixels)
        {
            std::free(pixels); // stbi_image_free is just free()
            pixels = nullptr;
        }
    }

    bool empty() const { return pixels == nullptr; }
};

// Load PNG/JPG/BMP/TGA from disk. Always decodes to RGBA8.
// Throws std::runtime_error if the file cannot be loaded.
LoadedImage loadImageFromFile(const std::filesystem::path &path);

// Decoded HDR image. Pixel data is 4-channel float (RGBA), 32 bits per channel.
// Upload with VK_FORMAT_R32G32B32A32_SFLOAT.
struct LoadedHdrImage
{
    float   *pixels = nullptr;
    uint32_t width  = 0;
    uint32_t height = 0;

    static constexpr VkFormat format = VK_FORMAT_R32G32B32A32_SFLOAT;

    LoadedHdrImage() = default;
    LoadedHdrImage(const LoadedHdrImage &)            = delete;
    LoadedHdrImage &operator=(const LoadedHdrImage &) = delete;

    LoadedHdrImage(LoadedHdrImage &&o) noexcept
        : pixels(o.pixels), width(o.width), height(o.height)
    { o.pixels = nullptr; o.width = 0; o.height = 0; }

    LoadedHdrImage &operator=(LoadedHdrImage &&o) noexcept
    {
        if (this != &o) { free(); pixels = o.pixels; width = o.width; height = o.height;
                          o.pixels = nullptr; o.width = 0; o.height = 0; }
        return *this;
    }

    ~LoadedHdrImage() { free(); }
    void free() { if (pixels) { std::free(pixels); pixels = nullptr; } }
    bool empty() const { return pixels == nullptr; }
};

// Load an HDR equirectangular image (.hdr / .exr via stb).
// Returns 4-channel float data (RGBA32F). Y-axis is flipped to match Vulkan.
// Throws std::runtime_error if the file cannot be loaded.
LoadedHdrImage loadHdrFromFile(const std::filesystem::path &path);

} // namespace lr
