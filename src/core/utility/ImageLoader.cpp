#include "core/utility/ImageLoader.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <spdlog/spdlog.h>

#include <stdexcept>
namespace lr
{

LoadedImage loadImageFromFile(const std::filesystem::path &path)
{
    LoadedImage img;
    int w, h, channels;

    stbi_set_flip_vertically_on_load(true); // Flip Y-axis to match Vulkan's coordinate system
    img.pixels = stbi_load(path.string().c_str(), &w, &h, &channels, STBI_rgb_alpha);
    if (!img.pixels)
    {
        throw std::runtime_error("ImageLoader: failed to load '" + path.string() +
                                    "': " + stbi_failure_reason());
    }
    stbi_set_flip_vertically_on_load(false); // Reset to default for any future loads
    img.width  = static_cast<uint32_t>(w);
    img.height = static_cast<uint32_t>(h);
    return img;
}

LoadedImage loadImageFromFile(const std::filesystem::path &path, glm::vec4 defaultColor)
{
    try
    {
        return loadImageFromFile(path);
    }
    catch (const std::exception &e)
    {
        spdlog::warn("Failed to load image '{}': {}. Using default color.", path.string(), e.what());
        return LoadedImage::singlePixel(defaultColor);
    }
}

LoadedHdrImage loadHdrFromFile(const std::filesystem::path &path)
{
    LoadedHdrImage img;
    int w, h, channels;

    stbi_set_flip_vertically_on_load(true);
    float *data = stbi_loadf(path.string().c_str(), &w, &h, &channels, STBI_rgb_alpha);
    stbi_set_flip_vertically_on_load(false);

    if (!data)
    {
        throw std::runtime_error("ImageLoader: failed to load HDR '" + path.string() +
                                 "': " + stbi_failure_reason());
    }

    img.pixels = data;
    img.width  = static_cast<uint32_t>(w);
    img.height = static_cast<uint32_t>(h);
    return img;
}

} // namespace lr
