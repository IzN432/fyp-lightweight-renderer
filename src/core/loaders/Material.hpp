#pragma once

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace lr
{
 
struct MaterialImage
{
    std::string          name;   // for debugging purposes
    std::vector<uint8_t> pixels;
    uint32_t             width  = 0;
    uint32_t             height = 0;

    bool empty() const { return pixels.empty(); }

    static MaterialImage singlePixel(glm::vec4 color)
    {
        auto to8 = [](float f) {
            return static_cast<uint8_t>(std::clamp(f, 0.0f, 1.0f) * 255.0f);
        };
        MaterialImage img;
        img.width  = 1;
        img.height = 1;
        img.pixels = { to8(color.r), to8(color.g), to8(color.b), to8(color.a) };
        return img;
    }
};

using MaterialValue = std::variant<float, glm::vec2, glm::vec3, glm::vec4>;

struct Material
{
    // Disable copying but allow moving, since Materials can contain large textures.
    Material() = default;
    Material(const Material&) = delete;
    Material& operator=(const Material&) = delete;
    Material(Material&&) = default;
    Material& operator=(Material&&) = default;

    std::string name;
    std::unordered_map<std::string, MaterialImage> textures;
    std::unordered_map<std::string, MaterialValue> parameters;
};

class GpuMaterialLayout
{
public:
    struct ScalarMapping
    {
        std::string name;
        uint32_t byteOffset;
        size_t scalarSize;
    };
    struct TextureMapping
    {
        std::string materialTextureName;
        VkFormat format = VK_FORMAT_R8G8B8A8_UNORM; // default format for uploaded textures
        bool shouldGenerateMipmaps = true;
    };
    /**
     * Call to add a scalar to the layout. The byte offset is relative to the start of the material entry,
     * and scalarSize is the size of the scalar type in bytes (e.g. 4 for float, 16 for vec4).
     */
    GpuMaterialLayout &addScalar(const std::string &name, uint32_t byteOffset, size_t scalarSize)
    {
        m_scalars.push_back({name, byteOffset, scalarSize});
        return *this;
    }
    GpuMaterialLayout &setStride(uint32_t stride) { m_stride = stride; return *this; }
    GpuMaterialLayout &addTexture(const std::string &materialTextureName,
                                VkFormat format = VK_FORMAT_R8G8B8A8_UNORM, 
                                const bool shouldGenerateMipmaps = true)
    {
        m_textures.push_back({materialTextureName, format, shouldGenerateMipmaps});
        return *this;
    }
    const std::vector<ScalarMapping> &scalars() const { return m_scalars; }
    const std::vector<TextureMapping> &textures() const { return m_textures; }
    uint32_t stride() const { return m_stride; }
private:
    uint32_t m_stride = 0; // total byte size of one material entry in the shader
    std::vector<ScalarMapping> m_scalars;
    std::vector<TextureMapping> m_textures;
};

}  // namespace lr