#include "core/upload/MaterialUploader.hpp"

#include <stdexcept>

namespace lr
{


MaterialUploader::MaterialUploader(ResourceRegistry &registry)
    : m_registry(registry)
{
}

MaterialUploadResult MaterialUploader::upload(const std::vector<Material> &materials,
                                            const GpuMaterialLayout &gpuLayout,
                                            const std::string &namePrefix)
{
    MaterialUploadResult result;

    // SECTION 1 - Upload scalar parameters to GPU buffer

    if (gpuLayout.scalars().empty())
    {
        throw std::runtime_error("GpuMaterialLayout must have at least one scalar mapping");
    }
    if (gpuLayout.stride() == 0)
    {
        throw std::runtime_error("GpuMaterialLayout stride must be greater than zero");
    }

    auto &scalars = gpuLayout.scalars();

    std::vector<std::byte> materialInfoBuffer;
    materialInfoBuffer.resize(materials.size() * gpuLayout.stride());

    for (size_t i = 0; i < materials.size(); ++i)
    {
        const auto &material = materials[i];

        for (const auto &scalar : scalars)
        {
            if (!material.parameters.contains(scalar.name))
            {
                throw std::runtime_error("Material is missing parameter: " + scalar.name);
            }

            const auto &materialParameter = material.parameters.at(scalar.name);
            std::byte *dst = materialInfoBuffer.data() + (i * gpuLayout.stride()) + scalar.byteOffset;

            std::visit([&](const auto &value) {
                using T = std::decay_t<decltype(value)>;
                if (scalar.scalarSize != sizeof(T))
                {
                    throw std::runtime_error("Material parameter " + scalar.name +
                        " size mismatch: layout expects " + std::to_string(scalar.scalarSize) +
                        " but type is " + std::to_string(sizeof(T)));
                }
                std::memcpy(dst, &value, sizeof(T));
            }, materialParameter);
        }
    }

    const std::string name = namePrefix + "_info";
    m_registry.uploadBuffer(name, 
                            materialInfoBuffer.data(), 
                            static_cast<VkDeviceSize>(materialInfoBuffer.size()),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    result.materialInfoBufferName = name;
    
    // SECTION 2 - Upload images to GPU textures

    const std::array<uint8_t, 4> kFallbackPixel = {255, 255, 255, 255};

    for (const auto &[materialTextureName, format, shouldGenerateMipmaps] : gpuLayout.textures())
    {
        const std::string resourceName = namePrefix + "_tex_" + materialTextureName;
        for (size_t i = 0; i < materials.size(); ++i)
        {
            const auto &material = materials[i];
            const MaterialImage *img = nullptr;
            if (material.textures.contains(materialTextureName))
                img = &material.textures.at(materialTextureName);

            if (img && !img->empty())
            {
                m_registry.uploadArrayImage(resourceName, i,
                                            img->pixels.data(), img->width, img->height,
                                            format, shouldGenerateMipmaps);
            }
            else
            {
                m_registry.uploadArrayImage(resourceName, i,
                                            kFallbackPixel.data(), 1, 1,
                                            format, false);
            }
        }
        result.textureNameMap[materialTextureName] = resourceName;
    }

    return result;
}

} // namespace lr