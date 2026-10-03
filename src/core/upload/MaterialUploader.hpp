#pragma once

#include "core/framegraph/ResourceRegistry.hpp"

#include "core/loaders/Material.hpp"
#include "core/loaders/MaterialStore.hpp"

#include <span>

namespace lr
{

struct MaterialUploadResult
{
    std::string                                  materialInfoBufferName;
    std::unordered_map<std::string, std::string> textureNameMap; // materialTextureName -> registry resource name
};

/**
 * Upload the scalar parameters of a Material to a GPU buffer, and create GPU textures for the material's images.
 */
class MaterialUploader
{
public:
    explicit MaterialUploader(ResourceRegistry &registry);

    MaterialUploadResult upload(const std::vector<const Material *> &materials, const GpuMaterialLayout &gpuLayout,
                                const std::string &namePrefix = "material");

    void update(const std::vector<const Material *> &materials, const GpuMaterialLayout &gpuLayout,
                const MaterialUploadResult &result);

    // Replaces texture-array slots for newly acquired material handles. Existing descriptor sets
    // must be rebuilt by the caller after the image views change.
    void updateTextures(const std::vector<const Material *> &materials, const GpuMaterialLayout &gpuLayout,
                        const MaterialUploadResult &result, std::span<const MaterialHandle> handles);

private:
    ResourceRegistry &m_registry;
};

} // namespace lr
