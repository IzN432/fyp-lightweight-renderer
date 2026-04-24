#include "core/framegraph/ResourceRegistry.hpp"

#include "core/loaders/Material.hpp"

namespace lr
{

struct MaterialUploadResult
{
    std::string materialInfoBufferName;
    std::unordered_map<std::string, std::string> textureNameMap; // materialTextureName -> registry resource name
};

/**
 * Upload the scalar parameters of a Material to a GPU buffer, and create GPU textures for the material's images.
 */
class MaterialUploader
{
public:
    explicit MaterialUploader(ResourceRegistry &registry);

    MaterialUploadResult upload(const std::vector<Material> &materials,
                                const GpuMaterialLayout &gpuLayout,
                                const std::string &namePrefix = "material");

private:
    ResourceRegistry &m_registry;
};

} // namespace lr