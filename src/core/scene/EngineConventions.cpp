#include "EngineConventions.hpp"

namespace lr::conventions
{

GltfLoaderConfig gltfLoaderConfig()
{
    return GltfLoaderConfig{
        .normalAttributeName          = normalAttribute,
        .tangentAttributeName         = tangentAttribute,
        .uvAttributeName              = uvAttribute,
        .diffuseTextureName           = baseColorTexture,
        .normalTextureName            = normalTexture,
        .metallicRoughnessTextureName = metallicRoughnessTexture,
        .emissiveTextureName          = emissiveTexture,
        .baseDiffuseName              = baseDiffuse,
        .baseRoughnessName            = baseRoughness,
        .baseMetallicName             = baseMetallic,
        .baseEmissiveName             = baseEmissive,
    };
}

SceneLoaderConfig sceneLoaderConfig()
{
    return SceneLoaderConfig{
        .gltf = gltfLoaderConfig(),
        .obj =
            {
                .normalAttributeName  = normalAttribute,
                .tangentAttributeName = tangentAttribute,
                .uvAttributeName      = uvAttribute,
                .diffuseTextureName   = baseColorTexture,
                .normalTextureName    = normalTexture,
                .roughnessTextureName = metallicRoughnessTexture,
                .emissiveTextureName  = emissiveTexture,
                .baseDiffuseName      = baseDiffuse,
                .baseRoughnessName    = baseRoughness,
                .baseMetallicName     = baseMetallic,
                .baseEmissiveName     = baseEmissive,
            },
    };
}

Material defaultMaterial()
{
    Material material;
    material.name                      = "Unused Material Slot";
    material.parameters[baseDiffuse]   = MaterialParam::ColorRGBA{glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)};
    material.parameters[baseEmissive]  = MaterialParam::ColorRGB{glm::vec3(0.0f)};
    material.parameters[baseRoughness] = MaterialParam::NormalizedFloat{1.0f};
    material.parameters[baseMetallic]  = MaterialParam::NormalizedFloat{0.0f};
    return material;
}

std::vector<std::string> geometryVertexAttributes() { return {normalAttribute, tangentAttribute, uvAttribute}; }

GpuMeshLayout geometryMeshLayout()
{
    MeshLayout layout;
    layout.addPerVertexAttr<glm::vec3>(normalAttribute)
        .addPerVertexAttr<glm::vec4>(tangentAttribute)
        .addPerVertexAttr<glm::vec2>(uvAttribute);

    GpuMeshLayout gpuLayout(layout);
    gpuLayout.mapPosition(0, 0, VK_FORMAT_R32G32B32_SFLOAT)
        .map(normalAttribute, 1, 1, VK_FORMAT_R32G32B32_SFLOAT)
        .map(tangentAttribute, 1, 2, VK_FORMAT_R32G32B32A32_SFLOAT)
        .map(uvAttribute, 1, 3, VK_FORMAT_R32G32_SFLOAT);
    return gpuLayout;
}

GpuMaterialLayout materialLayout()
{
    GpuMaterialLayout layout;
    layout.setStride(48)
        .addScalar(baseDiffuse, 0, sizeof(glm::vec4))
        .addScalar(baseEmissive, 16, sizeof(glm::vec3))
        .addScalar(baseRoughness, 32, sizeof(float))
        .addScalar(baseMetallic, 36, sizeof(float))
        .addTexture(baseColorTexture, VK_FORMAT_R8G8B8A8_SRGB)
        .addTexture(normalTexture, VK_FORMAT_R8G8B8A8_UNORM)
        .addTexture(metallicRoughnessTexture, VK_FORMAT_R8G8B8A8_UNORM)
        .addTexture(emissiveTexture, VK_FORMAT_R8G8B8A8_SRGB);
    return layout;
}

AreaLightVisualConfig areaLightVisualConfig()
{
    return AreaLightVisualConfig{
        .normalAttributeName  = normalAttribute,
        .tangentAttributeName = tangentAttribute,
        .uvAttributeName      = uvAttribute,
        .baseDiffuseName      = baseDiffuse,
        .baseEmissiveName     = baseEmissive,
        .baseRoughnessName    = baseRoughness,
        .baseMetallicName     = baseMetallic,
    };
}

} // namespace lr::conventions
