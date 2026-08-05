#include "core/scene/AreaLightVisual.hpp"

namespace lr
{

void buildAreaLightQuadMesh(Mesh &mesh, const Transform &transform, const AreaLight &light,
                             uint32_t materialIndex, const AreaLightVisualConfig &config)
{
    const glm::vec3 right   = transform.right() * (light.size.x * 0.5f);
    const glm::vec3 up      = transform.up() * (light.size.y * 0.5f);
    const glm::vec3 forward = transform.forward();
    const glm::vec3 center  = transform.position();

    mesh.setVertexCount(4);
    mesh.positions = {
        center - right - up,
        center + right - up,
        center + right + up,
        center - right + up,
    };
    mesh.setPerVertexArray<glm::vec3>(config.normalAttributeName, std::vector<glm::vec3>(4, forward));
    // Tangent = local right axis; w = +1 (no bitangent mirroring) matches geometry.frag's TBN build.
    mesh.setPerVertexArray<glm::vec4>(config.tangentAttributeName,
                                       std::vector<glm::vec4>(4, glm::vec4(transform.right(), 1.0f)));
    const std::vector<glm::vec2> uvs = { {0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f} };
    mesh.setPerVertexArray<glm::vec2>(config.uvAttributeName, uvs);

    mesh.setFaceCount(2);
    // Wound so the quad is visible (front-facing) from the `forward` side, matching the pass's
    // CCW-front backface culling.
    mesh.faces = { {0, 1, 2}, {0, 2, 3} };
    mesh.faceGroups = { materialIndex, materialIndex };
}

Material buildAreaLightMaterial(const AreaLight &light, const AreaLightVisualConfig &config)
{
    Material material;
    material.name = "AreaLightVisual";
    material.parameters[config.baseDiffuseName]   = MaterialParam::ColorRGBA{glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)};
    material.parameters[config.baseEmissiveName]  = MaterialParam::ColorRGB{light.color * light.intensity};
    material.parameters[config.baseRoughnessName] = MaterialParam::NormalizedFloat{1.0f};
    material.parameters[config.baseMetallicName]  = MaterialParam::NormalizedFloat{0.0f};
    return material;
}

} // namespace lr
