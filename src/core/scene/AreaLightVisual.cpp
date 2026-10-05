#include "core/scene/AreaLightVisual.hpp"

#include <array>

namespace lr
{

void buildAreaLightQuadMesh(Mesh &mesh, const TransformComponent &transform, const AreaLight &light,
                            MaterialHandle materialHandle, const AreaLightVisualConfig &config)
{
    const Transform &spatialTransform = transform.transform();
    const glm::vec3 right   = spatialTransform.right() * (light.size.x * 0.5f);
    const glm::vec3 up      = spatialTransform.up() * (light.size.y * 0.5f);
    const glm::vec3 forward = spatialTransform.forward();
    const glm::vec3 center  = spatialTransform.position();

    const std::array<glm::vec3, 4> corners = {
        center - right - up,
        center + right - up,
        center + right + up,
        center - right + up,
    };
    // Vertices 0-3: the front face, wound so it is visible (front-facing) from the `forward` side, the
    // side the light emits towards (see CalcAreaLight in pbr.frag), under GeometryPass's backface
    // culling. Vertices 4-7: the back face, wound the other way, for two-sided lights. A one-sided
    // light collapses it to a point instead of dropping it, so switching sides changes only vertex data.
    std::vector<glm::vec3> positions(corners.begin(), corners.end());
    for (const glm::vec3 &corner : corners)
    {
        positions.push_back(light.twoSided ? corner : center);
    }
    std::vector<uint32_t>   positionIndices = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<glm::uvec3> faces           = {{0, 2, 1}, {0, 3, 2}, {4, 5, 6}, {4, 6, 7}};
    // Once established, moving/resizing a light (or switching sides) changes vertex values, not
    // connectivity, so topology consumers (indices, skin position mappings) stay valid.
    if (mesh.positionIndices() == positionIndices && mesh.faces() == faces &&
        mesh.uniquePositionCount() == positions.size())
    {
        mesh.setPositions(positions);
    } else
    {
        mesh.setTopology(std::move(positions), std::move(positionIndices), std::move(faces));
    }

    std::vector<glm::vec3> normals(4, forward);
    normals.insert(normals.end(), 4, -forward);
    mesh.setPerVertexArray<glm::vec3>(config.normalAttributeName, normals);
    // Tangent = local right axis; w = +1 (no bitangent mirroring) matches geometry.frag's TBN build.
    mesh.setPerVertexArray<glm::vec4>(config.tangentAttributeName,
                                      std::vector<glm::vec4>(8, glm::vec4(spatialTransform.right(), 1.0f)));
    const std::vector<glm::vec2> uvs = {{0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f},
                                        {0.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 1.0f}};
    mesh.setPerVertexArray<glm::vec2>(config.uvAttributeName, uvs);

    mesh.setFaceGroups({materialHandle, materialHandle, materialHandle, materialHandle});
}

Material buildAreaLightMaterial(const AreaLight &light, const AreaLightVisualConfig &config)
{
    Material material;
    material.name                                 = "AreaLightVisual";
    material.parameters[config.baseDiffuseName]   = MaterialParam::ColorRGBA{glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)};
    material.parameters[config.baseEmissiveName]  = MaterialParam::ColorRGB{light.color * light.intensity};
    material.parameters[config.baseRoughnessName] = MaterialParam::NormalizedFloat{1.0f};
    material.parameters[config.baseMetallicName]  = MaterialParam::NormalizedFloat{0.0f};
    return material;
}

} // namespace lr
