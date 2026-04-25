#pragma once

#include <glm/glm.hpp>

namespace lr
{

struct MeshData
{
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec4> tangents;
    std::vector<glm::vec2> uvs;
    std::vector<glm::uvec3> faces;
    std::vector<uint32_t> faceGroups;
};

void generateTangents(MeshData &meshData);

} // namespace lr