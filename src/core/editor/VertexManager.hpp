#pragma once

#include "core/scene/Mesh.hpp"

#include <span>
#include <vector>
#include <unordered_set>
#include <glm/vec3.hpp>

namespace lr
{

class VertexManager
{
public:
    explicit VertexManager(Mesh &mesh) : m_mesh(&mesh) {}

    // Repoints this VertexManager at a different mesh (e.g. the Scene Hierarchy selection changed
    // to a different mesh object). GPU synchronization observes Mesh revisions independently.
    void rebind(Mesh &mesh) { m_mesh = &mesh; }

    const std::vector<glm::vec3> &getPositions() const { return m_mesh->positions(); }

    void updatePosition(uint32_t index, const glm::vec3 &newPosition);
    void translateSelectedVertices(const std::unordered_set<uint32_t> &indices, const glm::vec3 &translation);

    // Bulk position overwrite publishes one mesh revision. Needed by anything that moves many vertices to independent absolute
    // positions per call (e.g. an ARAP solve result), where translateSelectedVertices' shared
    // delta doesn't apply.
    void setPositions(const std::vector<uint32_t> &indices, const std::vector<glm::vec3> &positions);

private:
    Mesh *m_mesh;
};

} // namespace lr
