#pragma once

#include <span>
#include <vector>
#include <unordered_set>
#include <functional>
#include <glm/vec3.hpp>

namespace lr
{

class VertexManager
{
public:
    VertexManager(std::vector<glm::vec3> &positions) : m_positions(positions) {}

    const std::vector<glm::vec3> &getPositions() const { return m_positions; }

    void updatePosition(uint32_t index, const glm::vec3 &newPosition);
    void removeVertex(uint32_t index);
    void addVertex(const glm::vec3 &position);

    void translateSelectedVertices(const std::unordered_set<uint32_t> &indices, const glm::vec3 &translation);

    // Bulk position overwrite — writes every (indices[i], positions[i]) pair, then fires the
    // update callback once. Needed by anything that moves many vertices to independent absolute
    // positions per call (e.g. an ARAP solve result), where translateSelectedVertices' shared
    // delta doesn't apply.
    void setPositions(const std::vector<uint32_t> &indices, const std::vector<glm::vec3> &positions);

    // Multiple systems observe geometry edits (GPU upload, analysis invalidation, ...).
    void registerUpdateCallback(std::function<void()> callback) { m_updateCallbacks.push_back(std::move(callback)); }

private:
    std::vector<glm::vec3> &m_positions;
    std::vector<std::function<void()>> m_updateCallbacks;

    void notifyUpdateCallbacks();
};

} // namespace lr
