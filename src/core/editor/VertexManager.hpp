#pragma once

#include <span>
#include <vector>
#include <functional>
#include <glm/vec3.hpp>

namespace lr
{

class VertexManager
{
public:
    VertexManager(std::vector<glm::vec3> &positions)
        : m_positions(positions) 
    {}

    const std::vector<glm::vec3> &getPositions() const { return m_positions; }

    void updatePosition(uint32_t index, const glm::vec3 &newPosition);
    void removeVertex(uint32_t index);
    void addVertex(const glm::vec3 &position);

    void translateSelectedVertices(const std::vector<uint32_t> &indices, const glm::vec3 &translation);

    // Wholesale replace every vertex position (e.g. the result of an ARAP solve, which moves the
    // whole mesh, not just the dragged handles). newPositions must be the same size as the
    // existing position array.
    void setPositions(std::vector<glm::vec3> newPositions);

    void registerUpdateCallback(std::function<void()> callback) { m_updateCallback = std::move(callback); }

private:
    std::vector<glm::vec3> &m_positions;
    std::function<void()> m_updateCallback;
};

}