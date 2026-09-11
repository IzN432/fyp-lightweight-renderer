#pragma once

#include "core/passes/heatmap/HeatmapPass.hpp"
#include "core/scene/SceneManager.hpp"

#include <glm/vec3.hpp>

#include <span>

namespace lr
{

class VertexManager;

// Owns the interactive Laplace-Beltrami analysis workflow: calculation, heatmap visibility, GUI,
// and stale-result tracking after geometry edits.
class LaplaceBeltramiTool
{
public:
    LaplaceBeltramiTool(std::span<const glm::vec3> positions, std::span<const glm::uvec3> triangles,
                        SceneManager &sceneManager, HeatmapPass &heatmapPass, VertexManager &vertexManager);

    void onGui();

private:
    void calculate();
    void invalidate();

    std::span<const glm::vec3>  m_positions;
    std::span<const glm::uvec3> m_triangles;
    SceneManager &m_sceneManager;
    HeatmapPass  &m_heatmapPass;

    bool m_hasResult = false;
    bool m_isStale   = false;
};

} // namespace lr
