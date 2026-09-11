#include "LaplaceBeltramiTool.hpp"

#include "LaplaceBeltramiOperator.hpp"
#include "core/editor/VertexManager.hpp"
#include "core/utility/HeatmapColors.hpp"

#include <imgui.h>

namespace lr
{

LaplaceBeltramiTool::LaplaceBeltramiTool(std::span<const glm::vec3> positions,
                                             std::span<const glm::uvec3> triangles, SceneManager &sceneManager,
                                             HeatmapPass &heatmapPass, VertexManager &vertexManager)
    : m_positions(positions), m_triangles(triangles), m_sceneManager(sceneManager), m_heatmapPass(heatmapPass)
{
    vertexManager.registerUpdateCallback([this]() {
        invalidate();
    });
}

void LaplaceBeltramiTool::calculate()
{
    const auto magnitudes = LaplaceBeltramiOperator::calculateMagnitude(m_positions, m_triangles);
    m_sceneManager.setMainMeshHeatmapColors(makeHeatmapColors(magnitudes));
    m_hasResult = true;
    m_isStale   = false;
}

void LaplaceBeltramiTool::invalidate()
{
    if (m_hasResult)
    {
        m_isStale = true;
    }
}

void LaplaceBeltramiTool::onGui()
{
    ImGui::Begin("Laplace-Beltrami Heatmap");

    if (ImGui::Button(m_hasResult ? "Recalculate" : "Calculate Laplace-Beltrami Operator"))
    {
        calculate();
    }

    bool enabled = m_heatmapPass.isEnabled();
    ImGui::BeginDisabled(!m_hasResult);
    if (ImGui::Checkbox("Show heatmap", &enabled))
    {
        m_heatmapPass.setEnabled(enabled);
    }
    ImGui::EndDisabled();

    if (!m_hasResult)
    {
        ImGui::TextDisabled("Calculate the operator to generate the heatmap.");
    } else if (m_isStale)
    {
        ImGui::TextColored(ImVec4(1.0f, 0.65f, 0.15f, 1.0f),
                           "Mesh geometry changed. Heatmap values are out of date.");
    }

    ImGui::End();
}

} // namespace lr
