#include "LaplaceBeltramiTool.hpp"

#include "LaplaceBeltramiOperator.hpp"
#include "core/editor/VertexManager.hpp"
#include "core/utility/HeatmapColors.hpp"

#include <imgui.h>

namespace lr
{

LaplaceBeltramiTool::LaplaceBeltramiTool(std::span<const glm::vec3> positions,
                                         std::span<const glm::uvec3> triangles, SceneManager &sceneManager,
                                         VertexManager &vertexManager)
    : m_positions(positions), m_triangles(triangles), m_sceneManager(sceneManager)
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
    const ImGuiViewport *viewport = ImGui::GetMainViewport();
    const ImVec2 windowSize(viewport->WorkSize.x * 0.24f, viewport->WorkSize.y * 0.32f);
    ImGui::SetNextWindowPos(viewport->WorkPos, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(windowSize, ImGuiCond_FirstUseEver);
    ImGui::Begin("Laplace-Beltrami Heatmap");

    if (ImGui::Button(m_hasResult ? "Recalculate" : "Calculate Laplace-Beltrami Operator"))
    {
        calculate();
    }

    bool enabled = m_sceneManager.editorMode() == EditorMode::Analysis;
    ImGui::BeginDisabled(!m_hasResult);
    if (ImGui::Checkbox("Show heatmap", &enabled))
    {
        m_sceneManager.setEditorMode(enabled ? EditorMode::Analysis : EditorMode::View);
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
