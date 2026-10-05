#include "LaplaceBeltramiTool.hpp"

#include "LaplaceBeltramiOperator.hpp"
#include "core/utility/HeatmapColors.hpp"

#include <imgui.h>

namespace lr
{

LaplaceBeltramiTool::LaplaceBeltramiTool(const Mesh &mesh, SceneManager &sceneManager)
    : m_sceneManager(sceneManager)
{
    rebind(mesh);
}

void LaplaceBeltramiTool::rebind(const Mesh &mesh)
{
    m_mesh = &mesh;
    m_hasResult = false;
}

void LaplaceBeltramiTool::calculate()
{
    std::vector<glm::uvec3> triangles;
    triangles.reserve(m_mesh->faces().size());
    for (const glm::uvec3 &face : m_mesh->faces())
    {
        triangles.push_back({m_mesh->positionIndices()[face.x], m_mesh->positionIndices()[face.y],
                             m_mesh->positionIndices()[face.z]});
    }
    const auto magnitudes = LaplaceBeltramiOperator::calculateMagnitude(m_mesh->positions(), triangles);
    m_sceneManager.setSelectedMeshHeatmapColors(makeHeatmapColors(magnitudes));
    m_hasResult = true;
    m_resultPositionsRevision = m_mesh->positionsRevision();
    m_resultTopologyRevision = m_mesh->topologyRevision();
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
    } else if (m_resultPositionsRevision != m_mesh->positionsRevision() ||
               m_resultTopologyRevision != m_mesh->topologyRevision())
    {
        ImGui::TextColored(ImVec4(1.0f, 0.65f, 0.15f, 1.0f),
                           "Mesh geometry changed. Heatmap values are out of date.");
    }

    ImGui::End();
}

} // namespace lr
