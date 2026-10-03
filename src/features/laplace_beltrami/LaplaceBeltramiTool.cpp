#include "LaplaceBeltramiTool.hpp"

#include "LaplaceBeltramiOperator.hpp"
#include "core/editor/VertexManager.hpp"
#include "core/utility/HeatmapColors.hpp"

#include <imgui.h>

namespace lr
{

LaplaceBeltramiTool::LaplaceBeltramiTool(const Mesh &mesh, SceneManager &sceneManager, VertexManager &vertexManager)
    : m_sceneManager(sceneManager)
{
    rebind(mesh);
    vertexManager.registerUpdateCallback([this]() {
        invalidate();
    });
}

void LaplaceBeltramiTool::rebind(const Mesh &mesh)
{
    m_positions = mesh.positions();
    m_triangles.clear();
    m_triangles.reserve(mesh.faces().size());
    for (const glm::uvec3 &face : mesh.faces())
    {
        m_triangles.push_back({mesh.positionIndices()[face.x], mesh.positionIndices()[face.y],
                               mesh.positionIndices()[face.z]});
    }
    m_hasResult = false;
    m_isStale   = false;
}

void LaplaceBeltramiTool::calculate()
{
    const auto magnitudes = LaplaceBeltramiOperator::calculateMagnitude(m_positions, m_triangles);
    m_sceneManager.setSelectedMeshHeatmapColors(makeHeatmapColors(magnitudes));
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
