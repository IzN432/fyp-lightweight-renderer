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
    const Scene &scene = m_sceneManager.scene();
    const auto   selectedObject = scene.selectedObject();
    if (!selectedObject || !SceneManager::isEditable(scene.getSceneObject(*selectedObject)))
    {
        m_hasResult = false;
        if (m_sceneManager.editorMode() == EditorMode::Analysis)
        {
            m_sceneManager.setEditorMode(EditorMode::View);
        }
        ImGui::TextDisabled("Select a mesh object to use Laplace-Beltrami analysis.");
        return;
    }

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
}

} // namespace lr
