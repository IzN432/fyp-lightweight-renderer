#include "LaplaceBeltramiTool.hpp"

#include "LaplaceBeltramiOperator.hpp"
#include "core/editor/EditableMeshContext.hpp"
#include "core/editor/EditorStateController.hpp"
#include "core/utility/HeatmapColors.hpp"

#include <imgui.h>

#include <string>

namespace lr
{

namespace
{
constexpr std::string_view kAnalysisStateId = "analysis";
} // namespace

LaplaceBeltramiTool::LaplaceBeltramiTool(SceneManager &sceneManager) : m_sceneManager(sceneManager) {}

std::string_view LaplaceBeltramiTool::stateId() { return kAnalysisStateId; }

void LaplaceBeltramiTool::registerWith(EditorServices &services)
{
    m_states = &services.states();
    // Purely presentational: the heatmap replaces the shaded surface, and nothing about this state
    // is interactive, so it needs neither a per-frame hook nor a gizmo.
    m_states->registerState({
        .id           = std::string(kAnalysisStateId),
        .presentation = {.skinningEnabled = false, .heatmapVisible = true},
    });
}

void LaplaceBeltramiTool::onTargetChanged(const EditableMeshContext &target)
{
    discardResult();
    m_mesh = &target.mesh;
}

void LaplaceBeltramiTool::onTargetCleared()
{
    discardResult();
    m_mesh = nullptr;
}

void LaplaceBeltramiTool::discardResult()
{
    m_hasResult = false;
    if (m_states && m_states->isActive(kAnalysisStateId))
    {
        m_states->activateDefault();
    }
}

void LaplaceBeltramiTool::calculate()
{
    std::vector<glm::uvec3> triangles;
    triangles.reserve(m_mesh->faces().size());
    for (const glm::uvec3 &face : m_mesh->faces())
    {
        triangles.push_back(
            {m_mesh->positionIndices()[face.x], m_mesh->positionIndices()[face.y], m_mesh->positionIndices()[face.z]});
    }
    const auto magnitudes = LaplaceBeltramiOperator::calculateMagnitude(m_mesh->positions(), triangles);
    m_sceneManager.setSelectedMeshHeatmapColors(makeHeatmapColors(magnitudes));
    m_hasResult               = true;
    m_resultPositionsRevision = m_mesh->positionsRevision();
    m_resultTopologyRevision  = m_mesh->topologyRevision();
}

void LaplaceBeltramiTool::drawPanel()
{
    // TODO(step 7): this asks about the *hierarchy* selection, while m_mesh tracks the editable
    // target. They coincide today; separating them is what step 7 of the editor plan is for.
    const Scene &scene          = m_sceneManager.scene();
    const auto   selectedObject = scene.selectedObject();
    if (!m_mesh || !selectedObject || !SceneManager::isEditable(scene.getSceneObject(*selectedObject)))
    {
        discardResult();
        ImGui::TextDisabled("Select a mesh object to use Laplace-Beltrami analysis.");
        return;
    }

    if (ImGui::Button(m_hasResult ? "Recalculate" : "Calculate Laplace-Beltrami Operator"))
    {
        calculate();
    }

    bool showHeatmap = m_states && m_states->isActive(kAnalysisStateId);
    ImGui::BeginDisabled(!m_hasResult);
    if (ImGui::Checkbox("Show heatmap", &showHeatmap) && m_states)
    {
        if (showHeatmap)
        {
            m_states->activate(kAnalysisStateId);
        } else
        {
            m_states->activateDefault();
        }
    }
    ImGui::EndDisabled();

    if (!m_hasResult)
    {
        ImGui::TextDisabled("Calculate the operator to generate the heatmap.");
    } else if (m_resultPositionsRevision != m_mesh->positionsRevision() ||
               m_resultTopologyRevision != m_mesh->topologyRevision())
    {
        ImGui::TextColored(ImVec4(1.0f, 0.65f, 0.15f, 1.0f), "Mesh geometry changed. Heatmap values are out of date.");
    }
}

} // namespace lr
