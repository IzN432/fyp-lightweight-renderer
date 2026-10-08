#include "ArapTool.hpp"

#include "core/editor/EditableMeshContext.hpp"
#include "core/editor/EditorShortcuts.hpp"
#include "core/editor/EditorStateController.hpp"
#include "core/editor/VertexCentroid.hpp"
#include "core/math/LinearAlgebraHelpers.hpp"
#include "core/scene/SceneObject.hpp"
#include "core/scene/TransformComponent.hpp"

#include <GLFW/glfw3.h>
#include <imgui.h>

#include <algorithm>
#include <string>

namespace lr
{

namespace
{
constexpr glm::vec3        kAnchorColor{0.2f, 0.4f, 1.0f};
constexpr glm::vec3        kHandleColor{1.0f, 0.6f, 0.1f};
constexpr std::string_view kArapStateId = "arap";
} // namespace

ArapTool::ArapTool(SelectionManager &selectionManager, VertexManager &vertexManager, CommandManager &commandManager)
    : m_selectionManager(selectionManager), m_vertexManager(vertexManager),
      m_anchorRole(selectionManager.registerRole(kAnchorColor)),
      m_handleRole(selectionManager.registerRole(kHandleColor)),
      m_arapHandler(m_solver, vertexManager, selectionManager, m_handleRole, commandManager)
{
    m_roleChangedConnection = m_selectionManager.registerRoleChangedCallback([this]() {
        onRoleChanged();
    });
}

std::string_view ArapTool::stateId() { return kArapStateId; }

void ArapTool::registerWith(EditorServices &services)
{
    m_states = &services.states();
    m_states->registerState({
        .id           = std::string(kArapStateId),
        .presentation = {.skinningEnabled = false, .vertexPointsVisible = true, .vertexSelectionActive = true},
        .update =
            [this](const EditorFrameContext &frame) {
                onOverlayGui(frame.viewProjection, frame.extent);
            },
        .gizmoRequest = [this](const EditorFrameContext &) -> GizmoRequest {
            if (!hasDeformationTarget() || m_arapHandler.indices().empty())
            {
                return {};
            }
            return TranslateGizmoRequest{
                .origin  = worldCentroidOf(m_arapHandler.indices()),
                .handler = &m_arapHandler,
            };
        },
    });

    // The key belongs to the feature, not to whatever hosts it, so the host never learns that ARAP
    // is reachable by pressing A.
    services.shortcuts().add({.key = GLFW_KEY_A}, [this] {
        // With no editable target there is nothing to deform, and the state's vertex points and
        // vertex selection would have no mesh behind them, so entering is refused the way the
        // editor refuses TAB. Leaving stays available unconditionally.
        if (!m_mesh && !m_states->isActive(kArapStateId))
        {
            return;
        }
        m_states->toggle(kArapStateId);
    });
}

void ArapTool::onTargetChanged(const EditableMeshContext &target)
{
    // A factorization is only valid for the mesh it was built from, so a new target always drops
    // ARAP mode rather than leaving a stale solve reachable from the gizmo.
    retreatFromArapMode();
    m_object = &target.object;
    m_mesh   = &target.mesh;
    m_arapHandler.setTargetTransform(&target.object.getComponent<TransformComponent>());
    m_solver.invalidate();
    m_lastSolveFailed = false;
}

void ArapTool::onTargetCleared()
{
    retreatFromArapMode();
    m_object = nullptr;
    m_mesh   = nullptr;
    m_arapHandler.setTargetTransform(nullptr);
    m_solver.invalidate();
    m_lastSolveFailed = false;
}

void ArapTool::retreatFromArapMode()
{
    if (m_states && m_states->isActive(kArapStateId))
    {
        m_states->activateDefault();
    }
}

glm::vec3 ArapTool::worldCentroidOf(const std::unordered_set<uint32_t> &indices) const
{
    if (!m_object)
    {
        return glm::vec3(0.0f);
    }
    return worldVertexCentroid(m_vertexManager, m_object->getComponent<TransformComponent>(), indices);
}

void ArapTool::onRoleChanged() { m_solver.invalidate(); }

void ArapTool::onSolveClicked()
{
    const auto anchors = m_selectionManager.getIndicesWithRole(m_anchorRole);
    const auto handles = m_selectionManager.getIndicesWithRole(m_handleRole);

    m_lastSolveFailed = !m_solver.precompute(*m_mesh, anchors, handles);
}

void ArapTool::onOverlayGui(const glm::mat4 &viewProj, VkExtent2D extent)
{
    const std::unordered_set<uint32_t> &selected = m_selectionManager.getSelectedIndices();
    if (selected.empty())
    {
        return;
    }

    {
        glm::vec2 screenPos = math::worldToScreenPixels(worldCentroidOf(selected), viewProj, extent);
        screenPos.x         = std::clamp(screenPos.x, 8.0f, static_cast<float>(extent.width) - 120.0f);
        screenPos.y         = std::clamp(screenPos.y, 8.0f, static_cast<float>(extent.height) - 90.0f);

        ImGui::SetNextWindowPos(ImVec2(screenPos.x, screenPos.y));
        ImGui::Begin("##ArapRoleAssign", nullptr,
                     ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                         ImGuiWindowFlags_NoFocusOnAppearing);
        if (ImGui::Button("Anchor"))
        {
            m_selectionManager.classifySelectionAs(m_anchorRole);
            m_selectionManager.clearSelection();
        }
        ImGui::SameLine();
        if (ImGui::Button("Handle"))
        {
            m_selectionManager.classifySelectionAs(m_handleRole);
            m_selectionManager.clearSelection();
        }
        ImGui::SameLine();
        if (ImGui::Button("Clear"))
        {
            // kNoRole resets just the current selection back to unclassified — distinct from the
            // "Clear All" button below, which resets every vertex regardless of selection.
            m_selectionManager.classifySelectionAs(kNoRole);
            m_selectionManager.clearSelection();
        }
        ImGui::End();
    }
}

void ArapTool::drawPanel()
{
    if (!m_states || !m_states->isActive(kArapStateId))
    {
        ImGui::TextDisabled(m_mesh ? "Press A to enable ARAP mode."
                                   : "Select a mesh object to use ARAP deformation.");
        return;
    }

    ImGui::TextWrapped("Select vertices to classify anchors and handles.");

    const bool canSolve = !m_selectionManager.getIndicesWithRole(m_anchorRole).empty() &&
                          !m_selectionManager.getIndicesWithRole(m_handleRole).empty();
    ImGui::BeginDisabled(!canSolve);
    if (ImGui::Button("Solve"))
    {
        onSolveClicked();
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Clear All"))
    {
        m_selectionManager.clearAllRoles();
    }
    if (m_lastSolveFailed)
    {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                           "Solve failed - every free vertex must be able to reach an anchor or handle");
    }

    const ArapPerformanceStats &stats = m_solver.performanceStats();
    ImGui::SeparatorText("Performance");
    ImGui::Text("Solver: %.*s", static_cast<int>(m_solver.backendName().size()), m_solver.backendName().data());
    ImGui::Text("Mesh: %zu vertices, %zu triangles", stats.vertexCount, stats.triangleCount);
    ImGui::Text("Constraints: %zu", stats.constraintCount);
    ImGui::Text("Precompute: %.3f ms (backend: %.3f ms)", stats.precomputeMs, stats.solverPrecomputeMs);
    if (stats.solveCount > 0)
    {
        ImGui::Text("Last solve: %.3f ms (%d iteration%s)", stats.lastSolveMs, stats.lastIterations,
                    stats.lastIterations == 1 ? "" : "s");
        ImGui::Text("Estimated per iteration: %.3f ms", stats.lastSolveMs / stats.lastIterations);
        ImGui::Text("Solve avg/min/max: %.3f / %.3f / %.3f ms", stats.averageSolveMs, stats.minSolveMs,
                    stats.maxSolveMs);
        ImGui::Text("Solve samples: %llu", static_cast<unsigned long long>(stats.solveCount));
        ImGui::Text("Last %s update: %.3f ms", stats.lastWasRelease ? "release" : "drag", stats.lastInteractionMs);
    }
}

} // namespace lr
