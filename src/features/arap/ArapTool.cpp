#include "ArapTool.hpp"

#include "core/math/LinearAlgebraHelpers.hpp"

#include <algorithm>
#include <imgui.h>

namespace lr
{

namespace
{
constexpr glm::vec3 kAnchorColor{0.2f, 0.4f, 1.0f};
constexpr glm::vec3 kHandleColor{1.0f, 0.6f, 0.1f};
} // namespace

ArapTool::ArapTool(SelectionManager &selectionManager, VertexManager &vertexManager, CommandManager &commandManager,
                   const Mesh &mesh)
    : m_selectionManager(selectionManager), m_mesh(&mesh),
      m_anchorRole(selectionManager.registerRole(kAnchorColor)),
      m_handleRole(selectionManager.registerRole(kHandleColor)),
      m_arapHandler(m_solver, vertexManager, selectionManager, m_handleRole, commandManager)
{
    m_selectionManager.registerRoleChangedCallback([this]() {
        onRoleChanged();
    });
}

void ArapTool::rebind(const Mesh &mesh)
{
    m_mesh = &mesh;
    m_solver.invalidate();
    m_lastSolveFailed = false;
}

void ArapTool::onRoleChanged()
{
    m_solver.invalidate();
}

void ArapTool::onSolveClicked()
{
    const auto anchors = m_selectionManager.getIndicesWithRole(m_anchorRole);
    const auto handles = m_selectionManager.getIndicesWithRole(m_handleRole);

    m_lastSolveFailed = !m_solver.precompute(*m_mesh, anchors, handles);
}

void ArapTool::onOverlayGui(const glm::mat4 &viewProj, VkExtent2D extent,
                            const glm::vec3 &selectionWorldCentroid)
{
    const bool hasSelection = !m_selectionManager.getSelectedIndices().empty();
    if (!hasSelection)
    {
        return;
    }

    if (hasSelection)
    {
        glm::vec2 screenPos = math::worldToScreenPixels(selectionWorldCentroid, viewProj, extent);
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

void ArapTool::onPanelGui(bool active)
{
    if (!active)
    {
        ImGui::TextDisabled("Press A to enable ARAP mode.");
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
        ImGui::Text("Last %s update: %.3f ms", stats.lastWasRelease ? "release" : "drag",
                    stats.lastInteractionMs);
    }
}

} // namespace lr
