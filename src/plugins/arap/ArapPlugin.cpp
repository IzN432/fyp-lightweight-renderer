#include "ArapPlugin.hpp"

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

ArapPlugin::ArapPlugin(SelectionManager &selectionManager, VertexManager &vertexManager, CommandManager &commandManager,
                       const Mesh &mesh, VertexDragHandler &defaultHandler, std::vector<DragHandlerGizmo *> gizmos)
    : m_selectionManager(selectionManager), m_vertexManager(vertexManager), m_commandManager(commandManager),
      m_mesh(mesh), m_defaultHandler(defaultHandler), m_gizmos(std::move(gizmos)),
      m_anchorRole(selectionManager.registerRole(kAnchorColor)),
      m_handleRole(selectionManager.registerRole(kHandleColor)),
      m_arapHandler(m_solver, vertexManager, selectionManager, m_handleRole, commandManager)
{
    m_selectionManager.registerRoleChangedCallback([this]() { onRoleChanged(); });
}

void ArapPlugin::setModeActive(bool active)
{
    m_modeActive = active;
    if (!active)
        resetToDefaultHandler();
}

void ArapPlugin::onRoleChanged()
{
    m_solver.invalidate();
    resetToDefaultHandler();
}

void ArapPlugin::resetToDefaultHandler()
{
    for (DragHandlerGizmo *gizmo : m_gizmos)
        gizmo->setDragHandler(m_defaultHandler);
}

void ArapPlugin::onSolveClicked()
{
    const auto anchors = m_selectionManager.getIndicesWithRole(m_anchorRole);
    const auto handles = m_selectionManager.getIndicesWithRole(m_handleRole);

    m_lastSolveFailed = !m_solver.precompute(m_mesh, anchors, handles);
    if (!m_lastSolveFailed)
    {
        for (DragHandlerGizmo *gizmo : m_gizmos)
            gizmo->setDragHandler(m_arapHandler);
    }
}

void ArapPlugin::onGui(const glm::mat4 &viewProj, VkExtent2D extent, const glm::vec3 &selectionWorldCentroid)
{
    if (!m_modeActive)
        return;

    const bool hasSelection = !m_selectionManager.getSelectedIndices().empty();
    if (hasSelection)
    {
        glm::vec2 screenPos = math::worldToScreenPixels(selectionWorldCentroid, viewProj, extent);
        screenPos.x = std::clamp(screenPos.x, 8.0f, static_cast<float>(extent.width) - 120.0f);
        screenPos.y = std::clamp(screenPos.y, 8.0f, static_cast<float>(extent.height) - 90.0f);

        ImGui::SetNextWindowPos(ImVec2(screenPos.x, screenPos.y));
        ImGui::Begin("##ArapRoleAssign", nullptr,
                      ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoFocusOnAppearing);
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

    ImGui::Begin("ARAP");
    const bool canSolve = !m_selectionManager.getIndicesWithRole(m_anchorRole).empty()
                        && !m_selectionManager.getIndicesWithRole(m_handleRole).empty();
    ImGui::BeginDisabled(!canSolve);
    if (ImGui::Button("Solve"))
        onSolveClicked();
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Clear All"))
        m_selectionManager.clearAllRoles();
    if (m_lastSolveFailed)
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                            "Solve failed - every free vertex must be able to reach an anchor or handle");
    ImGui::End();
}

} // namespace lr
