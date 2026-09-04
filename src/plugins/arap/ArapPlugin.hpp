#pragma once

#include "ArapSolver.hpp"
#include "ArapDragHandler.hpp"

#include "core/editor/VertexDragHandler.hpp"
#include "core/editor/VertexManager.hpp"
#include "core/editor/selection/SelectionManager.hpp"
#include "core/editor/selection/VertexRole.hpp"
#include "core/editor/command/CommandManager.hpp"
#include "core/editor/gizmo/DragHandlerGizmo.hpp"
#include "core/scene/Mesh.hpp"

#include <vector>
#include <vulkan/vulkan.h>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace lr
{

// The ARAP feature's entire public surface — the only class a host application (main.cpp today,
// potentially a Python-driven one later) needs to name. Owns the anchor/handle role registration,
// the solver, the glue drag handler, and the popup/Solve-button UI.
class ArapPlugin
{
public:
    ArapPlugin(SelectionManager &selectionManager, VertexManager &vertexManager, CommandManager &commandManager,
               const Mesh &mesh, VertexDragHandler &defaultHandler, std::vector<DragHandlerGizmo *> gizmos);

    void setModeActive(bool active);
    bool isModeActive() const { return m_modeActive; }

    // Called once per frame from the host's ImGui callback. Draws the anchor/handle popup (only
    // while a selection exists) and the Solve button. selectionWorldCentroid is only meaningful
    // while a selection exists — the caller computes it the same way it already does for gizmo
    // placement (see main.cpp's centroid loop).
    void onGui(const glm::mat4 &viewProj, VkExtent2D extent, const glm::vec3 &selectionWorldCentroid);

private:
    void onRoleChanged();
    void onSolveClicked();
    void resetToDefaultHandler();

    SelectionManager               &m_selectionManager;
    VertexManager                  &m_vertexManager;
    CommandManager                 &m_commandManager;
    const Mesh                     &m_mesh;
    VertexDragHandler              &m_defaultHandler;
    std::vector<DragHandlerGizmo *> m_gizmos;

    VertexRoleId m_anchorRole;
    VertexRoleId m_handleRole;

    ArapSolver      m_solver;
    ArapDragHandler m_arapHandler;

    bool m_modeActive      = false;
    bool m_lastSolveFailed = false;
};

} // namespace lr
