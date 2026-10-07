#pragma once

#include "ArapSolver.hpp"
#include "ArapDragHandler.hpp"

#include "core/editor/VertexDragHandler.hpp"
#include "core/editor/VertexManager.hpp"
#include "core/editor/selection/SelectionManager.hpp"
#include "core/editor/selection/VertexRole.hpp"
#include "core/editor/command/CommandManager.hpp"
#include "core/scene/Mesh.hpp"

#include <vulkan/vulkan.h>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace lr
{

// The ARAP feature's entire public surface — the only class a host application (main.cpp today,
// potentially a Python-driven one later) needs to name. Owns the anchor/handle role registration,
// the solver, the glue drag handler, and the popup/Solve-button UI.
class ArapTool
{
public:
    ArapTool(SelectionManager &selectionManager, VertexManager &vertexManager, CommandManager &commandManager,
             const Mesh &mesh);

    void setModeActive(bool active);
    bool isModeActive() const { return m_modeActive; }
    bool hasDeformationTarget() const { return m_modeActive && m_solver.isPrecomputed(); }
    VertexDragHandler &dragHandler() { return m_arapHandler; }

    // Repoints this ArapTool at a different mesh (e.g. the Scene Hierarchy selection changed).
    // Invalidates any precomputed solve (it was factored for the old mesh's topology/vertex
    // domain) and forces ARAP mode off. The caller's SelectionManager/VertexManager are expected
    // to have already been rebound to the same mesh.
    void rebind(const Mesh &mesh);

    // Draws the anchor/handle popup near the current selection. selectionWorldCentroid is only
    // meaningful while a selection exists.
    void onOverlayGui(const glm::mat4 &viewProj, VkExtent2D extent, const glm::vec3 &selectionWorldCentroid);

    // Draws the controls intended to be embedded in the host's Features panel.
    void onPanelGui();

private:
    void onRoleChanged();
    void onSolveClicked();

    SelectionManager &m_selectionManager;
    const Mesh       *m_mesh;

    VertexRoleId m_anchorRole;
    VertexRoleId m_handleRole;

    ArapSolver      m_solver;
    ArapDragHandler m_arapHandler;

    bool m_modeActive      = false;
    bool m_lastSolveFailed = false;
};

} // namespace lr
