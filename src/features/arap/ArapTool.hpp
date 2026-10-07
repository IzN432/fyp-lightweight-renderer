#pragma once

#include "ArapSolver.hpp"
#include "ArapDragHandler.hpp"

#include "core/editor/EditorTool.hpp"
#include "core/editor/VertexDragHandler.hpp"
#include "core/editor/VertexManager.hpp"
#include "core/editor/selection/SelectionManager.hpp"
#include "core/editor/selection/VertexRole.hpp"
#include "core/editor/command/CommandManager.hpp"
#include "core/scene/Mesh.hpp"

#include <vulkan/vulkan.h>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include <string_view>
#include <unordered_set>

namespace lr
{

class SceneObject;

// The ARAP feature's entire public surface — the only class a host application (main.cpp today,
// potentially a Python-driven one later) needs to name. Owns the anchor/handle role registration,
// the solver, the glue drag handler, its editor state, and the popup/Solve-button UI.
class ArapTool final : public EditorTool
{
public:
    ArapTool(SelectionManager &selectionManager, VertexManager &vertexManager, CommandManager &commandManager);

    // The editor state this tool registers. Exposed so a host can bind a shortcut to it without
    // hard-coding the id string.
    static std::string_view stateId();

    bool               hasDeformationTarget() const { return m_solver.isPrecomputed(); }
    VertexDragHandler &dragHandler() { return m_arapHandler; }

    // EditorTool. Registering adds the ARAP state — which carries the tool's selection/presentation
    // policy, its role-assignment overlay, and its bid for the translate gizmo — plus the key that
    // toggles it.
    void registerWith(EditorServices &services) override;

    // Invalidates any precomputed solve (it was factored for the old mesh's topology/vertex domain)
    // and retreats out of ARAP mode. The host's SelectionManager/VertexManager are expected to have
    // already been rebound to the same mesh.
    void onTargetChanged(const EditableMeshContext &target) override;
    void onTargetCleared() override;

    void        drawPanel() override;
    const char *displayName() const override { return "ARAP"; }

private:
    void onRoleChanged();
    void onSolveClicked();

    // Drops back to the editor's default state if ARAP mode is the active one. Used whenever the
    // precomputed solve stops being valid for what is on screen.
    void retreatFromArapMode();

    // Draws the anchor/handle popup near the current selection.
    void onOverlayGui(const glm::mat4 &viewProj, VkExtent2D extent);

    glm::vec3 worldCentroidOf(const std::unordered_set<uint32_t> &indices) const;

    SelectionManager &m_selectionManager;
    VertexManager    &m_vertexManager;

    // Set by onTargetChanged; null until the host binds an editable mesh.
    const Mesh  *m_mesh   = nullptr;
    SceneObject *m_object = nullptr;

    EditorStateController *m_states = nullptr;

    VertexRoleId m_anchorRole;
    VertexRoleId m_handleRole;

    ArapSolver      m_solver;
    ArapDragHandler m_arapHandler;

    bool m_lastSolveFailed = false;
    // Declared last so role-change notifications detach before this tool's state is destroyed.
    CallbackConnection m_roleChangedConnection;
};

} // namespace lr
