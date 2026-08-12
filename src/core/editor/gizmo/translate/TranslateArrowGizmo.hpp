#pragma once

#include "core/editor/gizmo/Gizmo.hpp"

#include "core/scene/SceneObject.hpp"
#include "core/app/InputHandler.hpp"
#include "core/editor/VertexManager.hpp"
#include "core/editor/selection/SelectionManager.hpp"
#include "core/editor/command/CommandManager.hpp"

#include <glm/vec3.hpp>

#include <cstdint>
#include <functional>
#include <vector>

namespace lr
{

enum class TranslateArrowGizmoAxis
{
    X = 0,
    Y = 1,
    Z = 2,
};

class TranslateArrowGizmo : public Gizmo
{
public:
    // Invoked once on mouse release with the dragged indices and the net translation applied
    // over the drag. Defaults to appending a TranslatePointsCommand (today's behavior); override
    // via setCommitCallback() to redirect the release into something else (e.g. ArapController).
    using CommitCallback = std::function<void(const std::vector<uint32_t> &, const glm::vec3 &)>;

    // Invoked once on mouse press, before any vertex is moved. Defaults to a no-op; ArapController
    // uses this to snapshot the mesh's pre-drag positions so a single undo can later cover both the
    // live rigid drag and the ARAP solve it triggers on release, as one action.
    using BeginDragCallback = std::function<void()>;

    // Invoked every frame during a drag, after the handle vertices have already been moved
    // rigidly. Defaults to a no-op; ArapController uses this to run a cheap synchronous ARAP solve
    // per frame ("live drag") when that mode is enabled, so the rest of the mesh follows the
    // handle in real time instead of only snapping into shape on release.
    using DragUpdateCallback = std::function<void()>;

    explicit TranslateArrowGizmo(TranslateArrowGizmoAxis axis, const SceneObject &camera, const InputHandler &input,
                                 VertexManager &vertexManager, SelectionManager &selectionManager, CommandManager &commandManager);
    ~TranslateArrowGizmo() = default;

    // Mouse interaction callbacks: override these in derived classes to implement gizmo behavior
    // Mouse positions are in normalized device coordinates (NDC)
    void onMouseDown(double ndcX, double ndcY, double aspect) override;
    void onMouseUp(double ndcX, double ndcY, double aspect) override;
    void dragCallback(double ndcX, double ndcY, double dNdcX, double dNdcY, double aspect) override;

    void setCommitCallback(CommitCallback cb) { m_commitCallback = std::move(cb); }
    void setBeginDragCallback(BeginDragCallback cb) { m_beginDragCallback = std::move(cb); }
    void setDragUpdateCallback(DragUpdateCallback cb) { m_dragUpdateCallback = std::move(cb); }

private:
    const SceneObject  &m_camera;
    const InputHandler &m_input;
    VertexManager      &m_vertexManager;
    SelectionManager   &m_selectionManager;
    CommandManager     &m_commandManager;
    glm::vec3           m_axis;
    CommitCallback       m_commitCallback;
    BeginDragCallback     m_beginDragCallback = [] {};
    DragUpdateCallback    m_dragUpdateCallback = [] {};

    glm::vec3 m_currentDraggingOrigin;
    glm::vec3 m_draggingOrigin;
};

} // namespace lr
