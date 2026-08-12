#pragma once

#include "core/editor/gizmo/Gizmo.hpp"

#include "core/scene/SceneObject.hpp"
#include "core/app/InputHandler.hpp"
#include "core/editor/VertexManager.hpp"
#include "core/editor/selection/SelectionManager.hpp"
#include "core/editor/command/CommandManager.hpp"

#include <cstdint>
#include <functional>
#include <vector>

namespace lr
{

class TranslateBoxGizmo : public Gizmo
{
public:
    // See TranslateArrowGizmo::CommitCallback / BeginDragCallback / DragUpdateCallback — same contract.
    using CommitCallback = std::function<void(const std::vector<uint32_t> &, const glm::vec3 &)>;
    using BeginDragCallback = std::function<void()>;
    using DragUpdateCallback = std::function<void()>;

    explicit TranslateBoxGizmo(const SceneObject &camera, const InputHandler &input, VertexManager &vertexManager,
                               SelectionManager &selectionManager, CommandManager &commandManager);
    ~TranslateBoxGizmo() = default;

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
    CommitCallback       m_commitCallback;
    BeginDragCallback     m_beginDragCallback = [] {};
    DragUpdateCallback    m_dragUpdateCallback = [] {};

    glm::vec4 m_draggingPlane;
    glm::vec3 m_draggingOrigin;
    glm::vec3 m_currentDraggingOrigin;
};

} // namespace lr
