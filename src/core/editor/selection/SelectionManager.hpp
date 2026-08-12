#pragma once

#include "SelectionTool.hpp"

#include "core/app/InputHandler.hpp"
#include "core/scene/Transform.hpp"

#include <functional>
#include <memory>
#include <vulkan/vulkan.h>

namespace lr
{

// Which vertex set a SelectionTool currently writes into. Handle is the existing "selected
// vertices" set used everywhere (gizmo dragging, highlighting) — Anchor is a second, independent
// set used only by ARAP to mark vertices that must stay fixed during the solve.
enum class SelectionMode
{
    Handle,
    Anchor,
};

class SelectionManager
{
public:
    // meshTransform is the Transform GeometryPass applies to `vertices` at render time — hit-testing
    // needs to work in the same world space the mesh is actually drawn in, not raw local space.
    SelectionManager(const std::vector<glm::vec3> &vertices, const Transform &meshTransform, InputHandler &input)
        : m_vertices(vertices), m_meshTransform(meshTransform), m_input(input) {}
    ~SelectionManager() = default;

    void setSelectTool(std::unique_ptr<SelectionTool> tool);

    // Dispatches the callback of the active selection tool
    void mouseButtonCallback(int button, int action, bool shift, bool ctrl, bool alt);

    // Updates the state of the active selection tool (e.g. drag tracking).
    // Flags set by mouseButtonCallback are consumed on the first update() after they're set.
    void updateCallback(float dt, VkExtent2D extent);

    const std::vector<uint32_t> &getSelectedIndices() const { return m_selectedVertices; }
    std::vector<uint32_t> &getSelectedIndices() { return m_selectedVertices; }

    const std::vector<uint32_t> &getHighlightedIndices() const { return m_highlightedVertices; }
    std::vector<uint32_t> &getHighlightedIndices() { return m_highlightedVertices; }

    // Anchors — a second selection set, independent of getSelectedIndices() ("handles"), used
    // only by ARAP to mark vertices that must stay fixed during the solve. Populated by the same
    // SelectionTool, just routed to this set instead while setSelectionMode(Anchor) is active.
    const std::vector<uint32_t> &getAnchorIndices() const { return m_anchorVertices; }
    std::vector<uint32_t> &getAnchorIndices() { return m_anchorVertices; }
    void clearAnchors();

    void setSelectionMode(SelectionMode mode) { m_mode = mode; }
    SelectionMode getSelectionMode() const { return m_mode; }

    void clearSelection();

    void registerSelectionChangedCallback(std::function<void()> callback) { m_selectionChangedCallback = std::move(callback); }
    void registerHighlightChangedCallback(std::function<void()> callback) { m_highlightChangedCallback = std::move(callback); }
private:
    std::unique_ptr<SelectionTool> m_selectTool;
    std::vector<uint32_t> m_highlightedVertices;
    std::vector<uint32_t> m_selectedVertices;
    std::vector<uint32_t> m_anchorVertices;
    SelectionMode m_mode = SelectionMode::Handle;
    const std::vector<glm::vec3> &m_vertices;
    const Transform &m_meshTransform;
    InputHandler &m_input;
    bool m_mouseClickedThisFrame = false;
    bool m_mouseReleasedThisFrame = false;
    std::function<void()> m_selectionChangedCallback;
    std::function<void()> m_highlightChangedCallback;
};


} // namespace lr