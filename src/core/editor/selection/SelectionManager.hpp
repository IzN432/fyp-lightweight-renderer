#pragma once

#include "SelectionTool.hpp"
#include "VertexRole.hpp"

#include "core/app/InputHandler.hpp"
#include "core/scene/TransformComponent.hpp"

#include <functional>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vulkan/vulkan.h>

namespace lr
{

class SelectionManager
{
public:
    // meshTransform is the TransformComponent GeometryPass applies to `vertices` at render time — hit-testing
    // needs to work in the same world space the mesh is actually drawn in, not raw local space.
    SelectionManager(const std::vector<glm::vec3> &vertices, const TransformComponent &meshTransform, InputHandler &input)
        : m_colors(vertices.size(), kDefaultColor), m_roles(vertices.size(), kNoRole), m_vertices(&vertices),
          m_meshTransform(&meshTransform), m_input(input)
    {}
    ~SelectionManager() = default;

    // Repoints this SelectionManager at a different mesh's vertices/transform (e.g. the Scene
    // Hierarchy selection changed) — callers holding a SelectionManager& (drag handlers, ArapTool,
    // BoxSelectionTool) see the new mesh immediately. Resets selection, highlight, and per-vertex
    // role classification to the unclassified state, since indices only ever meant something in the
    // previous mesh's vertex domain; role *definitions* (ids/colors registered via registerRole())
    // are unaffected. Fires the selection/colors/role-changed callbacks so dependents (GPU color
    // upload, a tool's role-changed handler) resync.
    void rebind(const std::vector<glm::vec3> &vertices, const TransformComponent &meshTransform);

    void setSelectTool(std::unique_ptr<SelectionTool> tool);

    // Dispatches the callback of the active selection tool
    void mouseButtonCallback(int button, int action, bool shift, bool ctrl, bool alt);

    // Updates the state of the active selection tool (e.g. drag tracking).
    // Flags set by mouseButtonCallback are consumed on the first update() after they're set.
    void updateCallback(float dt, VkExtent2D extent);

    const std::unordered_set<uint32_t> &getSelectedIndices() const { return m_selectedVertices; }
    std::unordered_set<uint32_t>       &getSelectedIndices() { return m_selectedVertices; }

    const std::unordered_set<uint32_t> &getHighlightedIndices() const { return m_highlightedVertices; }
    std::unordered_set<uint32_t>       &getHighlightedIndices() { return m_highlightedVertices; }

    // One color per vertex (indices match `vertices` passed to the constructor) — kDefaultColor
    // where unclassified/unhighlighted, overridden by a registered role color (see registerRole),
    // overridden again by the active SelectionTool's highlightColor() where highlighted. A
    // persistent buffer reused in place by rebuildColors() rather than rebuilt from scratch each
    // time, since highlight changes fire every frame during a drag-select.
    const std::vector<glm::vec3> &getColors() const { return m_colors; }

    void clearSelection();

    // ---- Generic per-vertex role registry ----
    // SelectionManager never learns what a role "means" — it just hands out an id and remembers
    // the color to paint vertices tagged with it. Callers (e.g. an editor feature) define their own
    // semantics on top of the ids they register.

    // Mints a fresh, never-reused role id and remembers its color.
    VertexRoleId registerRole(const glm::vec3 &color);

    // Assigns `role` to every currently-selected index. Last-classified-wins on re-classification.
    // Pass kNoRole to clear the selection back to unclassified.
    void classifySelectionAs(VertexRoleId role);

    // Resets every vertex back to kNoRole, regardless of selection. No-op (fires no callbacks) if
    // every vertex is already unclassified.
    void clearAllRoles();

    VertexRoleId                     getVertexRole(uint32_t index) const;
    std::vector<uint32_t>            getIndicesWithRole(VertexRoleId role) const;
    const std::vector<VertexRoleId> &getRoles() const { return m_roles; }

    CallbackConnection registerSelectionChangedCallback(std::function<void()> callback)
    {
        return m_selectionChangedCallbacks.connect(std::move(callback));
    }
    // Fired by rebuildColors() any time m_colors changes, regardless of why (highlight drag or
    // role classification) — this is what should drive a GPU color re-upload.
    CallbackConnection registerColorsChangedCallback(std::function<void()> callback)
    {
        return m_colorsChangedCallbacks.connect(std::move(callback));
    }
    // Fired only by classifySelectionAs() — distinct from colors-changed since this is meant for
    // invalidation hooks (e.g. an ARAP tool re-running its precompute), not GPU sync.
    CallbackConnection registerRoleChangedCallback(std::function<void()> callback)
    {
        return m_roleChangedCallbacks.connect(std::move(callback));
    }

private:
    static inline const glm::vec3 kDefaultColor{1.0f, 0.0f, 1.0f};

    // Repaints m_colors from kDefaultColor, then registered role colors, then the current
    // m_highlightedVertices + the active tool's highlightColor() (highest priority).
    void rebuildColors();

    std::unique_ptr<SelectionTool>              m_selectTool;
    std::unordered_set<uint32_t>                m_highlightedVertices;
    std::unordered_set<uint32_t>                m_selectedVertices;
    std::vector<glm::vec3>                      m_colors;
    std::vector<VertexRoleId>                   m_roles;
    std::unordered_map<VertexRoleId, glm::vec3> m_roleColors;
    VertexRoleId                                m_nextRoleId = kNoRole + 1;
    const std::vector<glm::vec3>                *m_vertices;
    const TransformComponent                    *m_meshTransform;
    InputHandler                               &m_input;
    bool                                        m_mouseClickedThisFrame  = false;
    bool                                        m_mouseReleasedThisFrame = false;
    CallbackList<>                              m_selectionChangedCallbacks;
    CallbackList<>                              m_colorsChangedCallbacks;
    CallbackList<>                              m_roleChangedCallbacks;
    // Declared last so callbacks detach before the active tool or manager state is destroyed.
    CallbackConnection                          m_toolSelectionConnection;
    CallbackConnection                          m_toolHighlightConnection;
};

} // namespace lr
