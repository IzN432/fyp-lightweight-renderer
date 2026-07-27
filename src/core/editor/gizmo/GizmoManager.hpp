#pragma once

#include "Gizmo.hpp"

#include "core/passes/overlaygeometry/OverlayGeometryPass.hpp"
#include "core/framegraph/ImageReadback.hpp"
#include "core/app/InputHandler.hpp"

#include <memory>
#include <unordered_map>
#include <vulkan/vulkan.h>

namespace lr
{

class GizmoManager
{
public:
    GizmoManager(OverlayGeometryPass &overlayGeometryPass, InputHandler &input)
        : overlayGeometryPass(overlayGeometryPass), m_input(input) {}
    ~GizmoManager() = default;

    // Dispatches the callbacks of each gizmo it manages
    void mouseButtonCallback(int button, int action, bool shift, bool ctrl, bool alt);

    // Updates the state of each gizmo it manages (e.g. hover detection)
    void updateCallback(float dt, VkExtent2D extent, bool hasRenderedAtLeastOneFrame, ImageReadback &gizmoReadback, ResourceRegistry &resources);

    // Takes ownership of the gizmo and assigns it a picking id. Returns that id.
    int addGizmo(std::unique_ptr<Gizmo> gizmo);

    Gizmo &getGizmo(uint32_t gizmoId) { return *m_gizmos.at(gizmoId); }

    void hideGizmo(int gizmoId) { m_gizmoHiddenStates[gizmoId] = true; }
    void unhideGizmo(int gizmoId) { m_gizmoHiddenStates[gizmoId] = false; }

    // True once a gizmo has captured the mouse (i.e. a drag is in progress),
    // including the frame the drag started on. Use to stop other input
    // consumers (e.g. selection) from also reacting to the same click.
    bool isInteracting() const { return m_draggingGizmo != ~0u; }

    std::vector<OverlayInstance> getVisibleGizmoInstances() const;

private:
    std::unordered_map<uint32_t, std::unique_ptr<Gizmo>> m_gizmos;
    std::unordered_map<uint32_t, bool> m_gizmoHiddenStates;
    uint32_t m_nextGizmoId = 1;
    OverlayGeometryPass &overlayGeometryPass;
    uint32_t m_hoveredGizmo = ~0u;
    uint32_t m_draggingGizmo = ~0u;
    InputHandler &m_input;
    bool m_mouseClickedThisFrame = false;
    bool m_mouseReleasedThisFrame = false;
};

} // namespace lr