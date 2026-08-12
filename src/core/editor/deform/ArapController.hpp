#pragma once

#include "ArapDeformer.hpp"

#include "core/editor/VertexManager.hpp"
#include "core/editor/command/CommandManager.hpp"
#include "core/editor/selection/SelectionManager.hpp"

#include <cstdint>
#include <future>
#include <vector>

namespace lr
{

// Orchestrates ARAP editing: anchors + handles come from SelectionManager (anchors via its
// Anchor selection mode, handles via its existing "selected" set, which is also what the
// translate gizmos already drag). The expensive solve never runs on the main thread — it's
// launched on mouse release (see commitDrag()) and picked up later by update(), which is polled
// once per frame from the same place gizmo/selection updates already happen.
//
// Concurrency contract: the worker thread touches only ArapDeformer's own Eigen state. It never
// touches VertexManager, Mesh, SelectionManager, or anything Vulkan — all of that is written back
// on the main thread inside update(), once the future is ready. Callers MUST treat isBusy()==true
// as "do not start a new drag, change the anchor/handle selection, or mutate scene structure"
// until update() clears it; this class does not enforce that itself.
class ArapController
{
public:
    // Vertex count above which the "Live Drag" GUI toggle should warn that per-frame solving may
    // stutter. This is a heads-up, not an enforced limit — the user decides whether to enable it
    // regardless of mesh size; see the ARAP panel in main.cpp.
    static constexpr size_t kLiveDragVertexWarningThreshold = 1000;

    ArapController(VertexManager &vertexManager, SelectionManager &selectionManager,
                    CommandManager &commandManager, const std::vector<glm::uvec3> &faces);

    void setEnabled(bool enabled) { m_enabled = enabled; }
    bool isEnabled() const { return m_enabled; }

    // When on, dragCallback additionally triggers a cheap synchronous ARAP solve every frame (see
    // liveDragUpdate()) so the mesh deforms live instead of only snapping into shape on release.
    // Off by default — costs one solve() per frame of dragging, on the main thread.
    void setLiveDragEnabled(bool enabled) { m_liveDragEnabled = enabled; }
    bool isLiveDragEnabled() const { return m_liveDragEnabled; }

    // True from commitDrag() until update() consumes the result. Gate input on this.
    bool isBusy() const { return m_solvePending; }

    bool hasConstraints() const { return m_deformer.isPrecomputed(); }

    // Re-factorizes for the mesh's current pose and the current anchor ∪ handle selection.
    // Synchronous (precomputation is a one-off action, not per-frame) — do not call while isBusy().
    // Returns false if the combined anchor+handle set is empty or libigl's precomputation fails.
    bool confirmConstraints();

    // Call once on mouse press, before the gizmo moves anything. Snapshots the mesh's current
    // (pre-drag) positions so the eventual undo command covers the live rigid handle drag AND the
    // ARAP solve it triggers on release as a single action, rather than just the solve.
    void beginDrag();

    // Call every frame during a drag when isLiveDragEnabled() — runs ARAP's solve() synchronously,
    // on the main thread, and immediately applies the result. Unlike commitDrag() this is not
    // async: the whole point of live-drag is a solve fast enough to fit inside one frame, so
    // there's nothing to gain by deferring it, and doing so would risk overlapping solves as the
    // user keeps dragging. No-op if constraints haven't been confirmed or a release-triggered solve
    // is already in flight (isBusy()).
    void liveDragUpdate();

    // Call once on mouse release while ARAP mode is active and hasConstraints(). Snapshots the
    // live (post-drag) handle positions and the anchors' bind-pose positions, then launches the
    // solve asynchronously. No-op if already busy or constraints haven't been confirmed.
    void commitDrag();

    // Poll once per frame. No-op unless a solve is in flight; applies the result and clears
    // isBusy() once the background solve finishes.
    void update();

private:
    // Shared by commitDrag() and liveDragUpdate(): builds, in constrainedIndices() order, the
    // target position of every constrained vertex — anchors pinned at m_bindPositions, handles at
    // their current (live-dragged) position.
    std::vector<glm::vec3> buildConstraintPositions() const;

    // Expands a solve() result (one entry per welded/canonical vertex, see confirmConstraints())
    // back out to one entry per render vertex, so duplicate vertices at UV/normal seams all pick
    // up the same solved position instead of tearing apart.
    std::vector<glm::vec3> scatterToRenderSpace(const std::vector<glm::vec3> &canonicalPositions) const;

    VertexManager    &m_vertexManager;
    SelectionManager &m_selectionManager;
    CommandManager   &m_commandManager;
    const std::vector<glm::uvec3> &m_faces;

    ArapDeformer m_deformer;
    bool         m_enabled         = false;
    bool         m_liveDragEnabled = false;
    bool         m_solvePending    = false;

    std::future<std::vector<glm::vec3>> m_future;

    std::vector<glm::vec3> m_bindPositions;      // full-mesh snapshot at the last confirmConstraints()
    std::vector<glm::vec3> m_dragStartPositions; // full-mesh snapshot from the last beginDrag() — the undo "before" state

    // Render meshes duplicate a vertex at every UV/normal seam (e.g. each face of a subdivided
    // cube typically has its own UV island and flat normal, so every edge has 2-3 vertex entries
    // sharing one position) — necessary for rendering, but ARAP is fed a *welded* (position-deduped)
    // topology instead, because feeding it the render topology directly would leave those seam
    // copies disconnected from each other in the solve, and the mesh would tear apart at every seam
    // under a drag. m_vertexToCanonical[renderIdx] is the welded vertex index for that render vertex
    // (rebuilt every confirmConstraints()); m_constraintRepresentativeRenderIndex holds, in
    // m_deformer.constrainedIndices() order, one render index per welded constrained vertex — used
    // to look up whether that constraint is an anchor or a handle in render-index space.
    std::vector<uint32_t> m_vertexToCanonical;
    std::vector<uint32_t> m_constraintRepresentativeRenderIndex;
};

}  // namespace lr
