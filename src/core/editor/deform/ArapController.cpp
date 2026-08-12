#include "ArapController.hpp"

#include "core/editor/command/ArapDeformCommand.hpp"

#include <spdlog/spdlog.h>

#include <chrono>
#include <unordered_map>
#include <unordered_set>

namespace lr
{

namespace
{

struct Vec3Hash
{
    std::size_t operator()(const glm::vec3 &v) const
    {
        std::size_t seed = std::hash<float>{}(v.x);
        auto combine     = [&](float f) {
            std::size_t h = std::hash<float>{}(f);
            seed ^= h + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        };
        combine(v.y);
        combine(v.z);
        return seed;
    }
};

struct WeldedTopology
{
    std::vector<glm::vec3>  positions;        // canonical (welded) positions
    std::vector<glm::uvec3> faces;             // reindexed to canonical vertices
    std::vector<uint32_t>   vertexToCanonical; // renderIdx -> canonicalIdx, size == input positions.size()
};

// Collapses every render vertex onto a canonical vertex sharing its exact position — safe here
// because duplicate render vertices at a seam are re-emitted from the same source position, so
// they're bit-identical, not just numerically close. See ArapController.hpp for why ARAP needs
// this instead of the raw render topology.
WeldedTopology weldByPosition(const std::vector<glm::vec3> &positions, const std::vector<glm::uvec3> &faces)
{
    WeldedTopology topology;
    topology.vertexToCanonical.resize(positions.size());

    std::unordered_map<glm::vec3, uint32_t, Vec3Hash> canonicalLookup;
    for (size_t i = 0; i < positions.size(); ++i)
    {
        auto [it, inserted] =
            canonicalLookup.try_emplace(positions[i], static_cast<uint32_t>(topology.positions.size()));
        if (inserted)
            topology.positions.push_back(positions[i]);
        topology.vertexToCanonical[i] = it->second;
    }

    topology.faces.reserve(faces.size());
    for (const auto &f : faces)
        topology.faces.emplace_back(topology.vertexToCanonical[f.x], topology.vertexToCanonical[f.y],
                                     topology.vertexToCanonical[f.z]);

    return topology;
}

}  // namespace

ArapController::ArapController(VertexManager &vertexManager, SelectionManager &selectionManager,
                                CommandManager &commandManager, const std::vector<glm::uvec3> &faces)
    : m_vertexManager(vertexManager), m_selectionManager(selectionManager),
      m_commandManager(commandManager), m_faces(faces)
{
}

bool ArapController::confirmConstraints()
{
    const auto &handles = m_selectionManager.getSelectedIndices();
    const auto &anchors = m_selectionManager.getAnchorIndices();

    std::vector<uint32_t> constrained = handles;
    constrained.insert(constrained.end(), anchors.begin(), anchors.end());

    spdlog::info("ArapController::confirmConstraints: {} handle(s), {} anchor(s), {} mesh vertice(s)",
                 handles.size(), anchors.size(), m_vertexManager.getPositions().size());

    m_bindPositions = m_vertexManager.getPositions();

    const WeldedTopology topology = weldByPosition(m_bindPositions, m_faces);
    m_vertexToCanonical           = topology.vertexToCanonical;

    // Translate render-space constrained indices to welded (canonical) indices, deduping while
    // remembering one representative render index per canonical vertex — buildConstraintPositions()
    // uses it to ask "is this constrained vertex an anchor?" the same way it always has.
    std::unordered_map<uint32_t, uint32_t> canonicalToRepresentative;  // canonicalIdx -> renderIdx
    std::vector<uint32_t> canonicalConstrained;
    canonicalConstrained.reserve(constrained.size());
    for (uint32_t renderIdx : constrained)
    {
        if (renderIdx >= m_vertexToCanonical.size())
            continue;
        const uint32_t canonicalIdx = m_vertexToCanonical[renderIdx];
        if (canonicalToRepresentative.try_emplace(canonicalIdx, renderIdx).second)
            canonicalConstrained.push_back(canonicalIdx);
    }

    const bool ok = m_deformer.precompute(topology.positions, topology.faces, canonicalConstrained);

    if (!ok)
    {
        spdlog::warn("ArapController::confirmConstraints: precomputation FAILED "
                     "(need at least one handle or anchor selected; check that the mesh has faces)");
        return false;
    }

    // m_deformer.constrainedIndices() is the sorted/deduped canonical set — rebuild the
    // representative-render-index list in that exact order.
    m_constraintRepresentativeRenderIndex.clear();
    m_constraintRepresentativeRenderIndex.reserve(m_deformer.constrainedIndices().size());
    for (uint32_t canonicalIdx : m_deformer.constrainedIndices())
        m_constraintRepresentativeRenderIndex.push_back(canonicalToRepresentative.at(canonicalIdx));

    spdlog::info("ArapController::confirmConstraints: precomputation OK, {} constrained vertice(s) "
                 "({} after seam-welding)",
                 constrained.size(), m_deformer.constrainedIndices().size());

    return true;
}

void ArapController::beginDrag()
{
    m_dragStartPositions = m_vertexManager.getPositions();
}

std::vector<glm::vec3> ArapController::buildConstraintPositions() const
{
    const auto &anchorSet     = m_selectionManager.getAnchorIndices();
    const auto &livePositions = m_vertexManager.getPositions();

    const std::unordered_set<uint32_t> anchorLookup(anchorSet.begin(), anchorSet.end());

    std::vector<glm::vec3> constraintPositions;
    constraintPositions.reserve(m_constraintRepresentativeRenderIndex.size());
    for (uint32_t renderIdx : m_constraintRepresentativeRenderIndex)
    {
        // Anchors stay pinned at the pose captured when constraints were confirmed; handles are
        // wherever the live (rigid) drag has already moved them via the gizmo's dragCallback.
        constraintPositions.push_back(anchorLookup.count(renderIdx) ? m_bindPositions[renderIdx]
                                                                     : livePositions[renderIdx]);
    }
    return constraintPositions;
}

std::vector<glm::vec3> ArapController::scatterToRenderSpace(const std::vector<glm::vec3> &canonicalPositions) const
{
    std::vector<glm::vec3> result(m_vertexToCanonical.size());
    for (size_t i = 0; i < m_vertexToCanonical.size(); ++i)
        result[i] = canonicalPositions[m_vertexToCanonical[i]];
    return result;
}

void ArapController::liveDragUpdate()
{
    // Deliberately does NOT check m_liveDragEnabled — that's the caller's decision (see
    // gizmoDragUpdateCallback in main.cpp); this just performs the solve when asked to.
    if (m_solvePending || !m_deformer.isPrecomputed())
        return;

    std::vector<glm::vec3> canonicalResult = m_deformer.solve(buildConstraintPositions());
    if (!canonicalResult.empty())
        m_vertexManager.setPositions(scatterToRenderSpace(canonicalResult));
}

void ArapController::commitDrag()
{
    if (m_solvePending)
    {
        spdlog::warn("ArapController::commitDrag: ignored, a solve is already in flight");
        return;
    }
    if (!m_deformer.isPrecomputed())
    {
        spdlog::warn("ArapController::commitDrag: ignored, confirmConstraints() was never called "
                     "successfully — click \"Confirm Constraints\" first");
        return;
    }

    // beginDrag() should always have run first (via the gizmo's onMouseDown), but guard against a
    // missing/mismatched snapshot (e.g. ARAP was enabled mid-drag) — falling back to the live
    // positions still lets the solve itself undo correctly, it just won't also cover the preceding
    // rigid drag in that one undo step.
    const auto &livePositions = m_vertexManager.getPositions();
    if (m_dragStartPositions.size() != livePositions.size())
    {
        spdlog::warn("ArapController::commitDrag: no matching beginDrag() snapshot — undo will only "
                     "cover the ARAP solve, not the handle drag that preceded it");
        m_dragStartPositions = livePositions;
    }

    std::vector<glm::vec3> constraintPositions = buildConstraintPositions();

    m_solvePending = true;

    m_future = std::async(std::launch::async, [this, constraintPositions]() {
        return m_deformer.solve(constraintPositions);
    });
}

void ArapController::update()
{
    if (!m_solvePending)
        return;

    if (m_future.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
        return;

    std::vector<glm::vec3> canonicalResult = m_future.get();
    m_solvePending                         = false;

    if (canonicalResult.empty())
    {
        spdlog::warn("ArapController::update: solve() returned no result — leaving the mesh at its "
                     "live pre-solve (rigid-dragged) positions");
        return;
    }

    std::vector<glm::vec3> result = scatterToRenderSpace(canonicalResult);

    spdlog::info("ArapController::update: solve complete, applying {} position(s)", result.size());
    m_vertexManager.setPositions(result);
    m_commandManager.appendCommandWithoutExecuting(
        std::make_unique<ArapDeformCommand>(m_vertexManager, m_dragStartPositions, result));
}

}  // namespace lr
