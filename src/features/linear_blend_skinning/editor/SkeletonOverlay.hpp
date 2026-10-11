#pragma once

#include "core/passes/overlaygeometry/OverlayInstance.hpp"
#include "core/scene/SceneObjectId.hpp"
#include "features/linear_blend_skinning/editor/SkeletonSelection.hpp"

#include <glm/vec3.hpp>

#include <vector>
#include <optional>

namespace lr
{

class Scene;

// Appearance of the non-interactive skeleton preview. Sizes are relative to the average non-zero
// bone length so imported rigs remain legible regardless of the units they were authored in.
struct SkeletonOverlayStyle
{
    glm::vec3 boneColor             = {0.25f, 0.65f, 1.0f};
    glm::vec3 jointColor            = {1.0f, 0.65f, 0.15f};
    glm::vec3 selectedColor         = {1.0f, 0.25f, 0.1f};
    float     boneRadiusRatio       = 0.025f;
    float     jointRadiusRatio      = 0.075f;
    float     occludedOpacity       = 0.28f;
};

struct SkeletonOverlay
{
    std::vector<OverlayInstance>    instances;
    std::vector<SkeletonSelection> selectionsByPickingId;

    // Overlay picking IDs are one-based; zero is the cleared background and ~0u means no readback.
    const SkeletonSelection *selectionForPickingId(uint32_t pickingId) const;
};

// Builds the renderer-neutral instances and their stable editor references for one live
// SkinComponent. Bones are arrows from parent to child and every palette joint gets one sphere,
// including roots. References use the shared SkinId, not the representative mesh object's ID.
SkeletonOverlay buildSkeletonOverlay(const Scene &scene, SceneObjectId skinObject,
                                     const std::optional<SkeletonSelection> &selected = std::nullopt,
                                     const SkeletonOverlayStyle &style = {});

} // namespace lr
