#pragma once

#include "core/passes/overlaygeometry/OverlayInstance.hpp"
#include "core/scene/SceneObjectId.hpp"

#include <glm/vec3.hpp>

#include <vector>

namespace lr
{

class Scene;

// Appearance of the non-interactive skeleton preview. Sizes are relative to the average non-zero
// bone length so imported rigs remain legible regardless of the units they were authored in.
struct SkeletonOverlayStyle
{
    glm::vec3 boneColor             = {0.25f, 0.65f, 1.0f};
    glm::vec3 jointColor            = {1.0f, 0.65f, 0.15f};
    float     boneRadiusRatio       = 0.025f;
    float     jointRadiusRatio      = 0.075f;
    float     occludedOpacity       = 0.28f;
};

// Builds the renderer-neutral instances for one live SkinComponent. Bones are arrows from parent
// to child and every palette joint gets one sphere, including roots. Picking IDs deliberately stay
// disabled in this visualization slice; the interaction layer will assign them later.
std::vector<OverlayInstance> buildSkeletonOverlay(const Scene &scene, SceneObjectId skinObject,
                                                  const SkeletonOverlayStyle &style = {});

} // namespace lr
