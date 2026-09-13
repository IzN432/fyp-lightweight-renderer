#pragma once

#include "MaterialStore.hpp"
#include "MeshSequence.hpp"
#include "core/scene/Transform.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace lr
{

// A format-independent imported scene node. Node indices are stable within a
// MeshLoadResult and all hierarchy references index the same nodes vector.
struct MeshNode
{
    std::string             name;
    Transform               localTransform;
    std::optional<uint32_t> parent;
    std::vector<uint32_t>   children;
    std::optional<uint32_t> meshIndex;
    std::optional<uint32_t> skinIndex;
};

// Loader-space joint reference. SceneLoader resolves nodeIndex to a stable
// SceneObjectId before constructing the runtime Skin.
struct SkinJointLoadData
{
    uint32_t  nodeIndex = 0;
    glm::mat4 inverseBindMatrix{1.0f};
};

struct SkinLoadData
{
    std::vector<SkinJointLoadData> joints;
};

enum class AnimationPathLoad
{
    Translation,
    Rotation,
    Scale
};

enum class AnimationInterpolationLoad
{
    Linear,
    Step,
    CubicSpline
};

struct AnimationKeyframeLoadData
{
    float     seconds = 0.0f;
    glm::vec4 value{0.0f};
    glm::vec4 incomingTangent{0.0f};
    glm::vec4 outgoingTangent{0.0f};
};

struct AnimationChannelLoadData
{
    uint32_t                          nodeIndex = 0;
    AnimationPathLoad                 path = AnimationPathLoad::Translation;
    AnimationInterpolationLoad        interpolation = AnimationInterpolationLoad::Linear;
    std::vector<AnimationKeyframeLoadData> keyframes;
};

struct AnimationLoadData
{
    std::string                           name;
    std::vector<AnimationChannelLoadData> channels;
};

// Common output of scene-capable mesh loaders. Formats without hierarchy or
// skinning still produce the same shape: a single root node and no skins.
struct MeshLoadResult
{
    MeshSequence                sequence;
    std::vector<MaterialHandle> materialHandles;
    std::vector<SkinLoadData>   skins;
    std::vector<MeshNode>       nodes;
    std::vector<uint32_t>       sceneRoots;
    std::vector<AnimationLoadData> animations;
};

} // namespace lr
