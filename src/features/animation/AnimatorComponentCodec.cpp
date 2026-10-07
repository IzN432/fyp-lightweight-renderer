#include "core/scene/serialization/ComponentCodec.hpp"
#include "core/scene/serialization/JsonUtils.hpp"

#include "features/animation/AnimatorComponent.hpp"
#include "core/scene/Scene.hpp"

#include <memory>
#include <stdexcept>

namespace lr
{
namespace
{
using namespace scene_serialization;

const char *interpolationName(AnimationInterpolation interpolation)
{
    switch (interpolation)
    {
        case AnimationInterpolation::Linear: return "linear";
        case AnimationInterpolation::Step: return "step";
        case AnimationInterpolation::CubicSpline: return "cubic_spline";
    }
    return "linear";
}

AnimationInterpolation readInterpolation(const json &value, const std::string &where)
{
    const std::string name = value.get<std::string>();
    if (name == "linear") return AnimationInterpolation::Linear;
    if (name == "step") return AnimationInterpolation::Step;
    if (name == "cubic_spline") return AnimationInterpolation::CubicSpline;
    throw std::runtime_error("SceneSerializer: invalid animation interpolation '" + name + "' at " + where);
}

class AnimatorComponentCodec final : public ComponentCodec
{
public:
    std::string_view key() const override { return "animator"; }
    std::type_index componentType() const override { return typeid(AnimatorComponent); }

    json encode(const SceneObject &object, const ComponentSaveContext &) const override
    {
        const AnimatorComponent &animator = object.getComponent<AnimatorComponent>();
        json clips = json::array();
        for (const AnimationClip &clip : animator.clips())
        {
            json tracks = json::array();
            for (const AnimationChannel &channel : clip.tracks())
            {
                tracks.push_back(std::visit([](const auto &track) {
                    using Track = std::decay_t<decltype(track)>;
                    json keyframes = json::array();
                    for (const auto &keyframe : track.keyframes())
                    {
                        auto encodeValue = [](const auto &value) -> json {
                            using V = std::decay_t<decltype(value)>;
                            if constexpr (std::is_same_v<V, glm::quat>) return quat(value);
                            else return vec3(value);
                        };
                        keyframes.push_back({{"seconds", keyframe.seconds}, {"value", encodeValue(keyframe.value)},
                            {"incoming_tangent", encodeValue(keyframe.incomingTangent)},
                            {"outgoing_tangent", encodeValue(keyframe.outgoingTangent)}});
                    }
                    const char *property = std::is_same_v<Track, TranslationTrack> ? "translation" :
                                           std::is_same_v<Track, RotationTrack> ? "rotation" : "scale";
                    return json{{"target", toString(track.target())}, {"property", property},
                                {"interpolation", interpolationName(track.interpolation())},
                                {"keyframes", std::move(keyframes)}};
                }, channel));
            }
            clips.push_back({{"name", clip.name()}, {"tracks", std::move(tracks)}});
        }
        return {{"clips", std::move(clips)}, {"loop", animator.loop()},
                {"speed", animator.speedMultiplier()}};
    }

    void decode(const json &value, SceneObject &object, ComponentLoadContext &context,
                const std::string &where) const override
    {
        std::vector<AnimationClip> clips;
        const json &serializedClips = required(value, "clips", where);
        if (!serializedClips.is_array())
            throw std::runtime_error("SceneSerializer: " + where + ".clips must be an array");
        for (size_t clipIndex = 0; clipIndex < serializedClips.size(); ++clipIndex)
        {
            const json &clip = serializedClips[clipIndex];
            const std::string clipWhere = where + ".clips[" + std::to_string(clipIndex) + "]";
            std::vector<AnimationChannel> tracks;
            const json &serializedTracks = required(clip, "tracks", clipWhere);
            for (size_t trackIndex = 0; trackIndex < serializedTracks.size(); ++trackIndex)
            {
                const json &track = serializedTracks[trackIndex];
                const std::string trackWhere = clipWhere + ".tracks[" + std::to_string(trackIndex) + "]";
                const SceneObjectId target = readId(required(track, "target", trackWhere), trackWhere + ".target");
                if (!context.scene.contains(target))
                    throw std::runtime_error("SceneSerializer: missing animation target " + toString(target) + " at " + trackWhere);
                const auto interpolation = readInterpolation(required(track, "interpolation", trackWhere), trackWhere + ".interpolation");
                const std::string property = required(track, "property", trackWhere).get<std::string>();
                const json &keyframes = required(track, "keyframes", trackWhere);
                if (property == "rotation")
                {
                    RotationTrack result(target, interpolation);
                    for (const json &key : keyframes)
                    {
                        auto readQuaternion = [&](const char *name) {
                            const glm::vec4 v = readVector<4, float>(required(key, name, trackWhere), trackWhere + "." + name);
                            return glm::quat(v.w, v.x, v.y, v.z);
                        };
                        result.setKeyframe({required(key, "seconds", trackWhere).get<float>(), readQuaternion("value"),
                                            readQuaternion("incoming_tangent"), readQuaternion("outgoing_tangent")});
                    }
                    tracks.emplace_back(std::move(result));
                }
                else if (property == "translation" || property == "scale")
                {
                    auto fill = [&]<typename Track>(Track result) {
                        for (const json &key : keyframes)
                            result.setKeyframe({required(key, "seconds", trackWhere).get<float>(),
                                readVector<3, float>(required(key, "value", trackWhere), trackWhere + ".value"),
                                readVector<3, float>(required(key, "incoming_tangent", trackWhere), trackWhere + ".incoming_tangent"),
                                readVector<3, float>(required(key, "outgoing_tangent", trackWhere), trackWhere + ".outgoing_tangent")});
                        tracks.emplace_back(std::move(result));
                    };
                    if (property == "translation") fill(TranslationTrack(target, interpolation));
                    else fill(ScaleTrack(target, interpolation));
                }
                else throw std::runtime_error("SceneSerializer: invalid animation property '" + property + "' at " + trackWhere);
            }
            clips.emplace_back(required(clip, "name", clipWhere).get<std::string>(), std::move(tracks));
        }
        auto &animator = object.addComponent<AnimatorComponent>(std::move(clips));
        animator.setLoop(required(value, "loop", where).get<bool>());
        animator.setSpeedMultiplier(required(value, "speed", where).get<float>());
    }
};
} // namespace

void addAnimationComponentCodecs(ComponentCodecRegistry &registry)
{
    registry.add(std::make_unique<AnimatorComponentCodec>());
}

} // namespace lr
