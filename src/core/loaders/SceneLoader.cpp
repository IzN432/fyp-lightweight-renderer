#include "core/loaders/SceneLoader.hpp"

#include "core/scene/MeshComponent.hpp"
#include "core/scene/TransformComponent.hpp"
#include "features/animation/AnimatorComponent.hpp"
#include "features/linear_blend_skinning/SkinComponent.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace lr
{
namespace
{

MeshLoadResult loadFile(const std::filesystem::path &path, MaterialStore &materials,
                        const SceneLoaderConfig &config)
{
    std::string extension = path.extension().string();
    std::ranges::transform(extension, extension.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (extension == ".gltf" || extension == ".glb")
    {
        return GltfLoader::load(path, materials, config.gltf);
    }
    if (extension == ".obj")
    {
        return ObjLoader::load(path, materials, config.obj);
    }
    throw std::invalid_argument("SceneLoader: unsupported file extension '" + extension + "'");
}

} // namespace

SceneLoadResult SceneLoader::load(const std::filesystem::path &path, Scene &scene, MeshStore &meshStore,
                                  MaterialStore &materialStore,
                                  const SceneLoaderConfig &config)
{
    MeshLoadResult loaded = loadFile(path, materialStore, config);
    if (loaded.sequence.empty())
    {
        throw std::runtime_error("SceneLoader: file contains no meshes: '" + path.string() + "'");
    }

    SceneLoadResult result;
    result.materialHandles = loaded.materialHandles;
    SceneObject     &importRoot = scene.createSceneObject();
    importRoot.name = path.stem().string();
    importRoot.addComponent<TransformComponent>();
    result.rootObject = importRoot.id();
    result.nodeObjects.resize(loaded.nodes.size());
    std::vector<MeshHandle> meshHandles;
    meshHandles.reserve(loaded.sequence.frames.size());
    for (Mesh &mesh : loaded.sequence.frames)
    {
        meshHandles.push_back(meshStore.add(std::move(mesh)));
    }

    std::vector<bool> selected(loaded.nodes.size(), false);
    std::vector<uint32_t> pending = loaded.sceneRoots;
    while (!pending.empty())
    {
        const uint32_t nodeIndex = pending.back();
        pending.pop_back();
        if (nodeIndex >= loaded.nodes.size())
        {
            throw std::runtime_error("SceneLoader: scene root or child index is out of range");
        }
        if (selected[nodeIndex])
        {
            continue;
        }
        selected[nodeIndex] = true;
        pending.insert(pending.end(), loaded.nodes[nodeIndex].children.begin(), loaded.nodes[nodeIndex].children.end());
    }

    // A skin may reference joints outside the selected scene roots. Import the
    // complete ancestor chain for every joint used by a selected mesh node so
    // SceneObject::worldMatrix() has all required transforms.
    for (uint32_t nodeIndex = 0; nodeIndex < loaded.nodes.size(); ++nodeIndex)
    {
        const auto skinIndex = loaded.nodes[nodeIndex].skinIndex;
        if (!selected[nodeIndex] || !skinIndex)
        {
            continue;
        }
        if (skinIndex.value() >= loaded.skins.size())
        {
            throw std::runtime_error("SceneLoader: node skin index is out of range");
        }
        for (const SkinJointLoadData &joint : loaded.skins[skinIndex.value()].joints)
        {
            if (joint.nodeIndex >= loaded.nodes.size())
            {
                throw std::runtime_error("SceneLoader: skin joint node index is out of range");
            }
            std::optional<uint32_t> current = joint.nodeIndex;
            while (current && !selected[current.value()])
            {
                selected[current.value()] = true;
                current = loaded.nodes[current.value()].parent;
            }
        }
    }

    for (uint32_t nodeIndex = 0; nodeIndex < loaded.nodes.size(); ++nodeIndex)
    {
        if (!selected[nodeIndex])
        {
            continue;
        }
        const MeshNode &node = loaded.nodes[nodeIndex];
        SceneObject    &object = scene.createSceneObject();
        object.name = node.name.empty() ? "Imported Node " + std::to_string(nodeIndex) : node.name;
        object.addComponent<TransformComponent>(node.localTransform);
        result.nodeObjects[nodeIndex] = object.id();
    }

    for (uint32_t nodeIndex = 0; nodeIndex < loaded.nodes.size(); ++nodeIndex)
    {
        if (!result.nodeObjects[nodeIndex])
        {
            continue;
        }
        const auto parent = loaded.nodes[nodeIndex].parent;
        if (parent && parent.value() < result.nodeObjects.size() && result.nodeObjects[parent.value()])
        {
            scene.setParent(result.nodeObjects[nodeIndex].value(), result.nodeObjects[parent.value()].value());
        }
    }

    for (uint32_t nodeIndex = 0; nodeIndex < loaded.nodes.size(); ++nodeIndex)
    {
        if (!result.nodeObjects[nodeIndex])
        {
            continue;
        }
        const auto parent = loaded.nodes[nodeIndex].parent;
        if (!parent || !result.nodeObjects[parent.value()])
        {
            scene.setParent(result.nodeObjects[nodeIndex].value(), importRoot.id());
        }
    }

    std::vector<AnimationClip> clips;
    clips.reserve(loaded.animations.size());
    for (const AnimationLoadData &loadedAnimation : loaded.animations)
    {
        std::vector<AnimationChannel> channels;
        channels.reserve(loadedAnimation.channels.size());
        for (const AnimationChannelLoadData &loadedChannel : loadedAnimation.channels)
        {
            if (loadedChannel.nodeIndex >= result.nodeObjects.size() ||
                !result.nodeObjects[loadedChannel.nodeIndex])
            {
                continue;
            }

            AnimationInterpolation interpolation = AnimationInterpolation::Linear;
            if (loadedChannel.interpolation == AnimationInterpolationLoad::Step)
            {
                interpolation = AnimationInterpolation::Step;
            }
            else if (loadedChannel.interpolation == AnimationInterpolationLoad::CubicSpline)
            {
                interpolation = AnimationInterpolation::CubicSpline;
            }

            const SceneObjectId target = result.nodeObjects[loadedChannel.nodeIndex].value();
            if (loadedChannel.path == AnimationPathLoad::Rotation)
            {
                RotationTrack track(target, interpolation);
                for (const AnimationKeyframeLoadData &keyframe : loadedChannel.keyframes)
                {
                    const auto quaternion = [](const glm::vec4 &xyzw) {
                        return glm::quat(xyzw.w, xyzw.x, xyzw.y, xyzw.z);
                    };
                    track.setKeyframe({.seconds = keyframe.seconds,
                                       .value = quaternion(keyframe.value),
                                       .incomingTangent = quaternion(keyframe.incomingTangent),
                                       .outgoingTangent = quaternion(keyframe.outgoingTangent)});
                }
                channels.emplace_back(std::move(track));
            }
            else if (loadedChannel.path == AnimationPathLoad::Translation)
            {
                TranslationTrack track(target, interpolation);
                for (const AnimationKeyframeLoadData &keyframe : loadedChannel.keyframes)
                {
                    track.setKeyframe({.seconds = keyframe.seconds,
                                       .value = glm::vec3(keyframe.value),
                                       .incomingTangent = glm::vec3(keyframe.incomingTangent),
                                       .outgoingTangent = glm::vec3(keyframe.outgoingTangent)});
                }
                channels.emplace_back(std::move(track));
            }
            else
            {
                ScaleTrack track(target, interpolation);
                for (const AnimationKeyframeLoadData &keyframe : loadedChannel.keyframes)
                {
                    track.setKeyframe({.seconds = keyframe.seconds,
                                       .value = glm::vec3(keyframe.value),
                                       .incomingTangent = glm::vec3(keyframe.incomingTangent),
                                       .outgoingTangent = glm::vec3(keyframe.outgoingTangent)});
                }
                channels.emplace_back(std::move(track));
            }
        }
        if (!channels.empty())
        {
            clips.emplace_back(loadedAnimation.name, std::move(channels));
        }
    }
    if (!clips.empty())
    {
        importRoot.addComponent<AnimatorComponent>(std::move(clips));
    }

    for (uint32_t nodeIndex = 0; nodeIndex < loaded.nodes.size(); ++nodeIndex)
    {
        if (!result.nodeObjects[nodeIndex] || !loaded.nodes[nodeIndex].meshIndex)
        {
            continue;
        }
        const uint32_t meshIndex = loaded.nodes[nodeIndex].meshIndex.value();
        if (meshIndex >= meshHandles.size())
        {
            throw std::runtime_error("SceneLoader: node mesh index is out of range");
        }
        SceneObject &object = scene.getSceneObject(result.nodeObjects[nodeIndex].value());
        object.addComponent<MeshComponent>(meshHandles[meshIndex], meshStore, materialStore);
        if (loaded.nodes[nodeIndex].skinIndex)
        {
            const SkinLoadData &loadedSkin = loaded.skins[loaded.nodes[nodeIndex].skinIndex.value()];
            std::vector<Joint>  joints;
            joints.reserve(loadedSkin.joints.size());
            for (const SkinJointLoadData &joint : loadedSkin.joints)
            {
                if (!result.nodeObjects[joint.nodeIndex])
                {
                    throw std::runtime_error("SceneLoader: skin joint was not imported into the scene");
                }
                joints.push_back({.sceneObject = result.nodeObjects[joint.nodeIndex].value(),
                                  .inverseBindMatrix = joint.inverseBindMatrix});
            }
            auto &skinComponent = object.addComponent<SkinComponent>(Skin(scene, std::move(joints)));
            skinComponent.evaluate();
        }
        if (!result.firstMeshObject)
        {
            result.firstMeshObject = object.id();
        }
    }
    return result;
}

} // namespace lr
