#include "core/loaders/SceneLoader.hpp"

#include "core/scene/MeshComponent.hpp"
#include "core/scene/SceneManager.hpp"
#include "core/scene/TransformComponent.hpp"

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

SceneLoadResult SceneLoader::load(const std::filesystem::path &path, SceneManager &sceneManager,
                                  const SceneLoaderConfig &config)
{
    MeshLoadResult loaded = loadFile(path, sceneManager.materialStore(), config);
    if (loaded.sequence.empty())
    {
        throw std::runtime_error("SceneLoader: file contains no meshes: '" + path.string() + "'");
    }

    SceneLoadResult result;
    Scene           &scene = sceneManager.scene();
    SceneObject     &importRoot = scene.createSceneObject();
    importRoot.name = path.stem().string();
    importRoot.addComponent<TransformComponent>();
    result.rootObject = importRoot.id();
    result.nodeObjects.resize(loaded.nodes.size());
    result.meshHandles.reserve(loaded.sequence.frames.size());
    for (Mesh &mesh : loaded.sequence.frames)
    {
        result.meshHandles.push_back(sceneManager.meshStore().add(std::move(mesh)));
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

    for (uint32_t sourceRoot : loaded.sceneRoots)
    {
        if (sourceRoot < result.nodeObjects.size() && result.nodeObjects[sourceRoot])
        {
            scene.setParent(result.nodeObjects[sourceRoot].value(), importRoot.id());
        }
    }

    for (uint32_t nodeIndex = 0; nodeIndex < loaded.nodes.size(); ++nodeIndex)
    {
        if (!result.nodeObjects[nodeIndex] || !loaded.nodes[nodeIndex].meshIndex)
        {
            continue;
        }
        const uint32_t meshIndex = loaded.nodes[nodeIndex].meshIndex.value();
        if (meshIndex >= result.meshHandles.size())
        {
            throw std::runtime_error("SceneLoader: node mesh index is out of range");
        }
        SceneObject &object = scene.getSceneObject(result.nodeObjects[nodeIndex].value());
        object.addComponent<MeshComponent>(result.meshHandles[meshIndex], sceneManager.meshStore(),
                                           loaded.materialHandles, sceneManager.materialStore());
        if (!result.firstMeshObject)
        {
            result.firstMeshObject = object.id();
        }
    }

    result.skins = std::move(loaded.skins);
    return result;
}

} // namespace lr
