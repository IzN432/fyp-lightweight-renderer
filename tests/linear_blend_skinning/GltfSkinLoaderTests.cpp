#include "core/loaders/GltfLoader.hpp"
#include "core/loaders/SceneLoader.hpp"
#include "core/scene/MeshComponent.hpp"
#include "features/linear_blend_skinning/SkinComponent.hpp"

#include <glm/gtc/epsilon.hpp>

#include <cassert>
#include <filesystem>

int main()
{
    lr::MaterialStore materials(1, [] { return lr::Material{}; });
    auto result = lr::GltfLoader::load(
        std::filesystem::path(LR_TEST_ASSET_DIR) / "skin_hierarchy.gltf", materials);

    assert(result.sequence.empty());
    assert(result.skins.size() == 1);
    assert(result.nodes.size() == 3);
    assert(result.sceneRoots.size() == 2);
    assert(result.nodes[0].children.size() == 1);
    assert(result.nodes[0].children[0] == 1);
    assert(result.nodes[1].parent.value() == 0);

    const lr::SkinLoadData &skin = result.skins.front();
    assert(skin.joints.size() == 2);
    assert(skin.joints[0].nodeIndex == 1);
    assert(skin.joints[1].nodeIndex == 2);
    assert(skin.joints[0].inverseBindMatrix == glm::mat4(1.0f));

    lr::MaterialStore birdMaterials(32, [] { return lr::Material{}; });
    auto bird = lr::GltfLoader::load(
        std::filesystem::path(LR_SAMPLE_ASSET_DIR) / "bird_orange.glb", birdMaterials);

    assert(bird.sequence.frames.size() == 1);
    assert(bird.skins.size() == 1);
    assert(!bird.skins.front().joints.empty());
    assert(bird.nodes.size() == 54);
    assert(bird.sceneRoots.size() == 1);
    assert(bird.nodes[bird.sceneRoots.front()].name == "Sketchfab_model");
    assert(bird.nodes[51].meshIndex.value() == 0);
    assert(bird.nodes[51].skinIndex.value() == 0);
    assert(bird.nodes[51].parent.value() == 6);

    lr::Scene         scene;
    lr::MeshStore     meshStore;
    lr::MaterialStore importedMaterials(32, [] { return lr::Material{}; });
    auto imported = lr::SceneLoader::load(
        std::filesystem::path(LR_SAMPLE_ASSET_DIR) / "bird_orange.glb", scene, meshStore, importedMaterials);

    assert(imported.nodeObjects[0].has_value());
    assert(scene.getSceneObject(imported.nodeObjects[0].value()).parent().value() == imported.rootObject);
    lr::SceneObject &birdMeshObject = scene.getSceneObject(imported.nodeObjects[51].value());
    assert(birdMeshObject.hasComponent<lr::SkinComponent>());

    const lr::Mesh &birdMesh = birdMeshObject.getComponent<lr::MeshComponent>().mesh();
    const auto      influenceOffsets = birdMesh.rawGroupOffsets();
    assert(influenceOffsets.size() == birdMesh.uniquePositionCount() + 1);
    assert(influenceOffsets.front() == 0);
    assert(influenceOffsets.back() == birdMesh.rawGroupEntries().size());
    for (uint32_t position = 0; position < birdMesh.uniquePositionCount(); ++position)
    {
        assert(birdMesh.getVertexGroups(position).size() ==
               influenceOffsets[position + 1] - influenceOffsets[position]);
    }

    const lr::Skin &runtimeSkin = birdMeshObject.getComponent<lr::SkinComponent>().skin();
    assert(runtimeSkin.joints().size() == bird.skins[0].joints.size());
    for (size_t jointIndex = 0; jointIndex < runtimeSkin.joints().size(); ++jointIndex)
    {
        const uint32_t sourceNode = bird.skins[0].joints[jointIndex].nodeIndex;
        assert(runtimeSkin.joints()[jointIndex].sceneObject == imported.nodeObjects[sourceNode].value());
    }
    assert(runtimeSkin.jointMatrices().size() == runtimeSkin.joints().size());
}
