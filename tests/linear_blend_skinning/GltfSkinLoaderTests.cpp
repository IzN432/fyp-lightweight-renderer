#include "core/loaders/GltfLoader.hpp"

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

    lr::Skin &skin = result.skins.front();
    assert(skin.nodes().size() == 3);
    assert(skin.joints().size() == 2);
    assert(skin.joints()[0].node == 1);
    assert(skin.joints()[1].node == 2);
    assert(skin.node(1).parent.value() == 0);

    assert(skin.joints()[0].inverseBindMatrix == glm::mat4(1.0f));
    skin.evaluate(glm::mat4(1.0f));

    assert(glm::all(glm::epsilonEqual(glm::vec3(skin.nodeWorldMatrices()[1][3]),
                                     glm::vec3(1.0f, 2.0f, 0.0f), 0.0001f)));
    assert(glm::all(glm::epsilonEqual(glm::vec3(skin.nodeWorldMatrices()[2][3]),
                                     glm::vec3(0.0f, 0.0f, 3.0f), 0.0001f)));

    lr::MaterialStore birdMaterials(32, [] { return lr::Material{}; });
    auto bird = lr::GltfLoader::load(
        std::filesystem::path(LR_SAMPLE_ASSET_DIR) / "bird_orange.glb", birdMaterials);

    assert(bird.sequence.frames.size() == 1);
    assert(bird.skins.size() == 1);
    assert(!bird.skins.front().joints().empty());
    assert(bird.nodes.size() == 54);
    assert(bird.sceneRoots.size() == 1);
    assert(bird.nodes[bird.sceneRoots.front()].name == "Sketchfab_model");
    assert(bird.nodes[51].meshIndex.value() == 0);
    assert(bird.nodes[51].skinIndex.value() == 0);
    assert(bird.nodes[51].parent.value() == 6);
}
