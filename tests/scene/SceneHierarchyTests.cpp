#include "core/loaders/MaterialStore.hpp"
#include "core/scene/MeshComponent.hpp"
#include "core/scene/MeshStore.hpp"
#include "core/scene/Scene.hpp"
#include "core/scene/TransformComponent.hpp"

#include <glm/gtc/epsilon.hpp>

#include <cassert>
#include <stdexcept>

int main()
{
    lr::Scene scene;
    auto     &root = scene.createSceneObject();
    auto     &child = scene.createSceneObject();
    auto     &grandchild = scene.createSceneObject();

    root.addComponent<lr::TransformComponent>().setPosition(glm::vec3(1.0f, 0.0f, 0.0f));
    child.addComponent<lr::TransformComponent>().setPosition(glm::vec3(0.0f, 2.0f, 0.0f));
    grandchild.addComponent<lr::TransformComponent>().setPosition(glm::vec3(0.0f, 0.0f, 3.0f));

    scene.setParent(child.id(), root.id());
    scene.setParent(grandchild.id(), child.id());

    assert(child.parent().value() == root.id());
    assert(root.children().front() == child.id());
    assert(glm::all(glm::epsilonEqual(glm::vec3(grandchild.worldMatrix()[3]),
                                     glm::vec3(1.0f, 2.0f, 3.0f), 0.0001f)));

    bool cycleRejected = false;
    try
    {
        scene.setParent(root.id(), grandchild.id());
    }
    catch (const std::invalid_argument &)
    {
        cycleRejected = true;
    }
    assert(cycleRejected);

    scene.setParent(grandchild.id(), std::nullopt);
    assert(!grandchild.parent());
    assert(child.children().empty());

    lr::Mesh mesh;
    mesh.setTopology({glm::vec3(0.0f)}, {0}, {});
    lr::MeshStore meshStore;
    const lr::MeshHandle meshHandle = meshStore.add(std::move(mesh));
    lr::MaterialStore materialStore(1, [] { return lr::Material{}; });

    auto &firstInstance = scene.createSceneObject();
    auto &secondInstance = scene.createSceneObject();
    auto &firstMesh = firstInstance.addComponent<lr::MeshComponent>(meshHandle, meshStore,
                                                                    std::vector<lr::MaterialHandle>{}, materialStore);
    auto &secondMesh = secondInstance.addComponent<lr::MeshComponent>(meshHandle, meshStore,
                                                                      std::vector<lr::MaterialHandle>{}, materialStore);

    assert(firstMesh.meshHandle() == secondMesh.meshHandle());
    assert(&firstMesh.mesh() == &secondMesh.mesh());
}
