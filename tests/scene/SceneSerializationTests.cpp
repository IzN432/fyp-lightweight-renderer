#include "core/scene/Camera.hpp"
#include "core/scene/Light.hpp"
#include "core/scene/MeshComponent.hpp"
#include "core/scene/SceneAssets.hpp"
#include "core/scene/SceneSerializer.hpp"
#include "core/scene/TransformComponent.hpp"

#include <glm/gtc/epsilon.hpp>

#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>

namespace
{
bool near(const glm::vec3 &a, const glm::vec3 &b) { return glm::all(glm::epsilonEqual(a, b, 0.0001f)); }

struct TempScene
{
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("lr-scene-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    ~TempScene() { std::filesystem::remove_all(path); }
};
}

int main()
{
    lr::SceneAssets source;
    auto &root = source.scene.createSceneObject();
    root.name = "Camera root";
    root.addComponent<lr::TransformComponent>(glm::vec3(1, 2, 3), glm::quat(0.5f, 0.5f, 0.5f, 0.5f), glm::vec3(2));
    auto &camera = root.addComponent<lr::Camera>();
    camera.projectionType = lr::ProjectionType::Orthographic;
    camera.fovYDegrees = 73; camera.nearPlane = 0.25f; camera.farPlane = 400; camera.orthoHeight = 22;

    const lr::LightVariant lights[] = {
        lr::PointLight{{{1, 0, 0}, 2}},
        lr::SpotLight{{{0, 1, 0}, 3}, 12, 34},
        lr::AreaLight{{{0, 0, 1}, 4}, {5, 6}, false},
        lr::DirectionalLight{{{0.2f, 0.3f, 0.4f}, 5}},
        lr::ImageLight{{{0.7f, 0.8f, 0.9f}, 6}},
    };
    for (size_t i = 0; i < std::size(lights); ++i)
    {
        auto &object = source.scene.createSceneObject();
        object.name = "Light " + std::to_string(i);
        object.addComponent<lr::TransformComponent>(glm::vec3(static_cast<float>(i), 0, 0));
        object.addComponent<lr::Light>(lights[i]);
        source.scene.setParent(object.id(), root.id());
    }

    lr::Material material;
    material.name = "Complete material";
    material.parameters["rgba"] = lr::MaterialParam::ColorRGBA{{0.1f, 0.2f, 0.3f, 0.4f}};
    material.parameters["rgb"] = lr::MaterialParam::ColorRGB{{0.5f, 0.6f, 0.7f}};
    material.parameters["normalized"] = lr::MaterialParam::NormalizedFloat{0.8f};
    material.parameters["ranged"] = lr::MaterialParam::RangedFloat{3.0f, 1.0f, 5.0f};
    material.textures["albedo"] = {"checker", {1, 2, 3, 4, 5, 6, 7, 8}, 2, 1};
    const lr::MaterialHandle materialHandle = source.materials.acquire(std::move(material));

    lr::Mesh mesh;
    mesh.setTopology({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}, {0, 1, 2}, {{0, 1, 2}});
    mesh.setFaceGroups({0});
    mesh.setFaceGroupCount(1);
    mesh.setPerVertexArray<glm::vec2>("uv", std::vector<glm::vec2>{{0, 0}, {1, 0}, {0, 1}});
    mesh.setPerVertexArray<glm::vec3>("normal", std::vector<glm::vec3>(3, {0, 0, 1}));
    mesh.setPerVertexArray<glm::vec4>("tangent", std::vector<glm::vec4>(3, {1, 0, 0, 1}));
    mesh.setPerUniqueVertexArray<glm::vec3>("color", std::vector<glm::vec3>{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
    mesh.setPerFaceArray<uint32_t>("tag", std::vector<uint32_t>{17});
    mesh.setFaceGroupAttributeArray<float>("weight", std::vector<float>{0.75f});
    mesh.enableVertexGroups();
    mesh.setVertexGroupCount(2);
    const lr::VertexGroupEntry groups0[] = {{0, 0.25f}, {1, 0.75f}};
    mesh.setVertexGroups(0, groups0);
    mesh.setVertexGroups(1, std::span<const lr::VertexGroupEntry>{});
    const lr::VertexGroupEntry groups2[] = {{1, 1.0f}};
    mesh.setVertexGroups(2, groups2);
    mesh.setVertexGroupAttributeArray<float>("joint_data", std::vector<float>{10, 20});
    const lr::MeshHandle meshHandle = source.meshes.add(std::move(mesh));
    for (int i = 0; i < 2; ++i)
    {
        auto &object = source.scene.createSceneObject();
        object.name = "Shared mesh " + std::to_string(i);
        object.addComponent<lr::MeshComponent>(meshHandle, source.meshes, std::vector{materialHandle}, source.materials);
    }

    TempScene temporary;
    lr::SceneSerializer::save(source, temporary.path);
    auto loaded = lr::SceneSerializer::load(temporary.path);
    assert(loaded->scene.sceneObjects().size() == 8);
    const auto &loadedRoot = loaded->scene.getSceneObject(0);
    assert(loadedRoot.name == root.name && loadedRoot.children().size() == 5);
    const auto &loadedTransform = loadedRoot.getComponent<lr::TransformComponent>().transform();
    assert(near(loadedTransform.position(), {1, 2, 3}) && near(loadedTransform.scale(), {2, 2, 2}));
    const auto &loadedCamera = loadedRoot.getComponent<lr::Camera>();
    assert(loadedCamera.projectionType == lr::ProjectionType::Orthographic && loadedCamera.fovYDegrees == 73 &&
           loadedCamera.nearPlane == 0.25f && loadedCamera.farPlane == 400 && loadedCamera.orthoHeight == 22);
    for (size_t i = 0; i < std::size(lights); ++i)
    {
        const auto &object = loaded->scene.getSceneObject(static_cast<lr::SceneObjectId>(i + 1));
        assert(object.parent() == loadedRoot.id());
        assert(object.getComponent<lr::Light>().light.index() == lights[i].index());
    }
    const auto &meshA = loaded->scene.getSceneObject(6).getComponent<lr::MeshComponent>();
    const auto &meshB = loaded->scene.getSceneObject(7).getComponent<lr::MeshComponent>();
    assert(meshA.meshHandle() == meshB.meshHandle());
    assert(meshA.materialHandles() == meshB.materialHandles());
    const lr::Mesh &roundTripMesh = meshA.mesh();
    assert(roundTripMesh.positions().size() == 3 && roundTripMesh.faces().size() == 1);
    assert(roundTripMesh.getPerVertexArray<glm::vec2>("uv").size() == 3);
    assert(roundTripMesh.getPerVertexArray<glm::vec3>("normal")[0] == glm::vec3(0, 0, 1));
    assert(roundTripMesh.getPerFaceArray<uint32_t>("tag")[0] == 17);
    assert(roundTripMesh.getFaceGroupAttributeArray<float>("weight")[0] == 0.75f);
    assert(roundTripMesh.getVertexGroups(0).size() == 2 && roundTripMesh.getVertexGroups(2)[0].weight == 1.0f);
    assert(roundTripMesh.getVertexGroupAttributeArray<float>("joint_data")[1] == 20);
    const lr::Material &roundTripMaterial = loaded->materials.get(meshA.materialHandles()[0]);
    assert(roundTripMaterial.name == "Complete material");
    assert(std::get<lr::MaterialParam::RangedFloat>(roundTripMaterial.parameters.at("ranged")).ceiling == 5.0f);
    assert(roundTripMaterial.textures.at("albedo").pixels == std::vector<uint8_t>({1, 2, 3, 4, 5, 6, 7, 8}));

    // Loading is transactional: malformed input never mutates an existing SceneAssets instance.
    TempScene invalid;
    std::filesystem::create_directories(invalid.path);
    std::ofstream(invalid.path / "scene.json") << R"({"format":"lr.scene","version":99,"objects":[]})";
    bool rejected = false;
    try { (void)lr::SceneSerializer::load(invalid.path); }
    catch (const std::runtime_error &) { rejected = true; }
    assert(rejected);
}
