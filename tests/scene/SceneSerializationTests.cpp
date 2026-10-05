#include "core/scene/Camera.hpp"
#include "core/scene/Light.hpp"
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

    TempScene temporary;
    lr::SceneSerializer::save(source, temporary.path);
    auto loaded = lr::SceneSerializer::load(temporary.path);
    assert(loaded->scene.sceneObjects().size() == 6);
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

    // Loading is transactional: malformed input never mutates an existing SceneAssets instance.
    TempScene invalid;
    std::filesystem::create_directories(invalid.path);
    std::ofstream(invalid.path / "scene.json") << R"({"format":"lr.scene","version":99,"objects":[]})";
    bool rejected = false;
    try { (void)lr::SceneSerializer::load(invalid.path); }
    catch (const std::runtime_error &) { rejected = true; }
    assert(rejected);
}
