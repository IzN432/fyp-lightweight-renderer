#include "core/scene/Camera.hpp"
#include "core/scene/Light.hpp"
#include "core/scene/MeshComponent.hpp"
#include "core/scene/SceneAssets.hpp"
#include "core/scene/SceneSerializer.hpp"
#include "core/scene/TransformComponent.hpp"
#include "core/editor/camera/SphericalCameraController.hpp"
#include "features/animation/AnimatorComponent.hpp"
#include "features/linear_blend_skinning/SkinComponent.hpp"
#include "features/rigid_body/ColliderComponent.hpp"
#include "features/rigid_body/RigidBodyComponent.hpp"

#include <glm/gtc/epsilon.hpp>

#include <cassert>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <vector>

namespace
{
bool near(const glm::vec3 &a, const glm::vec3 &b) { return glm::all(glm::epsilonEqual(a, b, 0.0001f)); }

struct TempScene
{
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("lr-scene-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".lrscene");
    std::filesystem::path hdriPath = path.parent_path() / (path.stem().string() + ".hdr");
    ~TempScene()
    {
        std::filesystem::remove_all(path);
        std::filesystem::remove_all(hdriPath);
    }
};
}

int main()
{
    // Minimal generic-scene smoke test: saving and loading into a fresh SceneAssets must work
    // without relying on any editor-owned objects or camera-specific loading behavior.
    {
        TempScene genericTemporary;
        lr::SceneAssets genericSource;
        auto &genericRoot = genericSource.scene.createSceneObject();
        genericRoot.name = "Generic root";
        genericRoot.addComponent<lr::TransformComponent>(glm::vec3(1, 2, 3));

        auto &genericChild = genericSource.scene.createSceneObject();
        genericChild.name = "Generic child";
        genericChild.addComponent<lr::TransformComponent>(glm::vec3(4, 5, 6));
        genericSource.scene.setParent(genericChild.id(), genericRoot.id());

        lr::SceneSerializer::save(genericSource, genericTemporary.path);
        auto genericLoaded = lr::SceneSerializer::load(genericTemporary.path);

        assert(genericLoaded->scene.contains(genericRoot.id()));
        assert(genericLoaded->scene.contains(genericChild.id()));
        assert(genericLoaded->scene.getSceneObject(genericChild.id()).parent() == genericRoot.id());
        assert(near(genericLoaded->scene.getSceneObject(genericRoot.id())
                        .getComponent<lr::TransformComponent>()
                        .transform()
                        .position(),
                    {1, 2, 3}));
    }

    TempScene temporary;
    const std::array<std::byte, 4> hdriBytes{
        std::byte{0x23}, std::byte{0x3f}, std::byte{0x52}, std::byte{0x41}};
    std::ofstream(temporary.hdriPath, std::ios::binary).write(
        reinterpret_cast<const char *>(hdriBytes.data()), static_cast<std::streamsize>(hdriBytes.size()));

    lr::SceneAssets source;
    source.scene.setHdriPath(temporary.hdriPath);
    auto &root = source.scene.createSceneObject();
    root.name = "Camera root";
    root.addComponent<lr::TransformComponent>(glm::vec3(1, 2, 3), glm::quat(0.5f, 0.5f, 0.5f, 0.5f), glm::vec3(2));
    auto &camera = root.addComponent<lr::Camera>();
    camera.projectionType = lr::ProjectionType::Orthographic;
    camera.fovYDegrees = 73; camera.nearPlane = 0.25f; camera.farPlane = 400; camera.orthoHeight = 22;
    auto &cameraController = root.addComponent<lr::SphericalCameraController>();
    cameraController.setOrbitState({{4, 5, 6}, 12.0f, 0.75f, -0.25f});
    const glm::vec3 authoredCameraPosition = root.getComponent<lr::TransformComponent>().transform().position();
    // Push the transform away from the orbit-derived pose, so the saved file holds a transform that
    // disagrees with the saved orbit. Nothing in the controller's own deserialization touches the
    // transform, so the only way the round trip can recover the orbit-derived pose is through
    // Component::onLoaded running after the whole scene exists.
    root.getComponent<lr::TransformComponent>().setPosition(glm::vec3(99.0f, 99.0f, 99.0f));

    const lr::LightVariant lights[] = {
        lr::PointLight{{{1, 0, 0}, 2}},
        lr::SpotLight{{{0, 1, 0}, 3}, 12, 34},
        lr::AreaLight{{{0, 0, 1}, 4}, {5, 6}, false},
        lr::DirectionalLight{{{0.2f, 0.3f, 0.4f}, 5}},
        lr::ImageLight{{{0.7f, 0.8f, 0.9f}, 6}},
    };
    std::vector<lr::SceneObjectId> lightObjectIds;
    for (size_t i = 0; i < std::size(lights); ++i)
    {
        auto &object = source.scene.createSceneObject();
        object.name = "Light " + std::to_string(i);
        object.addComponent<lr::TransformComponent>(glm::vec3(static_cast<float>(i), 0, 0));
        object.addComponent<lr::Light>(lights[i]);
        source.scene.setParent(object.id(), root.id());
        lightObjectIds.push_back(object.id());
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
    std::vector<lr::SceneObjectId> meshObjectIds;
    for (int i = 0; i < 2; ++i)
    {
        auto &object = source.scene.createSceneObject();
        object.name = "Shared mesh " + std::to_string(i);
        object.addComponent<lr::MeshComponent>(meshHandle, source.meshes, std::vector{materialHandle}, source.materials);
        meshObjectIds.push_back(object.id());
    }
    lr::TranslationTrack translation(meshObjectIds[0], lr::AnimationInterpolation::CubicSpline);
    translation.setKeyframe({0.0f, {1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
    translation.setKeyframe({2.0f, {10, 11, 12}, {13, 14, 15}, {16, 17, 18}});
    lr::RotationTrack rotation(meshObjectIds[1], lr::AnimationInterpolation::Step);
    rotation.setKeyframe(0.0f, glm::quat(1, 0, 0, 0));
    auto &animator = root.addComponent<lr::AnimatorComponent>(
        std::vector<lr::AnimationClip>{lr::AnimationClip("Edited", {translation, rotation})});
    animator.setLoop(false);
    animator.setSpeedMultiplier(1.5f);

    auto &skinnedObject = source.scene.getSceneObject(meshObjectIds[0]);
    skinnedObject.addComponent<lr::SkinComponent>(lr::Skin(source.scene, {
        {root.id(), glm::mat4(1.0f)}, {lightObjectIds[0], glm::mat4(2.0f)}}));
    auto &body = skinnedObject.addComponent<lr::RigidBodyComponent>(7.0f, lr::RigidBodyType::Dynamic);
    body.setInertiaDiagonal({2, 3, 4});
    body.setLinearDrag(0.2f);
    body.setAngularDrag(0.4f);
    body.setLinearVelocity({5, 6, 7});
    body.setAngularVelocity({8, 9, 10});
    lr::Collider sphere;
    sphere.shape = lr::SphereCollider{1.25f};
    sphere.localPosition = {1, 2, 3};
    sphere.material = {0.7f, 0.8f};
    lr::Collider plane;
    plane.shape = lr::PlaneCollider{0.5f, {3, 4}};
    lr::Collider box;
    box.shape = lr::BoxCollider{{5, 6, 7}};
    skinnedObject.addComponent<lr::ColliderComponent>(std::vector{sphere, plane, box});

    lr::SceneSerializer::save(source, temporary.path);

    // The editor already owns a protected main camera. Its serialized counterpart is loaded under
    // a temporary identity so SceneManager can copy its state onto the live object and retire it.
    lr::SceneAssets editorTarget;
    auto &liveCamera = editorTarget.scene.createSceneObject(root.id());
    liveCamera.name = "Main Camera";
    liveCamera.addComponent<lr::Camera>();
    liveCamera.addComponent<lr::TransformComponent>();
    liveCamera.addComponent<lr::SphericalCameraController>();
    // Reproduce SceneManager's full replacement: the current authored objects are destroyed, but
    // their IDs remain reserved until the replacement boundary purges the tombstones.
    auto &existingLight = editorTarget.scene.createSceneObject(lightObjectIds.front());
    existingLight.addComponent<lr::TransformComponent>();
    existingLight.addComponent<lr::Light>(lr::PointLight{});
    editorTarget.scene.destroySceneObject(existingLight.id());
    editorTarget.scene.purgeDestroyedSceneObjects();
    const auto editorLoaded = lr::SceneSerializer::load(
        temporary.path, editorTarget.scene, editorTarget.meshes, editorTarget.materials, true);
    assert(editorLoaded.front() != liveCamera.id());
    const auto &temporaryCamera = editorTarget.scene.getSceneObject(editorLoaded.front());
    assert(temporaryCamera.hasComponent<lr::Camera>() && !temporaryCamera.parent());
    assert(temporaryCamera.getComponent<lr::Camera>().orthoHeight == 22);

    auto loaded = lr::SceneSerializer::load(temporary.path);
    assert(loaded->scene.hdriPath() == temporary.hdriPath.filename());
    assert(loaded->scene.hdriData() == std::vector<std::byte>(hdriBytes.begin(), hdriBytes.end()));
    assert(loaded->scene.sceneObjects().size() == 8);
    // Every object comes back under the identity it was saved with, which is what lets the
    // references below (parents, animation targets, skin joints) be checked against the source ids.
    const auto &loadedRoot = loaded->scene.getSceneObject(root.id());
    assert(loadedRoot.name == root.name && loadedRoot.children().size() == 5);
    const auto &loadedTransform = loadedRoot.getComponent<lr::TransformComponent>().transform();
    assert(near(loadedTransform.position(), authoredCameraPosition) && near(loadedTransform.scale(), {2, 2, 2}));
    const auto &loadedCamera = loadedRoot.getComponent<lr::Camera>();
    assert(loadedCamera.projectionType == lr::ProjectionType::Orthographic && loadedCamera.fovYDegrees == 73 &&
           loadedCamera.nearPlane == 0.25f && loadedCamera.farPlane == 400 && loadedCamera.orthoHeight == 22);
    const auto loadedOrbit = loadedRoot.getComponent<lr::SphericalCameraController>().orbitState();
    assert(near(loadedOrbit.target, {4, 5, 6}) && loadedOrbit.radius == 12.0f &&
           loadedOrbit.azimuth == 0.75f && loadedOrbit.elevation == -0.25f);
    for (size_t i = 0; i < std::size(lights); ++i)
    {
        const auto &object = loaded->scene.getSceneObject(lightObjectIds[i]);
        assert(object.parent() == loadedRoot.id());
        assert(object.getComponent<lr::Light>().light.index() == lights[i].index());
    }
    const auto &meshA = loaded->scene.getSceneObject(meshObjectIds[0]).getComponent<lr::MeshComponent>();
    const auto &meshB = loaded->scene.getSceneObject(meshObjectIds[1]).getComponent<lr::MeshComponent>();
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
    const auto &roundTripAnimator = loadedRoot.getComponent<lr::AnimatorComponent>();
    assert(!roundTripAnimator.loop() && roundTripAnimator.speedMultiplier() == 1.5f);
    assert(!roundTripAnimator.isPlaying() && !roundTripAnimator.activeClipIndex());
    assert(roundTripAnimator.clips().size() == 1 && roundTripAnimator.clips()[0].tracks().size() == 2);
    const auto &roundTripTranslation = std::get<lr::TranslationTrack>(roundTripAnimator.clips()[0].tracks()[0]);
    assert(roundTripTranslation.target() == meshObjectIds[0] && roundTripTranslation.interpolation() == lr::AnimationInterpolation::CubicSpline);
    assert(roundTripTranslation.keyframes()[1].outgoingTangent == glm::vec3(16, 17, 18));
    const auto &roundTripSkin = loaded->scene.getSceneObject(meshObjectIds[0]).getComponent<lr::SkinComponent>().skin();
    assert(roundTripSkin.joints().size() == 2 && roundTripSkin.joints()[0].sceneObject == loadedRoot.id());
    assert(roundTripSkin.joints()[1].inverseBindMatrix == glm::mat4(2.0f));
    const auto &roundTripBody = loaded->scene.getSceneObject(meshObjectIds[0]).getComponent<lr::RigidBodyComponent>();
    assert(roundTripBody.mass() == 7.0f && roundTripBody.inertiaDiagonal() == glm::vec3(2, 3, 4));
    assert(roundTripBody.linearVelocity() == glm::vec3(5, 6, 7) && roundTripBody.angularVelocity() == glm::vec3(8, 9, 10));
    assert(roundTripBody.accumulatedForce() == glm::vec3(0) && roundTripBody.accumulatedTorque() == glm::vec3(0));
    const auto &roundTripColliders = loaded->scene.getSceneObject(meshObjectIds[0]).getComponent<lr::ColliderComponent>().colliders();
    assert(roundTripColliders.size() == 3);
    assert(std::get<lr::SphereCollider>(roundTripColliders[0].shape).radius == 1.25f);
    assert(std::get<lr::PlaneCollider>(roundTripColliders[1].shape).halfExtents == glm::vec2(3, 4));
    assert(std::get<lr::BoxCollider>(roundTripColliders[2].shape).halfExtents == glm::vec3(5, 6, 7));

    // Identities are what the file names objects by, so loading the same file into a scene that
    // already holds those objects is a collision rather than a silent duplicate.
    bool collisionRejected = false;
    try { (void)lr::SceneSerializer::load(temporary.path, loaded->scene, loaded->meshes, loaded->materials); }
    catch (const std::exception &) { collisionRejected = true; }
    assert(collisionRejected);

    // Deleting an object must not disturb what the survivors are called. Under file-local indices
    // the hole left behind shifted every id after it; under identities nothing moves.
    source.scene.destroySceneObject(lightObjectIds[4]);
    TempScene afterDelete;
    lr::SceneSerializer::save(source, afterDelete.path);
    auto reloaded = lr::SceneSerializer::load(afterDelete.path);
    assert(reloaded->scene.sceneObjects().size() == 7);
    assert(!reloaded->scene.contains(lightObjectIds[4]));
    assert(reloaded->scene.getSceneObject(root.id()).name == root.name);
    for (size_t i = 0; i < 4; ++i)
    {
        assert(reloaded->scene.contains(lightObjectIds[i]));
        assert(reloaded->scene.getSceneObject(lightObjectIds[i]).parent() == root.id());
    }
    const auto &reloadedSkin = reloaded->scene.getSceneObject(meshObjectIds[0]).getComponent<lr::SkinComponent>().skin();
    assert(reloadedSkin.joints()[1].sceneObject == lightObjectIds[0]);
    const auto &reloadedAnimator = reloaded->scene.getSceneObject(root.id()).getComponent<lr::AnimatorComponent>();
    assert(std::get<lr::TranslationTrack>(reloadedAnimator.clips()[0].tracks()[0]).target() == meshObjectIds[0]);

    // Loading is transactional: malformed input never mutates an existing SceneAssets instance.
    TempScene invalid;
    std::ofstream(invalid.path, std::ios::binary) << R"({"format":"lr.scene","version":99,"objects":[]})";
    bool rejected = false;
    try { (void)lr::SceneSerializer::load(invalid.path); }
    catch (const std::runtime_error &) { rejected = true; }
    assert(rejected);
}
