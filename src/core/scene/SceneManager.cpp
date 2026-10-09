#include "SceneManager.hpp"

#include "MeshComponent.hpp"
#include "Camera.hpp"
#include "TransformComponent.hpp"
#include "SceneSerializer.hpp"
#include "core/editor/camera/SphericalCameraController.hpp"

#include "core/app/Viewer.hpp"

#include <algorithm>
#include <stdexcept>

namespace lr
{

namespace
{
// Matches SelectionManager's private kDefaultColor — seeded onto a mesh's "color" per-unique-vertex
// attribute the first time it becomes the edited mesh object, if it doesn't already have one (e.g.
// procedural geometry authored without editing in mind). SelectionManager overwrites it from here.
const glm::vec3 kDefaultVertexColor{1.0f, 0.0f, 1.0f};
} // namespace

SceneManager::SceneManager(ResourceRegistry &registry, SceneAssets &assets)
    : m_registry(registry), m_meshUploader(registry), m_assets(assets),
      m_animationSystem(assets.scene, assets.animations),
      m_gpu(std::make_unique<SceneGpu>(registry, assets.scene, assets.meshes, assets.materials))
{
    // Establishing empty topology is what makes the placeholder a usable Mesh: per-unique-vertex
    // attributes can only be set once a topology exists (see Mesh::setPerUniqueVertexArray), and
    // ensureSelectedMeshAttributes seeds the overlay attributes onto it like any other mesh.
    m_unboundMesh.setTopology({}, {}, {});
}

Mesh &SceneManager::editedMesh()
{
    return m_editedMeshObject ? m_editedMeshObject->getComponent<MeshComponent>().mesh() : m_unboundMesh;
}

const TransformComponent &SceneManager::editedMeshTransform() const
{
    return m_editedMeshObject ? m_editedMeshObject->getComponent<TransformComponent>() : m_unboundMeshTransform;
}

GpuMeshLayout SceneManager::selectedMeshPointsLayout()
{
    MeshLayout layout;
    layout.addPerUniqueVertexAttr<glm::vec3>("color");

    GpuMeshLayout gpuLayout(layout);
    gpuLayout.mapPosition(0, 0, VK_FORMAT_R32G32B32_SFLOAT)
        .mapUniqueVertex("color", 0, 1, VK_FORMAT_R32G32B32_SFLOAT);
    return gpuLayout;
}

GpuMeshLayout SceneManager::selectedMeshHeatmapLayout()
{
    MeshLayout layout;
    layout.addPerUniqueVertexAttr<glm::vec3>("heatmapColors");

    GpuMeshLayout gpuLayout(layout);
    gpuLayout.mapPosition(0, 0, VK_FORMAT_R32G32B32_SFLOAT)
        .mapUniqueVertex("heatmapColors", 0, 1, VK_FORMAT_R32G32B32_SFLOAT);
    return gpuLayout;
}

SceneGpu &SceneManager::gpu()
{
    return *m_gpu;
}

const SceneGpu &SceneManager::gpu() const { return const_cast<SceneManager *>(this)->gpu(); }

SceneObject &SceneManager::load(const std::filesystem::path &path, const SceneLoaderConfig &config)
{
    Scene          &target   = scene();
    SceneLoadResult imported = SceneLoader::load(path, target, meshStore(), materialStore(), animations(), config);
    if (!imported.firstMeshObject)
    {
        throw std::runtime_error("SceneManager::load: imported scene does not instantiate a mesh");
    }

    gpu().addLoaded(imported);
    return target.getSceneObject(imported.rootObject);
}

void SceneManager::save(const std::filesystem::path &path) const
{
    SceneSerializer::save(m_assets, path);
}

std::vector<SceneObjectId> SceneManager::loadScene(const std::filesystem::path &path)
{
    Scene &target = scene();
    std::vector<SceneObjectId> previousObjects;
    for (const auto &object : target.sceneObjects())
    {
        if (object && target.contains(object->id())) previousObjects.push_back(object->id());
    }
    // Clear first by design: a failed load leaves an empty authored scene. Protected application-owned
    // objects (notably the live editor camera) remain because the Viewer holds references to them.
    for (SceneObjectId id : previousObjects)
    {
        if (target.contains(id) && target.canDestroySceneObject(id)) target.destroySceneObject(id);
    }
    // Destruction normally keeps IDs reserved so stale editor references cannot silently bind to a
    // new object. A full scene replacement is the one boundary where those retired objects and
    // their IDs must be released before the saved objects are recreated.
    target.purgeDestroyedSceneObjects();
    gpu().clearSceneResources();
    // Before the stores go, so the SelectionManager stops referencing a Mesh that is about to be
    // destroyed. A loaded scene starts with nothing being vertex-edited; the Scene Hierarchy
    // selection is what picks a target.
    clearEditedMeshObject();
    meshStore().clear();
    materialStore().clear();

    m_animationSystem.stopAll();
    const std::vector<SceneObjectId> loadedObjects =
        SceneSerializer::load(path, target, meshStore(), materialStore(), animations(), true);

    std::vector<SceneObjectId> added;
    std::vector<MaterialHandle> loadedMaterials;
    std::vector<SceneObjectId> legacyEditorCameras;
    for (SceneObjectId loadedId : loadedObjects)
    {
        if (!target.contains(loadedId)) continue;
        SceneObject &object = target.getSceneObject(loadedId);
        // Preserve the live camera object's identity (Viewer/input systems reference it), but restore
        // all authored camera state from the serialized copy before retiring that temporary object.
        if (object.hasComponent<Camera>())
        {
            SceneObject *liveCamera = gpu().camera();
            if (liveCamera)
            {
                const Camera &loadedCamera = object.getComponent<Camera>();
                Camera &camera = liveCamera->getComponent<Camera>();
                camera.projectionType = loadedCamera.projectionType;
                camera.fovYDegrees    = loadedCamera.fovYDegrees;
                camera.nearPlane      = loadedCamera.nearPlane;
                camera.farPlane       = loadedCamera.farPlane;
                camera.orthoHeight    = loadedCamera.orthoHeight;

                if (object.hasComponent<TransformComponent>() &&
                    liveCamera->hasComponent<TransformComponent>())
                {
                    const Transform &loadedTransform = object.getComponent<TransformComponent>().transform();
                    auto &cameraTransform = liveCamera->getComponent<TransformComponent>();
                    cameraTransform.setPosition(loadedTransform.position());
                    cameraTransform.setRotation(loadedTransform.rotation());
                    cameraTransform.setScale(loadedTransform.scale());
                }
                if (object.hasComponent<SphericalCameraController>() &&
                    liveCamera->hasComponent<SphericalCameraController>())
                {
                    liveCamera->getComponent<SphericalCameraController>().setOrbitState(
                        object.getComponent<SphericalCameraController>().orbitState());
                }
            }
            legacyEditorCameras.push_back(object.id());
            continue;
        }
        added.push_back(object.id());
        if (object.hasComponent<MeshComponent>())
        {
            const auto &materials = object.getComponent<MeshComponent>().materialHandles();
            loadedMaterials.insert(loadedMaterials.end(), materials.begin(), materials.end());
            addMeshObject(object);
        }
    }
    for (SceneObjectId id : legacyEditorCameras)
    {
        if (!target.contains(id)) continue;
        const std::vector<SceneObjectId> children = target.getSceneObject(id).children();
        for (SceneObjectId child : children) target.setParent(child, std::nullopt);
        if (target.canDestroySceneObject(id)) target.destroySceneObject(id);
    }
    gpu().updateCamera();
    gpu().queueMaterialTextures(loadedMaterials);
    return added;
}

void SceneManager::addMeshObject(SceneObject &object)
{
    gpu().addMeshObject(object);
}

bool SceneManager::removeSceneObjects(std::span<const SceneObjectId> ids)
{
    gpu().removeSceneObjects(ids);

    if (!m_editedMeshObject || std::ranges::find(ids, m_editedMeshObject->id()) == ids.end())
    {
        return false;
    }
    clearEditedMeshObject();
    return true;
}

void SceneManager::clearEditedMeshObject()
{
    if (!m_editedMeshObject)
    {
        return;
    }
    // Nulled first: the rebind below fires the selection's colors-changed callback, which must find
    // no edited mesh rather than write a placeholder-sized color array onto the outgoing one.
    m_editedMeshObject = nullptr;

    ensureSelectedMeshAttributes(m_unboundMesh);
    if (m_selectionManager)
    {
        m_selectionManager->rebind(m_unboundMesh.positions(), m_unboundMeshTransform);
    }
    // The stamp belongs to whichever mesh is bound, so it moves with it (see onGeometryRebuilt).
    m_selectedTopologyRevision = m_unboundMesh.topologyRevision();
    // The overlay buffers keep the outgoing mesh's contents. Every presentation that draws them
    // requires an edited mesh, and synchronizeSelectedMeshBuffers() skips them while there is none,
    // so nothing samples them until setEditedMeshObject() replaces them for a real target.
}

void SceneManager::initialize(const AreaLightVisualConfig    &areaLightVisualConfig,
                              const GpuMaterialLayout        &materialLayout,
                              const std::vector<std::string> &vertexAttributeNames, InputHandler &input)
{
    gpu().initialize(areaLightVisualConfig, materialLayout, vertexAttributeNames);
    uploadSelectedMeshBuffers();
    // Imports and light changes re-pack the shared geometry; the overlays follow (the GPU-side
    // SceneGpu is owned by this SceneManager, so the listener can't outlive it).
    m_connections.push_back(gpu().onGeometryRebuilt([this](const SceneGpu &) {
        onGeometryRebuilt();
    }));

    // Constructed here rather than as a SceneManager member-initializer since it operates on the
    // selected mesh's Mesh/TransformComponent, which only exist once the meshes are uploaded. The
    // highlight-changed callback updates the CPU color attribute; late synchronization updates the
    // GPU. The application layer still owns wiring up a SelectionTool and its own UI on top of it.
    // This binds to the placeholder (see editedMesh()) and rebinds when the application picks an
    // edited mesh via setEditedMeshObject(); loading geometry alone does not pick one.
    m_selectionManager =
        std::make_unique<SelectionManager>(editedMesh().positions(), editedMeshTransform(), input);
    m_connections.push_back(m_selectionManager->registerColorsChangedCallback([this]() {
        updateSelectedMeshHighlightColors();
    }));
}

void SceneManager::registerCallbacks(Viewer &viewer)
{
    gpu().registerCallbacks(viewer);

    m_connections.push_back(viewer.onUpdate([this](float dt, VkExtent2D extent) {
        m_selectionManager->updateCallback(dt, extent);
    }));

    // After SceneGpu's own late update (skins, flushDirty), which this registers after.
    m_connections.push_back(viewer.onLateUpdate([this](float, VkExtent2D) {
        synchronizeSelectedMeshBuffers();
    }));
}

void SceneManager::flushDirty()
{
    gpu().flushDirty();
    synchronizeSelectedMeshBuffers();
}

void SceneManager::uploadMeshes(const GpuMaterialLayout        &materialLayout,
                                const std::vector<std::string> &vertexAttributeNames)
{
    gpu().uploadMeshes(materialLayout, vertexAttributeNames);
    uploadSelectedMeshBuffers();
}

void SceneManager::uploadSelectedMeshBuffers()
{
    // Deduped position + color, interleaved — unlike GeometryPass's position buffer (duped per
    // UV-seam corner), this is mesh.positions() verbatim, matching the index space VertexManager/
    // SelectionManager and the points-picking overlay already operate in. Color comes from the
    // selected mesh's own per-unique-vertex attribute.
    Mesh &selectedMesh = editedMesh();
    ensureSelectedMeshAttributes(selectedMesh);
    m_selectedMeshPoints = m_meshUploader.uploadUniqueVertexBuffer({&selectedMesh}, m_selectedMeshPointsUploadConfig);

    m_selectedMeshHeatmap      = m_meshUploader.uploadVertexBuffer({&selectedMesh}, m_selectedMeshHeatmapUploadConfig);
    m_selectedTopologyRevision = selectedMesh.topologyRevision();
}

void SceneManager::synchronizeSelectedMeshBuffers()
{
    if (!m_editedMeshObject || !m_selectionManager)
    {
        return;
    }
    const Mesh &mesh = m_editedMeshObject->getComponent<MeshComponent>().mesh();
    // Inactive overlays catch up on their next use from their own retained stamps.
    if (m_editorPresentation.current().vertexPointsVisible)
    {
        m_meshUploader.synchronizeUniqueVertexBuffer({&mesh}, m_selectedMeshPointsUploadConfig);
    }
    if (m_editorPresentation.current().heatmapVisible)
    {
        m_meshUploader.synchronizeVertexBuffer({&mesh}, m_selectedMeshHeatmapUploadConfig);
    }
}

void SceneManager::onGeometryRebuilt()
{
    if (!m_editedMeshObject)
    {
        return;
    }
    Mesh &mesh = m_editedMeshObject->getComponent<MeshComponent>().mesh();
    ensureSelectedMeshAttributes(mesh);
    m_selectedMeshPoints  = m_meshUploader.replaceUniqueVertexBuffer({&mesh}, m_selectedMeshPointsUploadConfig);
    m_selectedMeshHeatmap = m_meshUploader.replaceVertexBuffer({&mesh}, m_selectedMeshHeatmapUploadConfig);
    // Importing unrelated assets must not clear the current vertex selection.
    if (m_selectionManager && m_selectedTopologyRevision != mesh.topologyRevision())
    {
        m_selectionManager->rebind(mesh.positions(), m_editedMeshObject->getComponent<TransformComponent>());
    }
    m_selectedTopologyRevision = mesh.topologyRevision();
}

bool SceneManager::isEditable(const SceneObject &object)
{
    return object.hasComponent<MeshComponent>() && object.hasComponent<TransformComponent>();
}

void SceneManager::setEditedMeshObject(SceneObject &object)
{
    if (!isEditable(object))
    {
        throw std::runtime_error(
            "SceneManager::setEditedMeshObject: object must have a MeshComponent and TransformComponent");
    }
    if (m_editedMeshObject == &object)
    {
        return;
    }
    m_editedMeshObject = &object;

    Mesh &mesh = object.getComponent<MeshComponent>().mesh();

    ensureSelectedMeshAttributes(mesh);
    m_selectedMeshPoints = m_meshUploader.replaceUniqueVertexBuffer({&mesh}, m_selectedMeshPointsUploadConfig);
    m_selectedMeshHeatmap = m_meshUploader.replaceVertexBuffer({&mesh}, m_selectedMeshHeatmapUploadConfig);
    m_selectionManager->rebind(mesh.positions(), object.getComponent<TransformComponent>());
    m_selectedTopologyRevision = mesh.topologyRevision();
}

void SceneManager::ensureSelectedMeshAttributes(Mesh &mesh)
{
    if (!mesh.layout().findPerUniqueVertexAttr("color"))
    {
        mesh.setPerUniqueVertexArray<glm::vec3>(
            "color", std::vector<glm::vec3>(mesh.uniquePositionCount(), kDefaultVertexColor));
    }
    if (!mesh.layout().findPerUniqueVertexAttr("heatmapColors"))
    {
        mesh.setPerUniqueVertexArray<glm::vec3>("heatmapColors",
                                                std::vector<glm::vec3>(mesh.uniquePositionCount(), glm::vec3(0.0f)));
    }
}

void SceneManager::setSelectedMeshHeatmapColors(std::span<const glm::vec3> colors)
{
    editedMesh().setPerUniqueVertexArray("heatmapColors", colors);
}

void SceneManager::updateSelectedMeshHighlightColors()
{
    if (!m_editedMeshObject)
    {
        return;
    }
    // SelectionManager owns the coloring itself (persistent buffer, tool-customizable highlight
    // color) — this just stores its result on the currently edited Mesh (see
    // setEditedMeshObject(); SelectionManager::rebind() keeps its color buffer sized to whichever
    // mesh that currently is).
    auto &editedMesh = m_editedMeshObject->getComponent<MeshComponent>().mesh();
    editedMesh.setPerUniqueVertexArray("color", std::span<const glm::vec3>(m_selectionManager->getColors()));
}

const IndexBufferUploadPerMeshResult &SceneManager::selectedMeshIndexRange() const
{
    if (!m_editedMeshObject)
    {
        // Nothing is being edited: an empty range, so a pass configured from this draws nothing
        // until setEditedMeshObject() gives it a real one.
        static const IndexBufferUploadPerMeshResult kEmptyRange{};
        return kEmptyRange;
    }
    return gpu().indexRange(m_editedMeshObject->getComponent<MeshComponent>().mesh());
}

} // namespace lr
