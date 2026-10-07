#include "SceneManager.hpp"

#include "MeshComponent.hpp"
#include "Camera.hpp"
#include "TransformComponent.hpp"
#include "SceneSerializer.hpp"

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

SceneManager::SceneManager(ResourceRegistry &registry, uint32_t materialCapacity,
                           std::function<Material()> defaultMaterialFactory)
    : m_registry(registry), m_meshUploader(registry),
      m_materialStore(materialCapacity, std::move(defaultMaterialFactory))
{}

void SceneManager::setScene(Scene &scene)
{
    if (m_gpu)
    {
        throw std::logic_error("SceneManager::setScene: the scene can only be set once");
    }
    m_gpu = std::make_unique<SceneGpu>(m_registry, scene, m_meshStore, m_materialStore);
}

SceneGpu &SceneManager::gpu()
{
    if (!m_gpu)
    {
        throw std::logic_error("SceneManager: scene must be set first (see setScene)");
    }
    return *m_gpu;
}

const SceneGpu &SceneManager::gpu() const { return const_cast<SceneManager *>(this)->gpu(); }

SceneObject &SceneManager::load(const std::filesystem::path &path, const SceneLoaderConfig &config)
{
    Scene          &target   = scene();
    SceneLoadResult imported = SceneLoader::load(path, target, m_meshStore, m_materialStore, config);
    if (!imported.firstMeshObject)
    {
        throw std::runtime_error("SceneManager::load: imported scene does not instantiate a mesh");
    }

    gpu().addLoaded(imported);
    if (!m_editedMeshObject && !gpu().meshObjects().empty())
    {
        m_editedMeshObject = gpu().meshObjects().front();
    }
    return target.getSceneObject(imported.rootObject);
}

void SceneManager::save(const std::filesystem::path &path) const
{
    SceneSerializer::save(gpu().scene(), m_meshStore, m_materialStore, path);
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
    gpu().clearSceneResources();
    m_editedMeshObject = nullptr;
    m_meshStore.clear();
    m_materialStore.clear();

    const size_t firstObject = target.sceneObjects().size();
    SceneSerializer::load(path, target, m_meshStore, m_materialStore);

    std::vector<SceneObjectId> added;
    std::vector<MaterialHandle> loadedMaterials;
    std::vector<SceneObjectId> legacyEditorCameras;
    for (size_t i = firstObject; i < target.sceneObjects().size(); ++i)
    {
        SceneObject &object = *target.sceneObjects()[i];
        if (!target.contains(object.id())) continue;
        // Preserve the live camera object's identity (Viewer/input systems reference it), but restore
        // all authored camera state from the serialized copy before retiring that temporary object.
        if (object.name == "Main Camera" && object.hasComponent<Camera>())
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
    if (!m_editedMeshObject)
    {
        m_editedMeshObject = &object;
    }
}

SceneObject *SceneManager::removeSceneObjects(std::span<const SceneObjectId> ids)
{
    gpu().removeSceneObjects(ids);

    if (m_editedMeshObject && std::ranges::find(ids, m_editedMeshObject->id()) != ids.end())
    {
        const auto  &meshObjects = gpu().meshObjects();
        SceneObject *replacement = meshObjects.empty() ? nullptr : meshObjects.front();
        m_editedMeshObject       = nullptr;
        if (replacement && m_selectionManager)
        {
            setEditedMeshObject(*replacement);
        } else
        {
            m_editedMeshObject = replacement;
        }
    }
    return m_editedMeshObject;
}

void SceneManager::initialize(const AreaLightVisualConfig    &areaLightVisualConfig,
                              const GpuMaterialLayout        &materialLayout,
                              const std::vector<std::string> &vertexAttributeNames, InputHandler &input)
{
    if (!m_editedMeshObject)
    {
        throw std::runtime_error("SceneManager::initialize: at least one mesh object must be registered");
    }

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
    // GPU. The caller (main.cpp) still owns wiring up a SelectionTool and its own UI on top of it.
    auto &selectedMesh = m_editedMeshObject->getComponent<MeshComponent>().mesh();
    m_selectionManager = std::make_unique<SelectionManager>(
        selectedMesh.positions(), m_editedMeshObject->getComponent<TransformComponent>(), input);
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
    auto &selectedMesh = m_editedMeshObject->getComponent<MeshComponent>().mesh();
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
    if (m_editorPresentation.vertexPointsVisible)
    {
        m_meshUploader.synchronizeUniqueVertexBuffer({&mesh}, m_selectedMeshPointsUploadConfig);
    }
    if (m_editorPresentation.heatmapVisible)
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
    auto &selectedMesh = m_editedMeshObject->getComponent<MeshComponent>().mesh();
    selectedMesh.setPerUniqueVertexArray("heatmapColors", colors);
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
    return gpu().indexRange(m_editedMeshObject->getComponent<MeshComponent>().mesh());
}

void SceneManager::setEditorPresentation(EditorPresentation presentation)
{
    if (presentation == m_editorPresentation)
    {
        return;
    }

    const bool selectionWasActive = m_editorPresentation.vertexSelectionActive;
    m_editorPresentation = presentation;
    if (selectionWasActive && !presentation.vertexSelectionActive)
    {
        m_selectionManager->clearSelection();
    }

    m_editorPresentationChangedCallbacks.invoke(m_editorPresentation);
}

CallbackConnection SceneManager::registerEditorPresentationChangedCallback(
    std::function<void(const EditorPresentation &)> callback)
{
    return m_editorPresentationChangedCallbacks.connect(std::move(callback));
}

} // namespace lr
