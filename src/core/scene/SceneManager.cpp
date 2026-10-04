#include "SceneManager.hpp"

#include "MeshComponent.hpp"
#include "TransformComponent.hpp"

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

    // Constructed here rather than as a SceneManager member-initializer since it operates on the
    // selected mesh's Mesh/TransformComponent, which only exist once the meshes are uploaded. The
    // highlight-changed callback keeps the GPU color buffer in sync with selection state — the
    // caller (main.cpp) still owns wiring up a SelectionTool and its own UI on top of it.
    auto &selectedMesh = m_editedMeshObject->getComponent<MeshComponent>().mesh();
    m_selectionManager = std::make_unique<SelectionManager>(
        selectedMesh.positions(), m_editedMeshObject->getComponent<TransformComponent>(), input);
    m_selectionManager->registerColorsChangedCallback([this]() {
        updateSelectedMeshHighlightColors();
    });
}

void SceneManager::registerCallbacks(Viewer &viewer)
{
    gpu().registerCallbacks(viewer);

    viewer.onUpdate([this](float dt, VkExtent2D extent) {
        m_selectionManager->updateCallback(dt, extent);
    });
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

    // Seeds the corner-domain heatmap attribute from the per-unique-vertex analysis colors the
    // selected mesh, then uploads the heatmap buffer from it.
    m_selectedMeshHeatmap =
        m_meshUploader.uploadVertexBuffer({&syncSelectedMeshCornerHeatmapColors()}, m_selectedMeshHeatmapUploadConfig);
}

void SceneManager::updateSelectedMeshPositions()
{
    gpu().updatePositions();
    updateSelectedMeshPointsBuffer();
    updateSelectedMeshHeatmapBuffer();
}

void SceneManager::updateSelectedMeshPointsBuffer()
{
    const auto &editedMesh = m_editedMeshObject->getComponent<MeshComponent>().mesh();
    m_meshUploader.updateUniqueVertexBuffer({&editedMesh}, m_selectedMeshPointsUploadConfig);
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
    m_selectedMeshHeatmap =
        m_meshUploader.replaceVertexBuffer({&syncSelectedMeshCornerHeatmapColors()}, m_selectedMeshHeatmapUploadConfig);
    m_selectionManager->rebind(mesh.positions(), object.getComponent<TransformComponent>());
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

Mesh &SceneManager::syncSelectedMeshCornerHeatmapColors()
{
    auto                  &selectedMesh = m_editedMeshObject->getComponent<MeshComponent>().mesh();
    const auto             uniqueColor  = selectedMesh.getPerUniqueVertexArray<glm::vec3>("heatmapColors");
    std::vector<glm::vec3> cornerColor(selectedMesh.vertexCount());
    for (uint32_t v = 0; v < selectedMesh.vertexCount(); ++v)
    {
        cornerColor[v] = uniqueColor[selectedMesh.positionIndices()[v]];
    }
    selectedMesh.setPerVertexArray<glm::vec3>("heatmapColors", std::span<const glm::vec3>(cornerColor));
    return selectedMesh;
}

void SceneManager::updateSelectedMeshHeatmapBuffer()
{
    m_meshUploader.updateVertexBuffer({&syncSelectedMeshCornerHeatmapColors()}, m_selectedMeshHeatmapUploadConfig);
}

void SceneManager::setSelectedMeshHeatmapColors(std::span<const glm::vec3> colors)
{
    auto &selectedMesh = m_editedMeshObject->getComponent<MeshComponent>().mesh();
    selectedMesh.setPerUniqueVertexArray("heatmapColors", colors);
    updateSelectedMeshHeatmapBuffer();
}

void SceneManager::updateSelectedMeshHighlightColors()
{
    if (!m_editedMeshObject)
    {
        return;
    }
    // SelectionManager owns the coloring itself (persistent buffer, tool-customizable highlight
    // color) — this just pushes its result to the currently edited Mesh + GPU (see
    // setEditedMeshObject(); SelectionManager::rebind() keeps its color buffer sized to whichever
    // mesh that currently is).
    auto &editedMesh = m_editedMeshObject->getComponent<MeshComponent>().mesh();
    editedMesh.setPerUniqueVertexArray("color", std::span<const glm::vec3>(m_selectionManager->getColors()));
    updateSelectedMeshPointsBuffer();
}

const IndexBufferUploadPerMeshResult &SceneManager::selectedMeshIndexRange() const
{
    return gpu().indexRange(m_editedMeshObject->getComponent<MeshComponent>().mesh());
}

void SceneManager::setEditorMode(EditorMode mode)
{
    if (mode == m_editorMode)
    {
        return;
    }

    const EditorMode previousMode = m_editorMode;
    m_editorMode                  = mode;
    if (previousMode == EditorMode::Edit && mode != EditorMode::Edit)
    {
        m_selectionManager->clearSelection();
    }

    for (const auto &callback : m_editorModeChangedCallbacks)
    {
        callback(mode);
    }
}

void SceneManager::registerEditorModeChangedCallback(std::function<void(EditorMode)> callback)
{
    m_editorModeChangedCallbacks.push_back(std::move(callback));
}

} // namespace lr
