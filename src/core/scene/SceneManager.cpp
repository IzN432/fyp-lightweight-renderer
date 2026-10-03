#include "SceneManager.hpp"

#include "Camera.hpp"
#include "Light.hpp"
#include "MeshComponent.hpp"
#include "TransformComponent.hpp"
#include "features/animation/AnimatorComponent.hpp"
#include "features/linear_blend_skinning/SkinComponent.hpp"

#include "core/app/Viewer.hpp"

#include <algorithm>
#include <stdexcept>

namespace lr
{

namespace
{
// A light whose current type isn't AreaLight draws as a degenerate, zero-emissive quad rather
// than std::get-ing a variant that isn't AreaLight.
const AreaLight kHiddenAreaLightVisual{{glm::vec3(0.0f), 0.0f}, glm::vec2(0.0f)};

// Matches SelectionManager's private kDefaultColor — seeded onto a mesh's "color" per-unique-vertex
// attribute the first time it becomes the edited mesh object, if it doesn't already have one (e.g.
// procedural geometry authored without editing in mind). SelectionManager overwrites it from here.
const glm::vec3 kDefaultVertexColor{1.0f, 0.0f, 1.0f};
} // namespace

SceneManager::SceneManager(ResourceRegistry &registry, uint32_t materialCapacity,
                           std::function<Material()> defaultMaterialFactory)
    : m_registry(registry), m_meshUploader(registry), m_materialUploader(registry), m_lightUploader(registry),
      m_cameraUploader(registry), m_skinUploader(registry),
      m_materialStore(materialCapacity, std::move(defaultMaterialFactory))
{}

SceneObject &SceneManager::load(const std::filesystem::path &path, const SceneLoaderConfig &config)
{
    if (!m_scene)
    {
        throw std::runtime_error("SceneManager::load: scene must be set first");
    }
    SceneLoadResult imported = SceneLoader::load(path, *m_scene, m_meshStore, m_materialStore, config);
    if (!imported.firstMeshObject)
    {
        throw std::runtime_error("SceneManager::load: imported scene does not instantiate a mesh");
    }

    addMeshObject(m_scene->getSceneObject(imported.firstMeshObject.value()));
    return m_scene->getSceneObject(imported.rootObject);
}

void SceneManager::addMeshObject(SceneObject &object)
{
    if (std::find(m_meshObjects.begin(), m_meshObjects.end(), &object) == m_meshObjects.end())
    {
        m_meshObjects.push_back(&object);
    }
    if (!m_editedMeshObject)
    {
        m_editedMeshObject = &object;
    }
}

SceneObject *SceneManager::removeSceneObjects(std::span<const SceneObjectId> ids)
{
    const auto removed = [&](const SceneObject *object) {
        return object && std::ranges::find(ids, object->id()) != ids.end();
    };
    std::erase_if(m_meshObjects, removed);
    std::erase_if(m_lightVisualObjects, removed);

    if (removed(m_editedMeshObject))
    {
        SceneObject *replacement = m_meshObjects.empty() ? nullptr : m_meshObjects.front();
        m_editedMeshObject = nullptr;
        if (replacement && m_selectionManager)
        {
            setEditedMeshObject(*replacement);
        }
        else
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
    if (!m_scene)
    {
        throw std::runtime_error("SceneManager::initialize: scene must be set first (see setScene)");
    }
    if (m_meshObjects.empty() || !m_editedMeshObject)
    {
        throw std::runtime_error("SceneManager::initialize: at least one mesh object must be registered");
    }
    if (!m_defaultCamera)
    {
        throw std::runtime_error("SceneManager::initialize: default camera must be set first (see setDefaultCamera)");
    }

    createLightVisuals(areaLightVisualConfig);
    uploadLights();

    uploadMeshes(materialLayout, vertexAttributeNames);

    // Constructed here rather than as a SceneManager member-initializer since it operates on the
    // selected mesh's Mesh/TransformComponent, which only exist once uploadMeshes() above has run. The
    // highlight-changed callback keeps the GPU color buffer in sync with selection state — the
    // caller (main.cpp) still owns wiring up a SelectionTool and its own UI on top of it.
    auto &selectedMesh = m_editedMeshObject->getComponent<MeshComponent>().mesh();
    m_selectionManager =
        std::make_unique<SelectionManager>(selectedMesh.positions(),
                                           m_editedMeshObject->getComponent<TransformComponent>(), input);
    m_selectionManager->registerColorsChangedCallback([this]() {
        updateSelectedMeshHighlightColors();
    });

    // The camera buffer is registered by CameraUploader's constructor, but this is what actually
    // populates it, so do it once now rather than waiting for flushDirty()'s first pass. Ongoing
    // Camera/TransformComponent edits (Scene Hierarchy GUI, orbit controller, etc.) are picked up by
    // flushDirty() — see its doc comment.
    updateCamera();
}

void SceneManager::registerCallbacks(Viewer &viewer)
{
    viewer.onUpdate([this](float dt, VkExtent2D extent) {
        setAspect((extent.height == 0) ? 1.0f : static_cast<float>(extent.width) / static_cast<float>(extent.height));
    });

    viewer.onUpdate([this](float dt, VkExtent2D extent) {
        m_selectionManager->updateCallback(dt, extent);
    });

    viewer.onUpdate([this](float dt, VkExtent2D) {
        updateAnimations(dt);
    });

    viewer.onLateUpdate([this](float dt, VkExtent2D extent) {
        updateSkins();
        flushDirty();
    });
}

void SceneManager::createLightVisuals(const AreaLightVisualConfig &config)
{
    m_areaLightVisualConfig = config;

    m_lightVisualObjects.clear();
    for (const auto &object : m_scene->sceneObjects())
    {
        if (!m_scene->contains(object->id())) continue;
        if (object->hasComponent<Light>())
        {
            m_lightVisualObjects.push_back(object.get());
        }
    }

    for (SceneObject *lightObject : m_lightVisualObjects)
    {
        const auto      *areaLight = std::get_if<AreaLight>(&lightObject->getComponent<Light>().light);
        const AreaLight &lightData = areaLight ? *areaLight : kHiddenAreaLightVisual;
        const TransformComponent &transform = lightObject->getComponent<TransformComponent>();

        const MaterialHandle handle = m_materialStore.acquire(buildAreaLightMaterial(lightData, config));

        Mesh quadMesh;
        buildAreaLightQuadMesh(quadMesh, transform, lightData, handle, config);

        const MeshHandle meshHandle = m_meshStore.add(std::move(quadMesh));
        lightObject->addComponent<MeshComponent>(meshHandle, m_meshStore, std::vector<MaterialHandle>{handle},
                                                  m_materialStore, /*hideFromGui=*/true);
    }
}

void SceneManager::gatherGeometry(const std::vector<std::string> &vertexAttributeNames)
{
    m_geometryMeshes.clear();
    m_meshTransforms.clear();
    m_meshSkins.clear();
    m_geometryObjects.clear();
    for (SceneObject *object : m_meshObjects)
    {
        m_geometryObjects.push_back(object);
        m_geometryMeshes.push_back(&object->getComponent<MeshComponent>().mesh());
        m_meshTransforms.push_back(&object->getComponent<TransformComponent>());
        m_meshSkins.push_back(object->hasComponent<SkinComponent>()
                                   ? &object->getComponent<SkinComponent>().skin()
                                   : nullptr);
    }

    for (SceneObject *lightVisualObject : m_lightVisualObjects)
    {
        m_geometryObjects.push_back(lightVisualObject);
        m_geometryMeshes.push_back(&lightVisualObject->getComponent<MeshComponent>().mesh());
        // Light visuals bake their TransformComponent into vertex positions directly (see
        // AreaLightVisual.hpp), so they'd be double-transformed by also applying their TransformComponent
        // here — nullptr means "draw with an identity model matrix".
        m_meshTransforms.push_back(nullptr);
        m_meshSkins.push_back(nullptr);
    }

    m_meshPositionUploadConfig  = {.vertexBufferName = m_meshPositionBufferName, .includePosition = true};
    m_meshAttributeUploadConfig = {.vertexBufferName     = m_meshVertexBufferName,
                                   .vertexAttributeNames = vertexAttributeNames};
}

void SceneManager::uploadMeshes(const GpuMaterialLayout        &materialLayout,
                                const std::vector<std::string> &vertexAttributeNames)
{
    m_materialLayout = materialLayout;

    gatherGeometry(vertexAttributeNames);

    m_meshPositions = m_meshUploader.uploadVertexBuffer(m_geometryMeshes, m_meshPositionUploadConfig);
    m_meshUploader.uploadVertexBuffer(m_geometryMeshes, m_meshAttributeUploadConfig);

    // Deduped position + color, interleaved — unlike the buffer above (duped per UV-seam corner,
    // for GeometryPass), this is mesh.positions() verbatim, matching the index space VertexManager/
    // SelectionManager and the points-picking overlay already operate in. Color comes from the
    // selected mesh's own per-unique-vertex attribute.
    auto &selectedMesh = m_editedMeshObject->getComponent<MeshComponent>().mesh();
    ensureSelectedMeshAttributes(selectedMesh);
    m_selectedMeshPoints =
        m_meshUploader.uploadUniqueVertexBuffer({&selectedMesh}, m_selectedMeshPointsUploadConfig);

    // Seeds the corner-domain heatmap attribute from the per-unique-vertex analysis colors the
    // selected mesh, then uploads the heatmap buffer from it.
    m_selectedMeshHeatmap = m_meshUploader.uploadVertexBuffer(
        {&syncSelectedMeshCornerHeatmapColors()}, m_selectedMeshHeatmapUploadConfig);

    m_indexBuffer = m_meshUploader.uploadIndexBuffer(m_geometryMeshes, {.indexBufferName = m_meshIndexBufferName});
    m_meshUploader.uploadFaceGroupBuffer(m_geometryMeshes, {.faceGroupBufferName = m_meshFaceGroupBufferName});
    m_skinUploadResult = m_skinUploader.upload(m_geometryMeshes, m_meshSkins);

    m_materialUploadResult = m_materialUploader.upload(m_materialStore.snapshot(), m_materialLayout, "material");
}

void SceneManager::uploadLights()
{
    std::vector<SceneObject *> lights;
    for (const auto &object : m_scene->sceneObjects())
    {
        if (!m_scene->contains(object->id())) continue;
        if (object->hasComponent<Light>())
        {
            lights.push_back(object.get());
        }
    }
    m_lightUploader.upload(lights);
}

void SceneManager::updateSelectedMeshPositions()
{
    m_meshUploader.updateVertexBuffer(m_geometryMeshes, m_meshPositionUploadConfig);
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
    m_selectedMeshPoints =
        m_meshUploader.replaceUniqueVertexBuffer({&mesh}, m_selectedMeshPointsUploadConfig);
    m_selectedMeshHeatmap = m_meshUploader.replaceVertexBuffer(
        {&syncSelectedMeshCornerHeatmapColors()}, m_selectedMeshHeatmapUploadConfig);
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
        mesh.setPerUniqueVertexArray<glm::vec3>(
            "heatmapColors", std::vector<glm::vec3>(mesh.uniquePositionCount(), glm::vec3(0.0f)));
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
    const Mesh *selectedMesh = &m_editedMeshObject->getComponent<MeshComponent>().mesh();
    const auto  it           = std::find(m_geometryMeshes.begin(), m_geometryMeshes.end(), selectedMesh);
    if (it == m_geometryMeshes.end())
    {
        throw std::runtime_error("SceneManager::selectedMeshIndexRange: selected mesh is not renderable geometry");
    }
    return m_indexBuffer.singleMeshResults.at(static_cast<size_t>(std::distance(m_geometryMeshes.begin(), it)));
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

void SceneManager::updateAnimations(float deltaSeconds)
{
    for (const auto &object : m_scene->sceneObjects())
    {
        if (!m_scene->contains(object->id())) continue;
        if (object->hasComponent<AnimatorComponent>())
        {
            object->getComponent<AnimatorComponent>().update(deltaSeconds);
        }
    }
}

void SceneManager::registerEditorModeChangedCallback(std::function<void(EditorMode)> callback)
{
    m_editorModeChangedCallbacks.push_back(std::move(callback));
}

void SceneManager::updateMaterials()
{
    m_materialUploader.update(m_materialStore.snapshot(), m_materialLayout, m_materialUploadResult);
}

void SceneManager::updateCamera() { m_cameraUploader.upload(*m_defaultCamera, m_aspect); }

void SceneManager::updateSkins()
{
    for (size_t i = 0; i < m_meshSkins.size(); ++i)
    {
        if (!m_meshSkins[i] || !m_scene->contains(m_geometryObjects[i]->id()))
        {
            continue;
        }

        const bool missingJoint = std::ranges::any_of(m_meshSkins[i]->joints(), [&](const Joint &joint) {
            return !m_scene->contains(joint.sceneObject);
        });
        if (missingJoint) continue;

        const TransformComponent *transform = m_meshTransforms[i];
        m_meshSkins[i]->evaluate(transform ? transform->worldMatrix() : glm::mat4(1.0f));
    }
    m_skinUploader.updateJointMatrices(m_meshSkins);
}

void SceneManager::updateLightVisuals()
{
    for (SceneObject *lightObject : m_lightVisualObjects)
    {
        const auto      *areaLight       = std::get_if<AreaLight>(&lightObject->getComponent<Light>().light);
        const AreaLight &lightData       = areaLight ? *areaLight : kHiddenAreaLightVisual;
        const TransformComponent &transform = lightObject->getComponent<TransformComponent>();
        auto            &lightMesh = lightObject->getComponent<MeshComponent>();

        const MaterialHandle handle = lightMesh.materialHandles().front();
        buildAreaLightQuadMesh(lightMesh.mesh(), transform, lightData, handle, m_areaLightVisualConfig);
        m_materialStore.get(handle) = buildAreaLightMaterial(lightData, m_areaLightVisualConfig);
    }

    m_meshUploader.updateVertexBuffer(m_geometryMeshes, m_meshPositionUploadConfig);
    m_meshUploader.updateVertexBuffer(m_geometryMeshes, m_meshAttributeUploadConfig);
    updateMaterials();
}

void SceneManager::flushDirty()
{
    auto &cameraComponent = m_defaultCamera->getComponent<Camera>();
    auto &cameraTransform = m_defaultCamera->getComponent<TransformComponent>();
    if (cameraComponent.isDirty() || cameraTransform.isDirty())
    {
        updateCamera();
        cameraComponent.clearDirty();
        cameraTransform.clearDirty();
    }

    // Any single light visual going dirty rebuilds every light visual, since updateLightVisuals()
    // repacks the shared vertex/attribute buffers for all of them at once — including switching a
    // light to a different type at runtime (see Light::onGUIImpl's type combo), at which point its
    // quad collapses to (or springs from) the hidden zero-sized state.
    bool anyLightVisualDirty = false;
    for (SceneObject *lightObject : m_lightVisualObjects)
    {
        anyLightVisualDirty |=
            lightObject->getComponent<Light>().isDirty() ||
            lightObject->getComponent<TransformComponent>().isDirty();
    }

    if (anyLightVisualDirty)
    {
        updateLightVisuals();
        uploadLights();
        for (SceneObject *lightObject : m_lightVisualObjects)
        {
            lightObject->getComponent<Light>().clearDirty();
            lightObject->getComponent<TransformComponent>().clearDirty();
        }
    }

    // Pushes material edits made via the Scene Hierarchy's sliders (MeshComponent::onGUIImpl) to the
    // GPU materials SSBO — without this, dragging a slider only updates the MaterialStore's CPU copy.
    bool materialsDirty = false;
    for (SceneObject *object : m_meshObjects)
    {
        auto &meshComponent = object->getComponent<MeshComponent>();
        materialsDirty |= meshComponent.isDirty();
        meshComponent.clearDirty();
    }
    if (materialsDirty)
    {
        updateMaterials();
    }
}

} // namespace lr
