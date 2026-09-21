#include "SceneManager.hpp"

#include "Camera.hpp"
#include "Light.hpp"
#include "MeshComponent.hpp"
#include "TransformComponent.hpp"
#include "features/animation/AnimatorComponent.hpp"
#include "features/linear_blend_skinning/SkinComponent.hpp"

#include "core/app/Viewer.hpp"

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

    m_mainMeshObject = &m_scene->getSceneObject(imported.firstMeshObject.value());
    return m_scene->getSceneObject(imported.rootObject);
}

void SceneManager::initialize(const AreaLightVisualConfig    &areaLightVisualConfig,
                              const GpuMaterialLayout        &materialLayout,
                              const std::vector<std::string> &vertexAttributeNames, InputHandler &input)
{
    if (!m_scene)
    {
        throw std::runtime_error("SceneManager::initialize: scene must be set first (see setScene)");
    }
    if (!m_mainMeshObject)
    {
        throw std::runtime_error(
            "SceneManager::initialize: main mesh object must be set first (see setMainMeshObject)");
    }
    if (!m_defaultCamera)
    {
        throw std::runtime_error("SceneManager::initialize: default camera must be set first (see setDefaultCamera)");
    }

    createLightVisuals(areaLightVisualConfig);
    uploadLights();

    uploadMeshes(materialLayout, vertexAttributeNames);

    // Constructed here rather than as a SceneManager member-initializer since it operates on the
    // main mesh's Mesh/TransformComponent, which only exist once uploadMeshes() above has run. The
    // highlight-changed callback keeps the GPU color buffer in sync with selection state — the
    // caller (main.cpp) still owns wiring up a SelectionTool and its own UI on top of it.
    auto &mainMesh = m_mainMeshObject->getComponent<MeshComponent>().mesh();
    m_selectionManager =
        std::make_unique<SelectionManager>(mainMesh.positions(),
                                           m_mainMeshObject->getComponent<TransformComponent>(), input);
    m_selectionManager->registerColorsChangedCallback([this]() {
        updateMainMeshHighlightColors();
    });
    m_editedMeshObject = m_mainMeshObject;

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
    auto &mainMeshComponent = m_mainMeshObject->getComponent<MeshComponent>();

    m_geometryMeshes = {&mainMeshComponent.mesh()};
    m_meshTransforms = {&m_mainMeshObject->getComponent<TransformComponent>()};
    m_meshSkins = {m_mainMeshObject->hasComponent<SkinComponent>()
                       ? &m_mainMeshObject->getComponent<SkinComponent>().skin()
                       : nullptr};

    for (SceneObject *extraObject : m_extraMeshObjects)
    {
        m_geometryMeshes.push_back(&extraObject->getComponent<MeshComponent>().mesh());
        m_meshTransforms.push_back(&extraObject->getComponent<TransformComponent>());
        m_meshSkins.push_back(extraObject->hasComponent<SkinComponent>()
                                   ? &extraObject->getComponent<SkinComponent>().skin()
                                   : nullptr);
    }

    for (SceneObject *lightVisualObject : m_lightVisualObjects)
    {
        m_geometryMeshes.push_back(&lightVisualObject->getComponent<MeshComponent>().mesh());
        // Light visuals bake their TransformComponent into vertex positions directly (see
        // AreaLightVisual.hpp), so they'd be double-transformed by also applying their TransformComponent
        // here — nullptr means "draw with an identity model matrix".
        m_meshTransforms.push_back(nullptr);
        m_meshSkins.push_back(nullptr);
    }

    m_meshPositionUploadConfig  = {.vertexBufferName = m_mainMeshPositionBufferName, .includePosition = true};
    m_meshAttributeUploadConfig = {.vertexBufferName     = m_mainMeshVertexBufferName,
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
    // SelectionManager and the points-picking overlay already operate in. Color comes from the main
    // mesh's own "color" per-unique-vertex attribute (caller must seed it before initialize() — see
    // main.cpp), which is what SelectionManager's highlight indices are already in terms of.
    const auto &mainMesh = m_mainMeshObject->getComponent<MeshComponent>().mesh();
    m_mainMeshPoints     = m_meshUploader.uploadUniqueVertexBuffer({&mainMesh}, m_mainMeshPointsUploadConfig);

    // Seeds the corner-domain heatmap attribute from the per-unique-vertex analysis colors the
    // caller set (see main.cpp), then uploads the heatmap buffer from it.
    m_mainMeshHeatmap =
        m_meshUploader.uploadVertexBuffer({&syncMainMeshCornerHeatmapColors()}, m_mainMeshHeatmapUploadConfig);

    m_indexBuffer = m_meshUploader.uploadIndexBuffer(m_geometryMeshes, {.indexBufferName = m_mainMeshIndexBufferName});
    m_meshUploader.uploadFaceGroupBuffer(m_geometryMeshes, {.faceGroupBufferName = m_mainMeshFaceGroupBufferName});
    m_skinUploadResult = m_skinUploader.upload(m_geometryMeshes, m_meshSkins);

    m_materialUploadResult = m_materialUploader.upload(m_materialStore.snapshot(), m_materialLayout, "material");
}

void SceneManager::uploadLights()
{
    std::vector<SceneObject *> lights;
    for (const auto &object : m_scene->sceneObjects())
    {
        if (object->hasComponent<Light>())
        {
            lights.push_back(object.get());
        }
    }
    m_lightUploader.upload(lights);
}

void SceneManager::updateMainMeshPositions()
{
    m_meshUploader.updateVertexBuffer(m_geometryMeshes, m_meshPositionUploadConfig);
    updateMainMeshPointsBuffer();
    // Heatmap/Analysis mode stays scoped to mainMeshObject() regardless of what's currently being
    // edited (see setEditedMeshObject()'s doc comment) — skip the repack entirely when they differ,
    // since a drag on some other mesh can't have touched the main mesh's positions.
    if (m_editedMeshObject == m_mainMeshObject)
    {
        updateMainMeshHeatmapBuffer();
    }
}

void SceneManager::updateMainMeshPointsBuffer()
{
    const auto &editedMesh = m_editedMeshObject->getComponent<MeshComponent>().mesh();
    m_meshUploader.updateUniqueVertexBuffer({&editedMesh}, m_mainMeshPointsUploadConfig);
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

    // Procedural geometry may never have been authored with editing in mind — seed a default
    // "color" attribute the first time it's asked to become editable, rather than requiring every
    // mesh builder to remember to do this up front.
    if (!mesh.layout().findPerUniqueVertexAttr("color"))
    {
        std::vector<glm::vec3> defaultColors(mesh.uniquePositionCount(), kDefaultVertexColor);
        mesh.setPerUniqueVertexArray<glm::vec3>("color", defaultColors);
    }

    m_selectionManager->rebind(mesh.positions(), object.getComponent<TransformComponent>());
    m_mainMeshPoints = m_meshUploader.replaceUniqueVertexBuffer({&mesh}, m_mainMeshPointsUploadConfig);
}

Mesh &SceneManager::syncMainMeshCornerHeatmapColors()
{
    auto                  &mainMesh       = m_mainMeshObject->getComponent<MeshComponent>().mesh();
    const auto             uniqueColor    = mainMesh.getPerUniqueVertexArray<glm::vec3>("heatmapColors");
    std::vector<glm::vec3> cornerColor(mainMesh.vertexCount());
    for (uint32_t v = 0; v < mainMesh.vertexCount(); ++v)
    {
        cornerColor[v] = uniqueColor[mainMesh.positionIndices()[v]];
    }
    mainMesh.setPerVertexArray<glm::vec3>("heatmapColors", std::span<const glm::vec3>(cornerColor));
    return mainMesh;
}

void SceneManager::updateMainMeshHeatmapBuffer()
{
    m_meshUploader.updateVertexBuffer({&syncMainMeshCornerHeatmapColors()}, m_mainMeshHeatmapUploadConfig);
}

void SceneManager::setMainMeshHeatmapColors(std::span<const glm::vec3> colors)
{
    auto &mainMesh = m_mainMeshObject->getComponent<MeshComponent>().mesh();
    mainMesh.setPerUniqueVertexArray("heatmapColors", colors);
    updateMainMeshHeatmapBuffer();
}

void SceneManager::updateMainMeshHighlightColors()
{
    // SelectionManager owns the coloring itself (persistent buffer, tool-customizable highlight
    // color) — this just pushes its result to the currently edited Mesh + GPU (see
    // setEditedMeshObject(); SelectionManager::rebind() keeps its color buffer sized to whichever
    // mesh that currently is).
    auto &editedMesh = m_editedMeshObject->getComponent<MeshComponent>().mesh();
    editedMesh.setPerUniqueVertexArray("color", std::span<const glm::vec3>(m_selectionManager->getColors()));
    updateMainMeshPointsBuffer();
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
        if (!m_meshSkins[i])
        {
            continue;
        }

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
    auto &mainMeshComponent = m_mainMeshObject->getComponent<MeshComponent>();
    if (mainMeshComponent.isDirty())
    {
        updateMaterials();
        mainMeshComponent.clearDirty();
    }
}

} // namespace lr
