#include "SceneManager.hpp"

#include "Camera.hpp"
#include "Light.hpp"
#include "StaticMesh.hpp"
#include "Transform.hpp"

#include "core/app/Viewer.hpp"

#include <stdexcept>

namespace lr
{

namespace
{
// A light whose current type isn't AreaLight draws as a degenerate, zero-emissive quad rather
// than std::get-ing a variant that isn't AreaLight.
const AreaLight kHiddenAreaLightVisual{ {glm::vec3(0.0f), 0.0f}, glm::vec2(0.0f) };
}

SceneManager::SceneManager(ResourceRegistry &registry, uint32_t materialCapacity,
                            std::function<Material()> defaultMaterialFactory)
    : m_registry(registry)
    , m_meshUploader(registry)
    , m_materialUploader(registry)
    , m_lightUploader(registry)
    , m_cameraUploader(registry)
    , m_materialStore(materialCapacity, std::move(defaultMaterialFactory))
{
}

void SceneManager::initialize(const AreaLightVisualConfig &areaLightVisualConfig,
                               const GpuMaterialLayout &materialLayout,
                               const std::vector<std::string> &vertexAttributeNames)
{
    if (!m_scene)
        throw std::runtime_error("SceneManager::initialize: scene must be set first (see setScene)");
    if (!m_mainMeshObject)
        throw std::runtime_error("SceneManager::initialize: main mesh object must be set first (see setMainMeshObject)");
    if (!m_defaultCamera)
        throw std::runtime_error("SceneManager::initialize: default camera must be set first (see setDefaultCamera)");

    createLightVisuals(areaLightVisualConfig);
    uploadLights();

    uploadMeshes(materialLayout, vertexAttributeNames);

    // The camera buffer is registered by CameraUploader's constructor, but this is what actually
    // populates it, so do it once now rather than waiting for flushDirty()'s first pass. Ongoing
    // Camera/Transform edits (Scene Hierarchy GUI, orbit controller, etc.) are picked up by
    // flushDirty() — see its doc comment.
    updateCamera();
}

void SceneManager::registerCallbacks(Viewer &viewer)
{
    viewer.onUpdate([this](float dt, VkExtent2D extent) {
        setAspect((extent.height == 0)
            ? 1.0f
            : static_cast<float>(extent.width) / static_cast<float>(extent.height));
    });

    viewer.onLateUpdate([this](float dt, VkExtent2D extent) {
        flushDirty();
    });
}

void SceneManager::createLightVisuals(const AreaLightVisualConfig &config)
{
    m_areaLightVisualConfig = config;

    m_lightVisualObjects.clear();
    for (const auto &object : m_scene->sceneObjects())
        if (object->hasComponent<Light>())
            m_lightVisualObjects.push_back(object.get());

    for (SceneObject *lightObject : m_lightVisualObjects)
    {
        const auto *areaLight = std::get_if<AreaLight>(&lightObject->getComponent<Light>().light);
        const AreaLight &lightData = areaLight ? *areaLight : kHiddenAreaLightVisual;
        const Transform &transform = lightObject->getComponent<Transform>();

        const MaterialHandle handle = m_materialStore.acquire(buildAreaLightMaterial(lightData, config));

        Mesh quadMesh;
        buildAreaLightQuadMesh(quadMesh, transform, lightData, handle, config);

        lightObject->addComponent<StaticMesh>(quadMesh, std::vector<MaterialHandle>{handle}, m_materialStore,
                                               /*hideFromGui=*/true);
    }
}

void SceneManager::gatherGeometry(const std::vector<std::string> &vertexAttributeNames)
{
    auto &mainStaticMesh = m_mainMeshObject->getComponent<StaticMesh>();

    m_geometryMeshes = { &mainStaticMesh.mesh() };
    m_meshTransforms = { &m_mainMeshObject->getComponent<Transform>() };

    for (SceneObject *lightVisualObject : m_lightVisualObjects)
    {
        m_geometryMeshes.push_back(&lightVisualObject->getComponent<StaticMesh>().mesh());
        // Light visuals bake their Transform into vertex positions directly (see
        // AreaLightVisual.hpp), so they'd be double-transformed by also applying their Transform
        // here — nullptr means "draw with an identity model matrix".
        m_meshTransforms.push_back(nullptr);
    }

    m_meshPositionUploadConfig = {
        .vertexBufferName = m_mainMeshPositionBufferName,
        .includePosition  = true
    };
    m_meshAttributeUploadConfig = {
        .vertexBufferName     = m_mainMeshVertexBufferName,
        .vertexAttributeNames = vertexAttributeNames
    };
}

void SceneManager::uploadMeshes(const GpuMaterialLayout &materialLayout,
                                 const std::vector<std::string> &vertexAttributeNames)
{
    m_materialLayout = materialLayout;

    gatherGeometry(vertexAttributeNames);

    m_meshPositions = m_meshUploader.uploadVertexBuffer(m_geometryMeshes, m_meshPositionUploadConfig);
    m_meshUploader.uploadVertexBuffer(m_geometryMeshes, m_meshAttributeUploadConfig);

    // Color buffer is dynamic so selection highlights can be updated each frame.
    const auto &mainMesh = m_mainMeshObject->getComponent<StaticMesh>().mesh();
    std::vector<glm::vec3> pointColors(mainMesh.vertexCount(), glm::vec3(1.0f, 0.0f, 1.0f));
    m_registry.registerDynamicBuffer(m_mainMeshColorBufferName,
                                     pointColors.size() * sizeof(glm::vec3),
                                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    m_registry.updateBuffer(m_mainMeshColorBufferName,
                            pointColors.data(),
                            pointColors.size() * sizeof(glm::vec3));

    m_indexBuffer = m_meshUploader.uploadIndexBuffer(m_geometryMeshes, { .indexBufferName = m_mainMeshIndexBufferName });
    m_meshUploader.uploadFaceGroupBuffer(m_geometryMeshes, { .faceGroupBufferName = m_mainMeshFaceGroupBufferName });

    m_materialUploadResult = m_materialUploader.upload(m_materialStore.snapshot(), m_materialLayout, "material");
}

void SceneManager::uploadLights()
{
    std::vector<SceneObject*> lights;
    for (const auto &object : m_scene->sceneObjects())
        if (object->hasComponent<Light>())
            lights.push_back(object.get());
    m_lightUploader.upload(lights);
}

void SceneManager::updateMainMeshPositions()
{
    m_meshUploader.updateVertexBuffer(m_geometryMeshes, m_meshPositionUploadConfig);
}

void SceneManager::updateMaterials()
{
    m_materialUploader.update(m_materialStore.snapshot(), m_materialLayout, m_materialUploadResult);
}

void SceneManager::updateCamera()
{
    m_cameraUploader.upload(*m_defaultCamera, m_aspect);
}

void SceneManager::updateLightVisuals()
{
    for (SceneObject *lightObject : m_lightVisualObjects)
    {
        const auto *areaLight = std::get_if<AreaLight>(&lightObject->getComponent<Light>().light);
        const AreaLight &lightData = areaLight ? *areaLight : kHiddenAreaLightVisual;
        const Transform &transform = lightObject->getComponent<Transform>();
        auto &lightStaticMesh = lightObject->getComponent<StaticMesh>();

        const MaterialHandle handle = lightStaticMesh.materialHandles().front();
        buildAreaLightQuadMesh(lightStaticMesh.mesh(), transform, lightData, handle, m_areaLightVisualConfig);
        m_materialStore.get(handle) = buildAreaLightMaterial(lightData, m_areaLightVisualConfig);
    }

    m_meshUploader.updateVertexBuffer(m_geometryMeshes, m_meshPositionUploadConfig);
    m_meshUploader.updateVertexBuffer(m_geometryMeshes, m_meshAttributeUploadConfig);
    updateMaterials();
}

void SceneManager::flushDirty()
{
    auto &cameraComponent    = m_defaultCamera->getComponent<Camera>();
    auto &cameraTransform    = m_defaultCamera->getComponent<Transform>();
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
        anyLightVisualDirty |= lightObject->getComponent<Light>().isDirty() ||
                                lightObject->getComponent<Transform>().isDirty();

    if (anyLightVisualDirty)
    {
        updateLightVisuals();
        uploadLights();
        for (SceneObject *lightObject : m_lightVisualObjects)
        {
            lightObject->getComponent<Light>().clearDirty();
            lightObject->getComponent<Transform>().clearDirty();
        }
    }

    // Pushes material edits made via the Scene Hierarchy's sliders (StaticMesh::onGUIImpl) to the
    // GPU materials SSBO — without this, dragging a slider only updates the MaterialStore's CPU copy.
    auto &mainStaticMesh = m_mainMeshObject->getComponent<StaticMesh>();
    if (mainStaticMesh.isDirty())
    {
        updateMaterials();
        mainStaticMesh.clearDirty();
    }
}

}
