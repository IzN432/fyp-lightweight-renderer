#include "SceneGpu.hpp"

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
} // namespace

SceneGpu::SceneGpu(ResourceRegistry &registry, Scene &scene, MeshStore &meshStore, MaterialStore &materialStore)
    : m_registry(registry), m_scene(scene), m_meshStore(meshStore), m_materialStore(materialStore),
      m_meshUploader(registry), m_materialUploader(registry), m_lightUploader(registry), m_cameraUploader(registry),
      m_skinUploader(registry)
{}

void SceneGpu::addMeshObject(SceneObject &object)
{
    if (std::find(m_meshObjects.begin(), m_meshObjects.end(), &object) == m_meshObjects.end())
    {
        m_meshObjects.push_back(&object);
    }
}

void SceneGpu::addLoaded(const SceneLoadResult &result)
{
    for (const auto objectId : result.nodeObjects)
    {
        if (!objectId)
        {
            continue;
        }
        SceneObject &object = m_scene.getSceneObject(*objectId);
        if (object.hasComponent<MeshComponent>())
        {
            addMeshObject(object);
        }
    }
    m_pendingTextureUpdates.insert(m_pendingTextureUpdates.end(), result.materialHandles.begin(),
                                   result.materialHandles.end());
}

void SceneGpu::removeSceneObjects(std::span<const SceneObjectId> ids)
{
    const auto removed = [&](const SceneObject *object) {
        return object && std::ranges::find(ids, object->id()) != ids.end();
    };
    std::erase_if(m_meshObjects, removed);
    std::erase_if(m_lightVisualObjects, removed);
}

void SceneGpu::initialize(const AreaLightVisualConfig &areaLightVisualConfig, const GpuMaterialLayout &materialLayout,
                          const std::vector<std::string> &vertexAttributeNames)
{
    if (m_meshObjects.empty())
    {
        throw std::runtime_error("SceneGpu::initialize: at least one mesh object must be registered");
    }
    if (!m_camera)
    {
        throw std::runtime_error("SceneGpu::initialize: camera must be set first (see setCamera)");
    }

    createLightVisuals(areaLightVisualConfig);
    uploadLights();
    uploadMeshes(materialLayout, vertexAttributeNames);

    // The camera buffer is registered by CameraUploader's constructor, but this is what actually
    // populates it, so do it once now rather than waiting for flushDirty()'s first pass.
    updateCamera();
    m_initialized = true;
}

void SceneGpu::registerCallbacks(Viewer &viewer)
{
    viewer.onUpdate([this](float, VkExtent2D extent) {
        setAspect((extent.height == 0) ? 1.0f : static_cast<float>(extent.width) / static_cast<float>(extent.height));
    });

    viewer.onUpdate([this](float dt, VkExtent2D) {
        updateAnimations(dt);
    });

    viewer.onLateUpdate([this](float, VkExtent2D) {
        updateSkins();
        flushDirty();
    });
}

void SceneGpu::createLightVisuals(const AreaLightVisualConfig &config)
{
    m_areaLightVisualConfig = config;

    m_lightVisualObjects.clear();
    for (const auto &object : m_scene.sceneObjects())
    {
        if (!m_scene.contains(object->id()))
        {
            continue;
        }
        if (object->hasComponent<Light>())
        {
            m_lightVisualObjects.push_back(object.get());
        }
    }

    for (SceneObject *lightObject : m_lightVisualObjects)
    {
        const auto               *areaLight = std::get_if<AreaLight>(&lightObject->getComponent<Light>().light);
        const AreaLight          &lightData = areaLight ? *areaLight : kHiddenAreaLightVisual;
        const TransformComponent &transform = lightObject->getComponent<TransformComponent>();

        const MaterialHandle handle = m_materialStore.acquire(buildAreaLightMaterial(lightData, config));

        Mesh quadMesh;
        buildAreaLightQuadMesh(quadMesh, transform, lightData, handle, config);

        const MeshHandle meshHandle = m_meshStore.add(std::move(quadMesh));
        lightObject->addComponent<MeshComponent>(meshHandle, m_meshStore, std::vector<MaterialHandle>{handle},
                                                 m_materialStore, /*hideFromGui=*/true);
    }
}

void SceneGpu::gatherGeometry(const std::vector<std::string> &vertexAttributeNames)
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
        m_meshSkins.push_back(object->hasComponent<SkinComponent>() ? &object->getComponent<SkinComponent>().skin()
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

void SceneGpu::uploadMeshes(const GpuMaterialLayout        &materialLayout,
                            const std::vector<std::string> &vertexAttributeNames)
{
    m_materialLayout       = materialLayout;
    m_vertexAttributeNames = vertexAttributeNames;

    gatherGeometry(vertexAttributeNames);

    m_meshPositions = m_meshUploader.uploadVertexBuffer(m_geometryMeshes, m_meshPositionUploadConfig);
    m_meshUploader.uploadVertexBuffer(m_geometryMeshes, m_meshAttributeUploadConfig);

    m_indexBuffer = m_meshUploader.uploadIndexBuffer(m_geometryMeshes, {.indexBufferName = m_meshIndexBufferName});
    m_meshUploader.uploadFaceGroupBuffer(m_geometryMeshes, {.faceGroupBufferName = m_meshFaceGroupBufferName});
    m_skinUploadResult = m_skinUploader.upload(m_geometryMeshes, m_meshSkins);

    m_materialUploadResult = m_materialUploader.upload(m_materialStore.snapshot(), m_materialLayout, "material");
    m_pendingTextureUpdates.clear();
}

void SceneGpu::rebuildGeometry()
{
    if (m_meshObjects.empty())
    {
        throw std::runtime_error("SceneGpu::rebuildGeometry: scene has no renderable meshes");
    }

    gatherGeometry(m_vertexAttributeNames);
    m_meshPositions = m_meshUploader.replaceVertexBuffer(m_geometryMeshes, m_meshPositionUploadConfig);
    m_meshUploader.replaceVertexBuffer(m_geometryMeshes, m_meshAttributeUploadConfig);
    m_indexBuffer = m_meshUploader.replaceIndexBuffer(m_geometryMeshes, {.indexBufferName = m_meshIndexBufferName});
    m_meshUploader.replaceFaceGroupBuffer(m_geometryMeshes, {.faceGroupBufferName = m_meshFaceGroupBufferName});
    m_skinUploadResult = m_skinUploader.upload(m_geometryMeshes, m_meshSkins);
    updateMaterials();
    if (!m_pendingTextureUpdates.empty())
    {
        std::ranges::sort(m_pendingTextureUpdates);
        const auto unique = std::ranges::unique(m_pendingTextureUpdates);
        m_pendingTextureUpdates.erase(unique.begin(), unique.end());
        m_materialUploader.updateTextures(m_materialStore.snapshot(), m_materialLayout, m_materialUploadResult,
                                          m_pendingTextureUpdates);
        m_pendingTextureUpdates.clear();
    }
}

void SceneGpu::uploadLights()
{
    std::vector<SceneObject *> lights;
    for (const auto &object : m_scene.sceneObjects())
    {
        if (!m_scene.contains(object->id()))
        {
            continue;
        }
        if (object->hasComponent<Light>())
        {
            lights.push_back(object.get());
        }
    }
    m_lightUploader.upload(lights);
}

void SceneGpu::updatePositions() { m_meshUploader.updateVertexBuffer(m_geometryMeshes, m_meshPositionUploadConfig); }

void SceneGpu::updateMaterials()
{
    m_materialUploader.update(m_materialStore.snapshot(), m_materialLayout, m_materialUploadResult);
}

void SceneGpu::updateCamera() { m_cameraUploader.upload(*m_camera, m_aspect); }

void SceneGpu::updateAnimations(float deltaSeconds)
{
    for (const auto &object : m_scene.sceneObjects())
    {
        if (!m_scene.contains(object->id()))
        {
            continue;
        }
        if (object->hasComponent<AnimatorComponent>())
        {
            object->getComponent<AnimatorComponent>().update(deltaSeconds);
        }
    }
}

void SceneGpu::updateSkins()
{
    for (size_t i = 0; i < m_meshSkins.size(); ++i)
    {
        if (!m_meshSkins[i] || !m_scene.contains(m_geometryObjects[i]->id()))
        {
            continue;
        }

        const bool missingJoint = std::ranges::any_of(m_meshSkins[i]->joints(), [&](const Joint &joint) {
            return !m_scene.contains(joint.sceneObject);
        });
        if (missingJoint)
        {
            continue;
        }

        const TransformComponent *transform = m_meshTransforms[i];
        m_meshSkins[i]->evaluate(transform ? transform->worldMatrix() : glm::mat4(1.0f));
    }
    m_skinUploader.updateJointMatrices(m_meshSkins);
}

void SceneGpu::updateLightVisuals()
{
    for (SceneObject *lightObject : m_lightVisualObjects)
    {
        const auto               *areaLight = std::get_if<AreaLight>(&lightObject->getComponent<Light>().light);
        const AreaLight          &lightData = areaLight ? *areaLight : kHiddenAreaLightVisual;
        const TransformComponent &transform = lightObject->getComponent<TransformComponent>();
        auto                     &lightMesh = lightObject->getComponent<MeshComponent>();

        const MaterialHandle handle = lightMesh.materialHandles().front();
        buildAreaLightQuadMesh(lightMesh.mesh(), transform, lightData, handle, m_areaLightVisualConfig);
        m_materialStore.get(handle) = buildAreaLightMaterial(lightData, m_areaLightVisualConfig);
    }

    m_meshUploader.updateVertexBuffer(m_geometryMeshes, m_meshPositionUploadConfig);
    m_meshUploader.updateVertexBuffer(m_geometryMeshes, m_meshAttributeUploadConfig);
    updateMaterials();
}

void SceneGpu::flushDirty()
{
    auto &cameraComponent = m_camera->getComponent<Camera>();
    auto &cameraTransform = m_camera->getComponent<TransformComponent>();
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
            lightObject->getComponent<Light>().isDirty() || lightObject->getComponent<TransformComponent>().isDirty();
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

const IndexBufferUploadPerMeshResult &SceneGpu::indexRange(const Mesh &mesh) const
{
    const auto it = std::find(m_geometryMeshes.begin(), m_geometryMeshes.end(), &mesh);
    if (it == m_geometryMeshes.end())
    {
        throw std::runtime_error("SceneGpu::indexRange: mesh is not registered geometry");
    }
    return m_indexBuffer.singleMeshResults.at(static_cast<size_t>(std::distance(m_geometryMeshes.begin(), it)));
}

GeometryPass::Config SceneGpu::geometryPassConfig() const
{
    const auto &textures = m_materialUploadResult.textureNameMap;
    return GeometryPass::Config{
        .cameraBufferResourceName                  = cameraBufferName(),
        .vertexBufferResourceNames                 = {{0, m_meshPositionBufferName}, {1, m_meshVertexBufferName}},
        .vertexBufferUploadResult                  = m_meshPositions,
        .indexBufferUploadResult                   = m_indexBuffer,
        .meshTransforms                            = m_meshTransforms,
        .meshObjects                               = m_geometryObjects,
        .skinDrawInfos                             = m_skinUploadResult.drawInfos,
        .indexBufferResourceName                   = m_meshIndexBufferName,
        .faceGroupBufferResourceName               = m_meshFaceGroupBufferName,
        .diffuseTextureArrayResourceName           = textures.at(conventions::baseColorTexture),
        .normalTextureArrayResourceName            = textures.at(conventions::normalTexture),
        .metallicRoughnessTextureArrayResourceName = textures.at(conventions::metallicRoughnessTexture),
        .emissiveTextureArrayResourceName          = textures.at(conventions::emissiveTexture),
        .materialBufferResourceName                = m_materialUploadResult.materialInfoBufferName,
        .skinInfluenceEntriesBufferResourceName    = skinInfluenceEntriesBufferName(),
        .skinInfluenceOffsetsBufferResourceName    = skinInfluenceOffsetsBufferName(),
        .skinPositionIndicesBufferResourceName     = skinPositionIndicesBufferName(),
        .skinJointMatricesBufferResourceName       = skinJointMatricesBufferName(),
        .materialCount                             = m_materialStore.capacity(),
    };
}

} // namespace lr
