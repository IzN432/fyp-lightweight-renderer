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
#include <string>

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

SceneGpu::~SceneGpu() { releaseLightVisuals(); }

void SceneGpu::releaseLightVisuals()
{
    for (const LightVisual &visual : m_lightVisuals)
    {
        m_materialStore.release(visual.material);
    }
    m_lightVisuals.clear();
}

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

void SceneGpu::queueMaterialTextures(std::span<const MaterialHandle> handles)
{
    m_pendingTextureUpdates.insert(m_pendingTextureUpdates.end(), handles.begin(), handles.end());
}

void SceneGpu::clearSceneResources()
{
    releaseLightVisuals();
    m_meshObjects.clear();
    dropGeometry();
    m_pendingTextureUpdates.clear();
    // Keep SkinUploader's structural expectations synchronized with the cleared scene. Without an
    // explicit empty upload it still expects the previous scene's skin list and the next per-frame
    // palette update fails before (or when) replacement geometry is installed.
    m_skinUploadResult = m_skinUploader.upload({}, {});
}

void SceneGpu::removeSceneObjects(std::span<const SceneObjectId> ids)
{
    const auto removed = [&](const SceneObject *object) {
        return object && std::ranges::find(ids, object->id()) != ids.end();
    };
    std::erase_if(m_meshObjects, removed);
    // Light visuals stay until syncLights(): m_geometryMeshes still points at their meshes, and
    // GeometryPass already skips draws whose object has left the scene.
}

bool SceneGpu::isLiveLight(const SceneObject &object) const
{
    return m_scene.contains(object.id()) && object.hasComponent<Light>();
}

bool SceneGpu::lightsChanged() const
{
    // Each live light has at most one visual, so the sets match when every visual's light is live
    // and the counts agree.
    size_t liveLights = 0;
    for (const auto &object : m_scene.sceneObjects())
    {
        liveLights += isLiveLight(*object) ? 1 : 0;
    }
    return liveLights != m_lightVisuals.size() || std::ranges::any_of(m_lightVisuals, [&](const LightVisual &visual) {
               return !isLiveLight(*visual.light);
           });
}

SceneGpu::LightVisual SceneGpu::makeLightVisual(SceneObject &light)
{
    const auto               *areaLight = std::get_if<AreaLight>(&light.getComponent<Light>().light);
    const AreaLight          &lightData = areaLight ? *areaLight : kHiddenAreaLightVisual;
    const TransformComponent &transform = light.getComponent<TransformComponent>();

    LightVisual visual{.light    = &light,
                       .mesh     = std::make_unique<Mesh>(),
                       .material = m_materialStore.acquire(buildAreaLightMaterial(lightData, m_areaLightVisualConfig))};
    buildAreaLightQuadMesh(*visual.mesh, transform, lightData, visual.material, m_areaLightVisualConfig);
    return visual;
}

void SceneGpu::syncLights()
{
    // Checked before anything changes, so a throw leaves the current geometry intact.
    const auto liveLights = std::ranges::count_if(m_scene.sceneObjects(), [&](const auto &object) {
        return isLiveLight(*object);
    });
    if (static_cast<uint32_t>(liveLights) > maxLights())
    {
        throw std::length_error("SceneGpu: the scene has " + std::to_string(liveLights) + " lights, but at most " +
                                std::to_string(maxLights()) + " are supported");
    }

    std::erase_if(m_lightVisuals, [&](const LightVisual &visual) {
        if (isLiveLight(*visual.light))
        {
            return false;
        }
        m_materialStore.release(visual.material);
        return true;
    });
    for (const auto &object : m_scene.sceneObjects())
    {
        const bool hasVisual = std::ranges::any_of(m_lightVisuals, [&](const LightVisual &visual) {
            return visual.light == object.get();
        });
        if (isLiveLight(*object) && !hasVisual)
        {
            m_lightVisuals.push_back(makeLightVisual(*object));
        }
    }
    uploadLights();
    rebuildGeometry();
}

CallbackConnection SceneGpu::onGeometryRebuilt(std::function<void(const SceneGpu &)> listener)
{
    return m_geometryRebuiltCallbacks.connect(std::move(listener));
}

CallbackConnection SceneGpu::onLightsUploaded(std::function<void(uint32_t)> listener)
{
    return m_lightsUploadedCallbacks.connect(std::move(listener));
}

void SceneGpu::initialize(const AreaLightVisualConfig &areaLightVisualConfig, const GpuMaterialLayout &materialLayout,
                          const std::vector<std::string> &vertexAttributeNames)
{
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
    m_connections.push_back(viewer.onUpdate([this](float, VkExtent2D extent) {
        setAspect((extent.height == 0) ? 1.0f : static_cast<float>(extent.width) / static_cast<float>(extent.height));
    }));

    m_connections.push_back(viewer.onUpdate([this](float dt, VkExtent2D) {
        updateAnimations(dt);
    }));

    m_connections.push_back(viewer.onLateUpdate([this](float, VkExtent2D) {
        updateSkins();
        flushDirty();
    }));
}

void SceneGpu::createLightVisuals(const AreaLightVisualConfig &config)
{
    m_areaLightVisualConfig = config;

    releaseLightVisuals();
    for (const auto &object : m_scene.sceneObjects())
    {
        if (isLiveLight(*object))
        {
            m_lightVisuals.push_back(makeLightVisual(*object));
        }
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

    for (const LightVisual &visual : m_lightVisuals)
    {
        // The light object stands in for the quad in the draw list: GeometryPass skips draws whose
        // object has left the scene.
        m_geometryObjects.push_back(visual.light);
        m_geometryMeshes.push_back(visual.mesh.get());
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

bool SceneGpu::hasDrawableGeometry() const
{
    uint32_t vertices = 0;
    uint32_t faces    = 0;
    for (const Mesh *mesh : m_geometryMeshes)
    {
        vertices += mesh->vertexCount();
        faces += mesh->faceCount();
    }
    return vertices > 0 && faces > 0;
}

void SceneGpu::dropGeometry()
{
    m_geometryMeshes.clear();
    m_meshTransforms.clear();
    m_meshSkins.clear();
    m_geometryObjects.clear();
    m_meshPositions = {};
    m_indexBuffer   = {};
}

void SceneGpu::rebuildGeometry()
{
    gatherGeometry(m_vertexAttributeNames);
    if (hasDrawableGeometry())
    {
        m_meshPositions = m_meshUploader.replaceVertexBuffer(m_geometryMeshes, m_meshPositionUploadConfig);
        m_meshUploader.replaceVertexBuffer(m_geometryMeshes, m_meshAttributeUploadConfig);
        m_indexBuffer = m_meshUploader.replaceIndexBuffer(m_geometryMeshes, {.indexBufferName = m_meshIndexBufferName});
        m_meshUploader.replaceFaceGroupBuffer(m_geometryMeshes, {.faceGroupBufferName = m_meshFaceGroupBufferName});
    } else
    {
        // A scene with nothing to draw is a normal state (cleared before an import, or the last mesh
        // deleted). The shared buffers keep whatever they last held and MeshBufferCache keeps its
        // stamps: replacing them with empty payloads would gain nothing, and the stamps cannot
        // describe a draw list that shrank to nothing (see MeshBufferCache::synchronize). Publishing
        // empty draw lists is what makes it safe — GeometryPass issues no draws, so nothing reads
        // the stale contents, and the next rebuild with real geometry replaces the buffers properly.
        dropGeometry();
    }
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
    m_geometryRebuiltCallbacks.invoke(*this);
}

void SceneGpu::uploadLights()
{
    m_lightObjects.clear();
    for (const auto &object : m_scene.sceneObjects())
    {
        if (isLiveLight(*object))
        {
            m_lightObjects.push_back(object.get());
        }
    }
    m_lightUploader.upload(m_lightObjects);
    m_lightsUploadedCallbacks.invoke(numLights());
}

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
    for (const LightVisual &visual : m_lightVisuals)
    {
        if (!isLiveLight(*visual.light))
        {
            continue; // removed; dropped by the next syncLights()
        }
        const auto               *areaLight = std::get_if<AreaLight>(&visual.light->getComponent<Light>().light);
        const AreaLight          &lightData = areaLight ? *areaLight : kHiddenAreaLightVisual;
        const TransformComponent &transform = visual.light->getComponent<TransformComponent>();

        buildAreaLightQuadMesh(*visual.mesh, transform, lightData, visual.material, m_areaLightVisualConfig);
        m_materialStore.get(visual.material) = buildAreaLightMaterial(lightData, m_areaLightVisualConfig);
    }

    // The quads' new positions/attributes reach the GPU through synchronizeMeshes() at the end of
    // flushDirty().
    updateMaterials();
}

void SceneGpu::flushDirty()
{
    auto &cameraComponent = m_camera->getComponent<Camera>();
    auto &cameraTransform = m_camera->getComponent<TransformComponent>();
    if (cameraComponent.isDirty() || cameraTransform.isDirty() || m_cameraAspectDirty)
    {
        updateCamera();
        cameraComponent.clearDirty();
        cameraTransform.clearDirty();
        m_cameraAspectDirty = false;
    }

    // Lights added or removed since the last frame: new quads, a rebuilt geometry and a re-uploaded light
    // buffer, built from the lights' current state (so their dirty flags are handled below as well).
    if (lightsChanged())
    {
        syncLights();
    }

    // Any single light visual going dirty rebuilds every light visual, since updateLightVisuals()
    // repacks the shared vertex/attribute buffers for all of them at once — including switching a
    // light to a different type at runtime (see Light::onGUIImpl's type combo), at which point its
    // quad collapses to (or springs from) the hidden zero-sized state.
    bool anyLightVisualDirty = false;
    for (const LightVisual &visual : m_lightVisuals)
    {
        anyLightVisualDirty |=
            visual.light->getComponent<Light>().isDirty() || visual.light->getComponent<TransformComponent>().isDirty();
    }

    if (anyLightVisualDirty)
    {
        updateLightVisuals();
        uploadLights();
        for (const LightVisual &visual : m_lightVisuals)
        {
            visual.light->getComponent<Light>().clearDirty();
            visual.light->getComponent<TransformComponent>().clearDirty();
        }
    }

    // Material edits made via the Scene Hierarchy's sliders (MeshComponent::onGUIImpl) reach the GPU
    // materials SSBO here — without this, dragging a slider only updates the MaterialStore's CPU
    // copy. A component pointed at a different mesh needs more than that: the gathered Mesh* list
    // and every shared buffer packed from it describe the old one.
    bool materialsDirty = false;
    bool geometryDirty  = false;
    for (SceneObject *object : m_meshObjects)
    {
        auto &meshComponent = object->getComponent<MeshComponent>();
        materialsDirty |= meshComponent.isDirty(MeshComponent::Materials);
        geometryDirty |= meshComponent.isDirty(MeshComponent::Geometry);
        meshComponent.clearDirty();
    }
    if (geometryDirty)
    {
        // Re-packs and re-uploads everything, materials included, so it subsumes the branch below.
        // Safe without a device stall: the uploaders replace buffers through
        // ResourceRegistry::replaceUploadedBuffer, which retires the old allocation until the frames
        // still reading it have completed.
        rebuildGeometry();
    } else if (materialsDirty)
    {
        updateMaterials();
    }

    synchronizeMeshes();
}

void SceneGpu::synchronizeMeshes()
{
    if (m_geometryMeshes.empty())
    {
        return; // nothing registered: the buffers keep their last contents (see rebuildGeometry)
    }
    m_meshUploader.synchronizeVertexBuffer(m_geometryMeshes, m_meshPositionUploadConfig);
    m_meshUploader.synchronizeVertexBuffer(m_geometryMeshes, m_meshAttributeUploadConfig);
    m_meshUploader.synchronizeIndexBuffer(m_geometryMeshes, {.indexBufferName = m_meshIndexBufferName});
    m_meshUploader.synchronizeFaceGroupBuffer(m_geometryMeshes, {.faceGroupBufferName = m_meshFaceGroupBufferName});
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

SceneDrawList SceneGpu::drawList() const
{
    return SceneDrawList{m_meshPositions, m_indexBuffer, m_meshTransforms, m_geometryObjects,
                         m_skinUploadResult.drawInfos};
}

GeometryPass::Config SceneGpu::geometryPassConfig() const
{
    const auto &textures = m_materialUploadResult.textureNameMap;
    return GeometryPass::Config{
        .cameraBufferResourceName                  = cameraBufferName(),
        .vertexBufferResourceNames                 = {{0, m_meshPositionBufferName}, {1, m_meshVertexBufferName}},
        .draws                                     = drawList(),
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
