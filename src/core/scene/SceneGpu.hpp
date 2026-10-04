#pragma once

#include <span>
#include <string>
#include <vector>

#include "AreaLightVisual.hpp"
#include "EngineConventions.hpp"
#include "MeshStore.hpp"
#include "Scene.hpp"
#include "SceneObject.hpp"

#include "core/framegraph/ResourceRegistry.hpp"
#include "core/loaders/Material.hpp"
#include "core/loaders/MaterialStore.hpp"
#include "core/loaders/SceneLoader.hpp"
#include "core/passes/geometry/GeometryPass.hpp"
#include "core/upload/CameraUploader.hpp"
#include "core/upload/LightUploader.hpp"
#include "core/upload/MaterialUploader.hpp"
#include "core/upload/MeshUploader.hpp"
#include "core/upload/SkinUploader.hpp"

// SceneGpu turns a Scene (plus the MeshStore/MaterialStore its components refer to) into the GPU
// buffers the engine's passes read, and keeps them in sync as the scene changes:
//   - "camera"               camera UBO (CameraGpuData) from the camera object set with setCamera()
//   - "lights"               light SSBO (LightUploader) from every object with a Light
//   - "meshPositionBuffer"   vertex buffer, binding 0: vec3 positions of every registered mesh
//   - "meshVertexBuffer"     vertex buffer, binding 1: normal/tangent/uv, interleaved (see
//                            conventions::geometryVertexAttributes)
//   - "meshIndexBuffer"      uint32 indices; "meshFaceGroupBuffer" per-face material handle
//   - "material*"            materials SSBO + texture arrays (conventions::materialLayout)
//   - skin buffers           joint palettes etc. for skinned meshes (SkinUploader)
// Every light also gets a quad mesh (see AreaLightVisual.hpp) drawn with the rest of the geometry.
//
// It knows nothing about editing: SceneManager layers the editor's selection/heatmap buffers on top.
namespace lr
{

class Viewer;

class SceneGpu
{
public:
    SceneGpu(ResourceRegistry &registry, Scene &scene, MeshStore &meshStore, MaterialStore &materialStore);

    SceneGpu(const SceneGpu &)            = delete;
    SceneGpu &operator=(const SceneGpu &) = delete;

    Scene         &scene() { return m_scene; }
    MeshStore     &meshStore() { return m_meshStore; }
    MaterialStore &materialStore() { return m_materialStore; }

    // Registers renderable geometry: `object` needs a TransformComponent and a MeshComponent, applied as
    // its model matrix at draw time. Register before initialize(), or call rebuildGeometry() afterwards.
    void addMeshObject(SceneObject &object);

    // Registers every mesh a SceneLoader::load() call created, and queues its material textures for
    // upload by the next rebuildGeometry() (initialize() uploads every texture anyway).
    void addLoaded(const SceneLoadResult &result);

    // Forgets removed objects (meshes and light visuals). Call rebuildGeometry() afterwards.
    void removeSceneObjects(std::span<const SceneObjectId> ids);

    const std::vector<SceneObject *> &meshObjects() const { return m_meshObjects; }

    // The object whose Camera + TransformComponent fill the camera UBO. Need not belong to scene().
    // Must be set before initialize().
    void         setCamera(SceneObject &camera) { m_camera = &camera; }
    SceneObject *camera() const { return m_camera; }

    // Aspect ratio for the camera's projection; read by the next updateCamera().
    void setAspect(float aspect) { m_aspect = aspect; }

    // One-time setup: builds light visuals, then uploads lights, geometry, materials and the camera.
    // Requires at least one registered mesh and a camera.
    void initialize(const AreaLightVisualConfig    &areaLightVisualConfig = conventions::areaLightVisualConfig(),
                    const GpuMaterialLayout        &materialLayout        = conventions::materialLayout(),
                    const std::vector<std::string> &vertexAttributeNames  = conventions::geometryVertexAttributes());
    bool initialized() const { return m_initialized; }

    // Per frame: tracks the swapchain aspect ratio and advances animations (onUpdate), then evaluates
    // skins and calls flushDirty() (onLateUpdate, after everything else has changed the scene).
    void registerCallbacks(Viewer &viewer);

    // Re-uploads whatever the camera, light and mesh components have marked dirty since the last call,
    // and clears their flags.
    void flushDirty();

    // The steps initialize() runs, for callers that sequence setup themselves.
    void createLightVisuals(const AreaLightVisualConfig &config);
    void uploadMeshes(const GpuMaterialLayout &materialLayout, const std::vector<std::string> &vertexAttributeNames);

    // Re-packs all geometry after objects were added or removed. The caller must wait for the GPU
    // first and refresh GeometryPass's draw metadata afterwards (GeometryPass::setSceneGeometry).
    void rebuildGeometry();

    void uploadLights();
    // Re-uploads positions only, for edits that move vertices without changing topology.
    void updatePositions();
    void updateMaterials();
    void updateLightVisuals();
    void updateCamera();
    void updateSkins();
    void updateAnimations(float deltaSeconds);

    // Where `mesh` landed in the shared index buffer. Throws if it isn't registered geometry.
    const IndexBufferUploadPerMeshResult &indexRange(const Mesh &mesh) const;

    // Everything GeometryPass needs to draw this scene, with conventions::materialLayout() texture names.
    // Valid after initialize(); call again after rebuildGeometry().
    GeometryPass::Config geometryPassConfig() const;

    const std::string &cameraBufferName() const { return m_cameraUploader.bufferName(); }
    const std::string &lightBufferName() const { return m_lightUploader.bufferName(); }
    uint32_t           numLights() const { return m_lightUploader.numLights(); }

    const SkinUploadResult &skinUploadResult() const { return m_skinUploadResult; }
    const std::string &skinInfluenceEntriesBufferName() const { return m_skinUploader.influenceEntriesBufferName(); }
    const std::string &skinInfluenceOffsetsBufferName() const { return m_skinUploader.influenceOffsetsBufferName(); }
    const std::string &skinPositionIndicesBufferName() const { return m_skinUploader.positionIndicesBufferName(); }
    const std::string &skinJointMatricesBufferName() const { return m_skinUploader.jointMatricesBufferName(); }

    const std::string &meshPositionBufferName() const { return m_meshPositionBufferName; }
    const std::string &meshVertexBufferName() const { return m_meshVertexBufferName; }
    const std::string &meshIndexBufferName() const { return m_meshIndexBufferName; }
    const std::string &meshFaceGroupBufferName() const { return m_meshFaceGroupBufferName; }

    const VertexBufferUploadResult                &meshPositions() const { return m_meshPositions; }
    const IndexBufferUploadResult                 &indexBuffer() const { return m_indexBuffer; }
    const std::vector<const TransformComponent *> &meshTransforms() const { return m_meshTransforms; }
    const std::vector<SceneObject *>              &geometryObjects() const { return m_geometryObjects; }
    const MaterialUploadResult                    &materialUploadResult() const { return m_materialUploadResult; }

private:
    void gatherGeometry(const std::vector<std::string> &vertexAttributeNames);

    ResourceRegistry &m_registry;
    Scene            &m_scene;
    MeshStore        &m_meshStore;
    MaterialStore    &m_materialStore;

    MeshUploader     m_meshUploader;
    MaterialUploader m_materialUploader;
    LightUploader    m_lightUploader;
    CameraUploader   m_cameraUploader;
    SkinUploader     m_skinUploader;

    SceneObject *m_camera = nullptr;
    // Matches Viewer::Config's default window size until setAspect() is called with the real
    // swapchain extent.
    float                      m_aspect      = 1600.0f / 900.0f;
    bool                       m_initialized = false;
    std::vector<SceneObject *> m_meshObjects;
    std::vector<SceneObject *> m_lightVisualObjects;
    AreaLightVisualConfig      m_areaLightVisualConfig;

    // Cached once in uploadMeshes(), reused by position/light-visual updates so
    // every repack targets the same combined mesh list / buffer configs.
    std::vector<const Mesh *>               m_geometryMeshes;
    std::vector<const TransformComponent *> m_meshTransforms;
    std::vector<Skin *>                     m_meshSkins;
    std::vector<SceneObject *>              m_geometryObjects;
    VertexBufferUploadConfig                m_meshPositionUploadConfig;
    VertexBufferUploadConfig                m_meshAttributeUploadConfig;
    GpuMaterialLayout                       m_materialLayout;
    std::vector<std::string>                m_vertexAttributeNames;
    std::vector<MaterialHandle>             m_pendingTextureUpdates;

    const std::string m_meshPositionBufferName  = "meshPositionBuffer";
    const std::string m_meshVertexBufferName    = "meshVertexBuffer";
    const std::string m_meshIndexBufferName     = "meshIndexBuffer";
    const std::string m_meshFaceGroupBufferName = "meshFaceGroupBuffer";

    VertexBufferUploadResult m_meshPositions;
    IndexBufferUploadResult  m_indexBuffer;
    MaterialUploadResult     m_materialUploadResult;
    SkinUploadResult         m_skinUploadResult;
};

} // namespace lr
