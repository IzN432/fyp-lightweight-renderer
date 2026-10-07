#pragma once

#include <functional>
#include <memory>
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
// SceneGpu owns those quads and their MaterialStore slots: the scene itself is never modified, so it
// can be uploaded again later (e.g. by the next Viewer), and the quads aren't scene meshes anyone can
// select or edit.
//
// It knows nothing about editing: SceneManager layers the editor's selection/heatmap buffers on top.
namespace lr
{

class Viewer;

class SceneGpu
{
public:
    SceneGpu(ResourceRegistry &registry, Scene &scene, MeshStore &meshStore, MaterialStore &materialStore);
    // Returns the light visuals' MaterialStore slots.
    ~SceneGpu();

    SceneGpu(const SceneGpu &)            = delete;
    SceneGpu &operator=(const SceneGpu &) = delete;

    Scene         &scene() { return m_scene; }
    const Scene   &scene() const { return m_scene; }
    MeshStore     &meshStore() { return m_meshStore; }
    const MeshStore &meshStore() const { return m_meshStore; }
    MaterialStore &materialStore() { return m_materialStore; }
    const MaterialStore &materialStore() const { return m_materialStore; }

    // Registers renderable geometry: `object` needs a TransformComponent and a MeshComponent, applied as
    // its model matrix at draw time. Register before initialize(), or call rebuildGeometry() afterwards
    // (which is also how geometry is added back to a scene that was cleared).
    void addMeshObject(SceneObject &object);

    // Registers every mesh a SceneLoader::load() call created, and queues its material textures for
    // upload by the next rebuildGeometry() (initialize() uploads every texture anyway).
    void addLoaded(const SceneLoadResult &result);

    // Marks newly acquired material slots whose texture descriptors must be refreshed on rebuild.
    void queueMaterialTextures(std::span<const MaterialHandle> handles);

    // Drops all authored-scene registrations before their CPU asset stores are cleared.
    void clearSceneResources();

    // Forgets removed mesh objects. Call rebuildGeometry() afterwards. (Removed lights need nothing:
    // flushDirty() notices them, see syncLights().)
    void removeSceneObjects(std::span<const SceneObjectId> ids);

    const std::vector<SceneObject *> &meshObjects() const { return m_meshObjects; }

    // The object whose Camera + TransformComponent fill the camera UBO. Need not belong to scene().
    // Must be set before initialize().
    void         setCamera(SceneObject &camera) { m_camera = &camera; }
    SceneObject *camera() const { return m_camera; }

    // Aspect ratio for the camera's projection. A change re-uploads the camera on the next flushDirty(),
    // even if the camera itself didn't move (e.g. the window was resized).
    void setAspect(float aspect)
    {
        if (m_aspect != aspect)
        {
            m_aspect            = aspect;
            m_cameraAspectDirty = true;
        }
    }

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
    // and clears their flags. Also calls syncLights() if lights were added to or removed from the scene,
    // then synchronizes the shared mesh buffers from the meshes' revisions: mesh edits (positions,
    // attributes, face groups, same-size topology) never upload directly, and several edits in a frame
    // collapse into one upload per stale buffer (see docs/mesh-synchronization.md).
    void flushDirty();

    // Brings the light visuals and the light buffer in line with the lights now in the scene: drops the
    // quads of removed lights, builds quads for new ones, rebuilds the geometry and re-uploads the lights.
    // Safe between frames (replaced buffers are retired, not destroyed); listeners are notified.
    void syncLights();

    // Called after rebuildGeometry() (draw lists changed: refresh GeometryPass::setSceneGeometry) and
    // after uploadLights() (with the new light count: PbrPass::setNumLights). Listeners must stay valid
    // until removed, or for the SceneGpu's lifetime.
    using ListenerId = uint64_t;
    ListenerId onGeometryRebuilt(std::function<void(const SceneGpu &)> listener);
    ListenerId onLightsUploaded(std::function<void(uint32_t numLights)> listener);
    void       removeListener(ListenerId id);

    // The steps initialize() runs, for callers that sequence setup themselves.
    void createLightVisuals(const AreaLightVisualConfig &config);
    void uploadMeshes(const GpuMaterialLayout &materialLayout, const std::vector<std::string> &vertexAttributeNames);

    // Re-packs all geometry after objects were added or removed, then notifies onGeometryRebuilt
    // listeners, which refresh GeometryPass's draw metadata (GeometryPass::setSceneGeometry).
    // A scene left with nothing to draw is supported: the draw lists go empty and the shared buffers
    // are left as they are, so clearing the scene and then importing or adding objects back works.
    void rebuildGeometry();

    void uploadLights();
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
    uint32_t           maxLights() const { return m_lightUploader.maxLights(); }

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
    const std::vector<const Mesh *>               &geometryMeshes() const { return m_geometryMeshes; }
    const MaterialUploadResult                    &materialUploadResult() const { return m_materialUploadResult; }

private:
    // A light's quad (see AreaLightVisual.hpp). The mesh is heap-allocated so the pointers in
    // m_geometryMeshes stay valid as visuals are added or removed.
    struct LightVisual
    {
        SceneObject          *light = nullptr;
        std::unique_ptr<Mesh> mesh;
        MaterialHandle        material;
    };

    void gatherGeometry(const std::vector<std::string> &vertexAttributeNames);
    // Whether the geometry gathered by gatherGeometry() would pack into non-empty vertex and index
    // buffers, i.e. whether there is anything for GeometryPass to draw.
    bool hasDrawableGeometry() const;
    // Empties the draw lists without touching the GPU buffers (see rebuildGeometry()).
    void dropGeometry();
    void releaseLightVisuals();
    // Uploads whichever shared mesh buffers' source revisions changed (see flushDirty()).
    void synchronizeMeshes();
    // Whether `object` is a light still in the scene.
    bool isLiveLight(const SceneObject &object) const;
    // True if lights were added to or removed from the scene since the visuals were built.
    bool        lightsChanged() const;
    LightVisual makeLightVisual(SceneObject &light);

    struct Listener
    {
        ListenerId                              id;
        std::function<void(const SceneGpu &)>   geometryRebuilt;
        std::function<void(uint32_t numLights)> lightsUploaded;
    };
    std::vector<Listener> m_listeners;
    ListenerId            m_nextListenerId = 1;

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
    bool                       m_cameraAspectDirty = false;
    bool                       m_initialized = false;
    std::vector<SceneObject *> m_meshObjects;
    std::vector<LightVisual>   m_lightVisuals;
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
