#pragma once

#include <functional>
#include <string>
#include <vector>

#include "SceneObject.hpp"
#include "Scene.hpp"
#include "AreaLightVisual.hpp"

#include "core/framegraph/ResourceRegistry.hpp"
#include "core/loaders/Material.hpp"
#include "core/loaders/MaterialStore.hpp"
#include "core/upload/CameraUploader.hpp"
#include "core/upload/LightUploader.hpp"
#include "core/upload/MaterialUploader.hpp"
#include "core/upload/MeshUploader.hpp"

// SceneManager holds all the scene objects in the scene, owns the MaterialStore, and is
// responsible for packing/uploading the GPU-facing buffers (mesh vertex/index/facegroup buffers,
// the materials SSBO + texture arrays, the lights buffer) and keeping them in sync as the scene
// changes.
namespace lr
{

class Viewer;

class SceneManager
{
public:
    SceneManager(ResourceRegistry &registry, uint32_t materialCapacity,
                 std::function<Material()> defaultMaterialFactory);

    void setScene(Scene& scene) { m_scene = &scene; }
    Scene& scene() { return *m_scene; }

    MaterialStore& materialStore() { return m_materialStore; }

    // The scene's single non-light-visual mesh — its Transform is applied via the model matrix at
    // draw time (unlike light visuals, which bake their Transform into vertex positions directly).
    // Must be set before initialize().
    void setMainMeshObject(SceneObject &object) { m_mainMeshObject = &object; }

    // The camera whose Camera/Transform state drives the camera UBO. Must be set before
    // initialize().
    void setDefaultCamera(SceneObject &camera) { m_defaultCamera = &camera; }

    // Updated once per frame from the current swapchain extent (window resize) — read by
    // updateCamera() the next time it runs, so there's no need to force a re-upload here.
    void setAspect(float aspect) { m_aspect = aspect; }

    // Performs all one-time scene setup that would otherwise have to be manually sequenced by the
    // caller: builds light visuals and uploads the initial lights/mesh/material/camera buffers.
    // Requires setScene(), setMainMeshObject() and setDefaultCamera() to have been called first.
    void initialize(const AreaLightVisualConfig &areaLightVisualConfig,
                     const GpuMaterialLayout &materialLayout,
                     const std::vector<std::string> &vertexAttributeNames);

    // Registers the per-frame callbacks SceneManager needs — an onUpdate that tracks the
    // swapchain aspect ratio (see setAspect) and an onLateUpdate that calls flushDirty() — so the
    // caller doesn't need to know what SceneManager wires up each frame.
    void registerCallbacks(Viewer &viewer);

    // Polls the camera, light visual, and main mesh components for the dirty flag their setters/
    // onGUIImpl set via Component::markDirty(), re-uploads whatever's dirty (at most once per
    // resource, however many times it changed this frame), and clears the flags. Call once per
    // frame, after every other update callback has had a chance to mutate the scene — see
    // Viewer::onLateUpdate.
    void flushDirty();

    // Builds one hidden quad StaticMesh (and one MaterialStore slot) per Light currently in the
    // scene — see AreaLightVisual.hpp for why every light gets one regardless of its current type.
    // Called by initialize(); exposed separately in case a caller needs to set up light visuals
    // without going through the full initialize() sequence.
    void createLightVisuals(const AreaLightVisualConfig &config);

    // Packs the main mesh + every light visual into the shared vertex/index/facegroup buffers and
    // uploads them, plus the initial materials SSBO/texture arrays snapshot.
    void uploadMeshes(const GpuMaterialLayout &materialLayout, const std::vector<std::string> &vertexAttributeNames);

    void uploadLights();

    // Repacks just the position buffer — for edits that only move vertices (vertex-drag editing).
    void updateMainMeshPositions();

    // Re-uploads the materials SSBO from the MaterialStore's current contents — called by
    // flushDirty() when the main mesh's StaticMesh is dirty (e.g. a Scene Hierarchy slider edit,
    // see StaticMesh::onGUIImpl), so the edit reaches the GPU.
    void updateMaterials();

    // Rebuilds every light visual's quad geometry + MaterialStore slot from its current Light/
    // Transform state, then re-uploads (positions, attributes, materials). Called by flushDirty()
    // when any light-visual object's Light or Transform is dirty.
    void updateLightVisuals();

    // Re-uploads the camera UBO from the default camera's current Camera/Transform state and the
    // last aspect ratio set via setAspect(). Called once during initialize(), and by flushDirty()
    // when the default camera's Camera or Transform is dirty.
    void updateCamera();

    const std::string &cameraBufferName() const { return m_cameraUploader.bufferName(); }

    const std::string &mainMeshPositionBufferName()  const { return m_mainMeshPositionBufferName; }
    const std::string &mainMeshVertexBufferName()    const { return m_mainMeshVertexBufferName; }
    const std::string &mainMeshColorBufferName()     const { return m_mainMeshColorBufferName; }
    const std::string &mainMeshIndexBufferName()     const { return m_mainMeshIndexBufferName; }
    const std::string &mainMeshFaceGroupBufferName() const { return m_mainMeshFaceGroupBufferName; }

    const VertexBufferUploadResult &meshPositions() const { return m_meshPositions; }
    const IndexBufferUploadResult  &indexBuffer()   const { return m_indexBuffer; }
    const std::vector<const Transform*> &meshTransforms() const { return m_meshTransforms; }

    const MaterialUploadResult &materialUploadResult() const { return m_materialUploadResult; }

    const std::string &lightBufferName() const { return m_lightUploader.bufferName(); }
    uint32_t numLights() const { return m_lightUploader.numLights(); }

private:
    void gatherGeometry(const std::vector<std::string> &vertexAttributeNames);

    Scene* m_scene = nullptr;
    ResourceRegistry &m_registry;

    MeshUploader m_meshUploader;
    MaterialUploader m_materialUploader;
    LightUploader m_lightUploader;
    CameraUploader m_cameraUploader;
    MaterialStore m_materialStore;

    SceneObject* m_mainMeshObject = nullptr;
    SceneObject* m_defaultCamera = nullptr;
    // Matches Viewer::Config's default window size until setAspect() is called with the real
    // swapchain extent.
    float m_aspect = 1600.0f / 900.0f;
    std::vector<SceneObject*> m_lightVisualObjects;
    AreaLightVisualConfig m_areaLightVisualConfig;

    // Cached once in uploadMeshes(), reused by updateMainMeshPositions()/updateLightVisuals() so
    // every repack targets the same combined mesh list / buffer configs.
    std::vector<const Mesh*> m_geometryMeshes;
    std::vector<const Transform*> m_meshTransforms;
    VertexBufferUploadConfig m_meshPositionUploadConfig;
    VertexBufferUploadConfig m_meshAttributeUploadConfig;
    GpuMaterialLayout m_materialLayout;

    const std::string m_mainMeshPositionBufferName  = "meshPositionBuffer";
    const std::string m_mainMeshVertexBufferName    = "meshVertexBuffer";
    const std::string m_mainMeshColorBufferName      = "meshColorBuffer";
    const std::string m_mainMeshIndexBufferName     = "meshIndexBuffer";
    const std::string m_mainMeshFaceGroupBufferName = "meshFaceGroupBuffer";

    VertexBufferUploadResult m_meshPositions;
    IndexBufferUploadResult  m_indexBuffer;
    MaterialUploadResult     m_materialUploadResult;
};

}
