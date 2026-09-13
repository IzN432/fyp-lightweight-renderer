#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "SceneObject.hpp"
#include "Scene.hpp"
#include "AreaLightVisual.hpp"
#include "MeshStore.hpp"

#include "core/editor/selection/SelectionManager.hpp"
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

// The editor's current interaction mode over the main mesh — View just renders normally, Edit
// shows the vertex-picking overlay and lets the SelectionManager's SelectionTool receive clicks.
// Only two states for now; may grow (e.g. per-tool edit modes) later.
enum class SelectionState
{
    View,
    Edit,
};

class SceneManager
{
public:
    SceneManager(ResourceRegistry &registry, uint32_t materialCapacity,
                 std::function<Material()> defaultMaterialFactory);

    void   setScene(Scene &scene) { m_scene = &scene; }
    Scene &scene() { return *m_scene; }

    MaterialStore &materialStore() { return m_materialStore; }
    MeshStore     &meshStore() { return m_meshStore; }

    // The scene's single non-light-visual mesh — its TransformComponent is applied via the model matrix at
    // draw time (unlike light visuals, which bake their TransformComponent into vertex positions directly).
    // Must be set before initialize().
    void setMainMeshObject(SceneObject &object) { m_mainMeshObject = &object; }

    // The camera whose Camera/TransformComponent state drives the camera UBO. Must be set before
    // initialize().
    void setDefaultCamera(SceneObject &camera) { m_defaultCamera = &camera; }

    // Updated once per frame from the current swapchain extent (window resize) — read by
    // updateCamera() the next time it runs, so there's no need to force a re-upload here.
    void setAspect(float aspect) { m_aspect = aspect; }

    // Performs all one-time scene setup that would otherwise have to be manually sequenced by the
    // caller: builds light visuals, uploads the initial lights/mesh/material/camera buffers, and
    // constructs the SelectionManager that operates on the main mesh (see selectionManager()) —
    // input is needed for that. Requires setScene(), setMainMeshObject() and setDefaultCamera() to
    // have been called first.
    void initialize(const AreaLightVisualConfig &areaLightVisualConfig, const GpuMaterialLayout &materialLayout,
                    const std::vector<std::string> &vertexAttributeNames, InputHandler &input);

    // Registers the per-frame callbacks SceneManager needs — an onUpdate that tracks the
    // swapchain aspect ratio (see setAspect), an onUpdate that drives the SelectionManager's mouse/
    // drag handling, and an onLateUpdate that calls flushDirty() — so the caller doesn't need to
    // know what SceneManager wires up each frame.
    void registerCallbacks(Viewer &viewer);

    // Selection over the main mesh's deduped-position space — constructed by initialize(), so only
    // valid to call after it. Highlight changes are wired internally to push the highlighted-vertex
    // colors to the GPU (see updateMainMeshPointsBuffer()); the caller still owns wiring up a SelectionTool
    // (setSelectTool), the mouse click handoff with gizmos, and reading getSelectedIndices()/
    // getHighlightedIndices() for its own UI (translate gizmo placement, etc.).
    SelectionManager &selectionManager() { return *m_selectionManager; }

    SelectionState selectionState() const { return m_selectionState; }

    // Switches the editor's interaction mode. Leaving Edit clears the current selection (mirrors
    // the old Tab-toggle behavior). The caller is still responsible for toggling the points-overlay
    // pass's own visibility to match (see OverlayPointsPass::setEnabled) — SceneManager doesn't own
    // any render passes.
    void setSelectionState(SelectionState state);

    // Polls the camera, light visual, and main mesh components for the dirty flag their setters/
    // onGUIImpl set via Component::markDirty(), re-uploads whatever's dirty (at most once per
    // resource, however many times it changed this frame), and clears the flags. Call once per
    // frame, after every other update callback has had a chance to mutate the scene — see
    // Viewer::onLateUpdate.
    void flushDirty();

    // Builds one hidden quad MeshComponent (and one MaterialStore slot) per Light currently in the
    // scene — see AreaLightVisual.hpp for why every light gets one regardless of its current type.
    // Called by initialize(); exposed separately in case a caller needs to set up light visuals
    // without going through the full initialize() sequence.
    void createLightVisuals(const AreaLightVisualConfig &config);

    // Packs the main mesh + every light visual into the shared vertex/index/facegroup buffers and
    // uploads them, plus the initial materials SSBO/texture arrays snapshot.
    void uploadMeshes(const GpuMaterialLayout &materialLayout, const std::vector<std::string> &vertexAttributeNames);

    void uploadLights();

    // Repacks the GBuffer position buffer plus the deduped position+color buffer (see
    // updateMainMeshPointsBuffer()) — for edits that only move vertices (vertex-drag editing).
    void updateMainMeshPositions();

    // Replaces the analysis colors used by HeatmapPass without touching the selection-highlight
    // colors used by the points overlay. Colors are indexed by mesh.positions().
    void setMainMeshHeatmapColors(std::span<const glm::vec3> colors);

    // Re-uploads the materials SSBO from the MaterialStore's current contents — called by
    // flushDirty() when the main mesh's MeshComponent is dirty (e.g. a Scene Hierarchy slider edit,
    // see MeshComponent::onGUIImpl), so the edit reaches the GPU.
    void updateMaterials();

    // Rebuilds every light visual's quad geometry + MaterialStore slot from its current Light/
    // TransformComponent state, then re-uploads (positions, attributes, materials). Called by flushDirty()
    // when any light-visual object's Light or TransformComponent is dirty.
    void updateLightVisuals();

    // Re-uploads the camera UBO from the default camera's current Camera/TransformComponent state and the
    // last aspect ratio set via setAspect(). Called once during initialize(), and by flushDirty()
    // when the default camera's Camera or TransformComponent is dirty.
    void updateCamera();

    const std::string &cameraBufferName() const { return m_cameraUploader.bufferName(); }

    const std::string &mainMeshPositionBufferName() const { return m_mainMeshPositionBufferName; }
    // Interleaved unique/deduped position + color buffer — see m_mainMeshPointsBufferName. Distinct
    // from mainMeshPositionBufferName(), which is duped per UV-seam corner for GeometryPass.
    const std::string &mainMeshPointsBufferName() const { return m_mainMeshPointsBufferName; }
    const std::string &mainMeshVertexBufferName() const { return m_mainMeshVertexBufferName; }
    const std::string &mainMeshIndexBufferName() const { return m_mainMeshIndexBufferName; }
    const std::string &mainMeshFaceGroupBufferName() const { return m_mainMeshFaceGroupBufferName; }
    // Interleaved position + color buffer, duped per UV-seam corner like mainMeshVertexBufferName()
    // (unlike mainMeshPointsBufferName(), which is deduped) — for HeatmapPass, which needs the
    // color Gouraud-interpolated across the same triangles GeometryPass draws, so it must share
    // GeometryPass's corner-indexed topology (see mainMeshIndexBufferName()) rather than the
    // deduped-position space the points overlay uses.
    const std::string &mainMeshHeatmapBufferName() const { return m_mainMeshHeatmapBufferName; }

    const VertexBufferUploadResult &meshPositions() const { return m_meshPositions; }
    // Main mesh's unique/deduped position+color buffer — see mainMeshPointsBufferName().
    const VertexBufferUploadResult &mainMeshPoints() const { return m_mainMeshPoints; }
    // Main mesh's corner-domain position+color buffer — see mainMeshHeatmapBufferName(). Only
    // ever holds one mesh (singleMeshResults[0]), unlike meshPositions()/indexBuffer().
    const VertexBufferUploadResult       &mainMeshHeatmap() const { return m_mainMeshHeatmap; }
    const IndexBufferUploadResult        &indexBuffer() const { return m_indexBuffer; }
    const std::vector<const TransformComponent *> &meshTransforms() const { return m_meshTransforms; }

    const MaterialUploadResult &materialUploadResult() const { return m_materialUploadResult; }

    const std::string &lightBufferName() const { return m_lightUploader.bufferName(); }
    uint32_t           numLights() const { return m_lightUploader.numLights(); }

private:
    void gatherGeometry(const std::vector<std::string> &vertexAttributeNames);

    // Rebuilds the main mesh's "color" per-unique-vertex attribute from the SelectionManager's
    // current highlighted indices and pushes it to the GPU. Wired as m_selectionManager's
    // highlight-changed callback in initialize().
    void updateMainMeshHighlightColors();

    // Repacks and re-uploads the deduped position+color buffer (m_mainMeshPointsBufferName) from
    // the main mesh's current positions/"color" attribute. Shared by updateMainMeshPositions() (a
    // vertex moved) and updateMainMeshHighlightColors() (a color changed) since both fields live in
    // the same interleaved buffer.
    void updateMainMeshPointsBuffer();

    // Expands the main mesh's per-unique-vertex "heatmapColors" attribute to the corner domain
    // (via positionIndices) and registers the matching per-vertex attribute on the same Mesh, ready
    // for packing into m_mainMeshHeatmapBufferName. Returns the mutable mesh reference so callers
    // can pack/upload it (initial upload vs. re-upload need different MeshUploader calls).
    Mesh &syncMainMeshCornerHeatmapColors();

    // Repacks and re-uploads m_mainMeshHeatmapBufferName. Called when positions or independently
    // stored analysis colors change.
    void updateMainMeshHeatmapBuffer();

    Scene            *m_scene = nullptr;
    ResourceRegistry &m_registry;

    std::unique_ptr<SelectionManager> m_selectionManager;
    SelectionState                    m_selectionState = SelectionState::View;

    MeshUploader     m_meshUploader;
    MaterialUploader m_materialUploader;
    LightUploader    m_lightUploader;
    CameraUploader   m_cameraUploader;
    MaterialStore    m_materialStore;
    MeshStore        m_meshStore;

    SceneObject *m_mainMeshObject = nullptr;
    SceneObject *m_defaultCamera  = nullptr;
    // Matches Viewer::Config's default window size until setAspect() is called with the real
    // swapchain extent.
    float                      m_aspect = 1600.0f / 900.0f;
    std::vector<SceneObject *> m_lightVisualObjects;
    AreaLightVisualConfig      m_areaLightVisualConfig;

    // Cached once in uploadMeshes(), reused by updateMainMeshPositions()/updateLightVisuals() so
    // every repack targets the same combined mesh list / buffer configs.
    std::vector<const Mesh *>      m_geometryMeshes;
    std::vector<const TransformComponent *> m_meshTransforms;
    VertexBufferUploadConfig       m_meshPositionUploadConfig;
    VertexBufferUploadConfig       m_meshAttributeUploadConfig;
    GpuMaterialLayout              m_materialLayout;

    const std::string m_mainMeshPositionBufferName  = "meshPositionBuffer";
    const std::string m_mainMeshPointsBufferName    = "meshPointsBuffer";
    const std::string m_mainMeshVertexBufferName    = "meshVertexBuffer";
    const std::string m_mainMeshIndexBufferName     = "meshIndexBuffer";
    const std::string m_mainMeshFaceGroupBufferName = "meshFaceGroupBuffer";
    const std::string m_mainMeshHeatmapBufferName   = "meshHeatmapBuffer";

    // Config for the deduped position+color buffer above — same shape as m_meshPositionUploadConfig/
    // m_meshAttributeUploadConfig, cached so updateMainMeshPositions()/updateMainMeshHighlightColors()
    // both repack it identically.
    VertexBufferUploadConfig m_mainMeshPointsUploadConfig = {
        .vertexBufferName = m_mainMeshPointsBufferName, .vertexAttributeNames = {"color"}, .includePosition = true};

    // Config for the corner-domain position+heatmapColors buffer.
    VertexBufferUploadConfig m_mainMeshHeatmapUploadConfig = {
        .vertexBufferName = m_mainMeshHeatmapBufferName,
        .vertexAttributeNames = {"heatmapColors"},
        .includePosition = true};

    VertexBufferUploadResult m_meshPositions;
    VertexBufferUploadResult m_mainMeshPoints;
    VertexBufferUploadResult m_mainMeshHeatmap;
    IndexBufferUploadResult  m_indexBuffer;
    MaterialUploadResult     m_materialUploadResult;
};

} // namespace lr
