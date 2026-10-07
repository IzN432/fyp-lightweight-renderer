#pragma once

#include <functional>
#include <memory>
#include <string>
#include <span>
#include <vector>

#include "SceneObject.hpp"
#include "Scene.hpp"
#include "SceneGpu.hpp"
#include "AreaLightVisual.hpp"
#include "Mesh.hpp"
#include "MeshStore.hpp"
#include "TransformComponent.hpp"

#include "core/editor/selection/SelectionManager.hpp"
#include "core/editor/EditorPresentationState.hpp"
#include "core/framegraph/ResourceRegistry.hpp"
#include "core/loaders/Material.hpp"
#include "core/loaders/MaterialStore.hpp"
#include "core/loaders/SceneLoader.hpp"
#include "core/upload/MeshUploader.hpp"

// SceneManager owns the MeshStore/MaterialStore for a Scene and is the editor's view of it. Keeping the
// GPU-facing scene buffers (meshes, materials, lights, camera, skins) in sync is SceneGpu's job — see
// gpu() — and SceneManager adds the editor's state on top: the edited mesh, vertex selection, editor
// presentation capabilities, and the selected-mesh points/heatmap buffers editor overlays draw.
namespace lr
{

class Viewer;

class SceneManager
{
public:
    SceneManager(ResourceRegistry &registry, uint32_t materialCapacity,
                 std::function<Material()> defaultMaterialFactory);

    // Must be called once, before anything that touches the GPU side (it creates gpu()).
    void   setScene(Scene &scene);
    Scene &scene() { return gpu().scene(); }

    // The scene's GPU buffers. Valid after setScene().
    SceneGpu       &gpu();
    const SceneGpu &gpu() const;

    MaterialStore &materialStore() { return m_materialStore; }
    MeshStore     &meshStore() { return m_meshStore; }

    // Loads OBJ, glTF, or GLB content into this manager's Scene and asset
    // stores. Returns an identity-transform container for the imported asset.
    SceneObject &load(const std::filesystem::path &path, const SceneLoaderConfig &config = {});

    // Appends a native .lrscene and registers its renderable objects. Returns the IDs added.
    std::vector<SceneObjectId> loadScene(const std::filesystem::path &path);

    // Saves the complete authored scene and its referenced assets as a single .lrscene file.
    void save(const std::filesystem::path &path) const;

    // Registers renderable scene geometry (see SceneGpu::addMeshObject). The first registered object
    // becomes the initial edited mesh; load() registers imported meshes automatically. Later
    // position/attribute edits are detected from the mesh's revisions.
    void         addMeshObject(SceneObject &object);
    SceneObject *removeSceneObjects(std::span<const SceneObjectId> ids);

    // The object currently targeted by vertex editing (SelectionManager, the vertex-picking
    // overlay's points buffer and heatmap analysis). The caller drives this from Scene Hierarchy
    // selection. Defaults to the first registered mesh.
    SceneObject *editedMeshObject() { return m_editedMeshObject; }

    // The Mesh and TransformComponent the overlay buffers, SelectionManager and the editor's
    // VertexManager bind to. With no edited mesh object these are an empty placeholder mesh and a
    // detached transform owned by this manager, so those bindings are always valid and non-optional
    // — an application does not have to load an asset just to construct its editor.
    Mesh                     &editedMesh();
    const TransformComponent &editedMeshTransform() const;

    // True if `object` has what setEditedMeshObject() requires (a MeshComponent + TransformComponent).
    static bool isEditable(const SceneObject &object);

    // Repoints vertex editing at `object` (see editedMeshObject()) — rebinds the SelectionManager
    // and replaces the vertex-picking overlay's points buffer for the new mesh's vertex count,
    // which may differ arbitrarily from the previous mesh's. Resets selection, highlighting, and
    // per-vertex role classification (SelectionManager::rebind's doc comment). The caller is still
    // responsible for rebinding anything it owns directly against the old mesh (VertexManager's
    // rebind(), each EditorTool's onTargetChanged()) and for ensuring the GPU is done with the
    // previous points
    // buffer first (see ResourceRegistry::replaceUploadedBuffer's doc comment) — e.g.
    // viewer.context().waitIdle() before calling this, the way the HDRI-reload path does.
    // Also replaces the selected heatmap buffer. Throws if `object` isn't isEditable(). No-op if
    // `object` is already selected.
    void setEditedMeshObject(SceneObject &object);

    // The camera whose Camera/TransformComponent state drives the camera UBO. Must be set before
    // initialize().
    void setDefaultCamera(SceneObject &camera) { gpu().setCamera(camera); }

    // Updated once per frame from the current swapchain extent (window resize) — read by
    // updateCamera() the next time it runs, so there's no need to force a re-upload here.
    void setAspect(float aspect) { gpu().setAspect(aspect); }

    // Performs all one-time scene setup that would otherwise have to be manually sequenced by the
    // caller: SceneGpu::initialize() (light visuals; lights/mesh/material/camera buffers), the
    // selected mesh's points/heatmap buffers, and the SelectionManager that operates on the initially
    // selected mesh. Requires a scene and a default camera; registered geometry is optional (see
    // editedMesh()).
    void initialize(const AreaLightVisualConfig &areaLightVisualConfig, const GpuMaterialLayout &materialLayout,
                    const std::vector<std::string> &vertexAttributeNames, InputHandler &input);

    // Registers SceneGpu's per-frame callbacks (aspect ratio, animations, skins, flushDirty()), an
    // onUpdate that drives the SelectionManager's mouse/drag handling, and an onLateUpdate that
    // synchronizes the selected-mesh overlay buffers (see synchronizeSelectedMeshBuffers()).
    void registerCallbacks(Viewer &viewer);

    // Selection over the selected mesh's deduped-position space — constructed by initialize(), so only
    // valid to call after it. Highlight changes are wired internally to store the highlighted-vertex
    // colors in the mesh's CPU data; late synchronization updates the GPU. The caller still owns wiring up
    // a SelectionTool
    // (setSelectTool), the mouse click handoff with gizmos, and reading getSelectedIndices()/
    // getHighlightedIndices() for its own UI (translate gizmo placement, etc.).
    SelectionManager &selectionManager() { return *m_selectionManager; }

    const EditorPresentation &editorPresentation() const { return m_editorPresentation.current(); }

    // Applies capabilities selected by the application-level editor state. SceneManager does not
    // know state names; it only wires this manager's selection to the transition policy that
    // EditorPresentationState owns, and forwards the flags to its own overlay synchronization.
    void setEditorPresentation(EditorPresentation presentation) { m_editorPresentation.set(presentation); }
    CallbackConnection
    registerEditorPresentationChangedCallback(std::function<void(const EditorPresentation &)> callback)
    {
        return m_editorPresentation.registerChangedCallback(std::move(callback));
    }

    // SceneGpu::flushDirty() (component edits and the shared mesh buffers), then the selected-mesh
    // overlays. Mesh edits never upload directly; call once per frame, after every other update.
    void flushDirty();

    // GPU sync, forwarded to gpu() — see SceneGpu.
    void createLightVisuals(const AreaLightVisualConfig &config) { gpu().createLightVisuals(config); }
    // Re-packs all render geometry after runtime imports/deletions. Also replaces the selected-mesh
    // overlays, and rebinds the selection if the selected mesh's topology changed (the caller rebinds
    // its own tools). Passes registered with SceneGpu::onGeometryRebuilt are refreshed.
    void rebuildGeometry() { gpu().rebuildGeometry(); }
    void uploadLights() { gpu().uploadLights(); }
    void updateMaterials() { gpu().updateMaterials(); }
    void updateLightVisuals() { gpu().updateLightVisuals(); }
    void updateCamera() { gpu().updateCamera(); }
    void updateSkins() { gpu().updateSkins(); }
    void updateAnimations(float deltaSeconds) { gpu().updateAnimations(deltaSeconds); }

    // SceneGpu::uploadMeshes plus the selected mesh's points/heatmap buffers.
    void uploadMeshes(const GpuMaterialLayout &materialLayout, const std::vector<std::string> &vertexAttributeNames);

    // Replaces the analysis colors used by HeatmapPass without touching the selection-highlight
    // colors used by the points overlay. Colors are indexed by mesh.positions().
    void setSelectedMeshHeatmapColors(std::span<const glm::vec3> colors);

    // Vulkan vertex inputs matching the two overlay buffers above. Declared here, next to the
    // upload configs that define their packing, so a pass can be built before any mesh exists
    // rather than deriving the layout from whichever mesh happens to be loaded.
    static GpuMeshLayout selectedMeshPointsLayout();
    static GpuMeshLayout selectedMeshHeatmapLayout();

    const std::string &cameraBufferName() const { return gpu().cameraBufferName(); }

    const SkinUploadResult &skinUploadResult() const { return gpu().skinUploadResult(); }
    const std::string      &skinInfluenceEntriesBufferName() const { return gpu().skinInfluenceEntriesBufferName(); }
    const std::string      &skinInfluenceOffsetsBufferName() const { return gpu().skinInfluenceOffsetsBufferName(); }
    const std::string      &skinPositionIndicesBufferName() const { return gpu().skinPositionIndicesBufferName(); }
    const std::string      &skinJointMatricesBufferName() const { return gpu().skinJointMatricesBufferName(); }

    const std::string &meshPositionBufferName() const { return gpu().meshPositionBufferName(); }
    const std::string &meshVertexBufferName() const { return gpu().meshVertexBufferName(); }
    const std::string &meshIndexBufferName() const { return gpu().meshIndexBufferName(); }
    const std::string &meshFaceGroupBufferName() const { return gpu().meshFaceGroupBufferName(); }
    const std::string &selectedMeshPointsBufferName() const { return m_selectedMeshPointsBufferName; }
    // Interleaved position + color buffer, duped per UV-seam corner (unlike the selected points
    // buffer, which is deduped) — for HeatmapPass, which needs the
    // color Gouraud-interpolated across the same triangles GeometryPass draws, so it must share
    // GeometryPass's corner-indexed topology rather than the
    // deduped-position space the points overlay uses. The colors themselves stay per unique
    // position; packing gathers them through positionIndices.
    const std::string &selectedMeshHeatmapBufferName() const { return m_selectedMeshHeatmapBufferName; }

    const VertexBufferUploadResult &meshPositions() const { return gpu().meshPositions(); }
    // Selected mesh's unique/deduped position+color buffer.
    const VertexBufferUploadResult &selectedMeshPoints() const { return m_selectedMeshPoints; }
    // Selected mesh's corner-domain position+color buffer. Only
    // ever holds one mesh (singleMeshResults[0]), unlike meshPositions()/indexBuffer().
    const VertexBufferUploadResult                &selectedMeshHeatmap() const { return m_selectedMeshHeatmap; }
    const IndexBufferUploadPerMeshResult          &selectedMeshIndexRange() const;
    const IndexBufferUploadResult                 &indexBuffer() const { return gpu().indexBuffer(); }
    const std::vector<const TransformComponent *> &meshTransforms() const { return gpu().meshTransforms(); }
    const std::vector<SceneObject *>              &geometryObjects() const { return gpu().geometryObjects(); }

    const MaterialUploadResult &materialUploadResult() const { return gpu().materialUploadResult(); }

    const std::string &lightBufferName() const { return gpu().lightBufferName(); }
    uint32_t           numLights() const { return gpu().numLights(); }

private:
    // Uploads the selected mesh's points/heatmap buffers for the first time.
    void uploadSelectedMeshBuffers();

    // Rebuilds the selected mesh's "color" per-unique-vertex attribute from the SelectionManager's
    // current highlighted indices. Wired as m_selectionManager's highlight-changed callback in
    // initialize().
    void updateSelectedMeshHighlightColors();

    // Uploads whichever overlays the current editor presentation requests. Inactive overlays keep
    // their own stamps and catch up when next active.
    void synchronizeSelectedMeshBuffers();

    // After SceneGpu re-packs geometry: replaces the overlays, and rebinds the selection if the
    // selected mesh's topology changed (importing unrelated assets keeps the current selection).
    void onGeometryRebuilt();

    void ensureSelectedMeshAttributes(Mesh &mesh);

    ResourceRegistry &m_registry;
    MeshUploader      m_meshUploader;
    MaterialStore     m_materialStore;
    MeshStore         m_meshStore;
    // Declared after the stores it refers to, so it is destroyed first.
    std::unique_ptr<SceneGpu> m_gpu;

    std::unique_ptr<SelectionManager> m_selectionManager;
    // Selection is created lazily by initialize(), so the withdrawal hook tolerates not having one
    // yet; a presentation cannot withdraw vertex selection before something granted it.
    EditorPresentationState m_editorPresentation{[this] {
        if (m_selectionManager)
        {
            m_selectionManager->clearSelection();
        }
    }};

    SceneObject   *m_editedMeshObject         = nullptr;
    Mesh::Revision m_selectedTopologyRevision = 0;

    // Null object for "nothing is being edited": an empty mesh (topology established, so the
    // overlay attributes can be seeded onto it) and a transform with no owning SceneObject. Every
    // consumer that needs a Mesh&/TransformComponent& binds to these while m_editedMeshObject is
    // null, which keeps the overlay buffers registered — passes resolve them at build time — and
    // keeps SelectionManager/VertexManager pointer-stable. See editedMesh().
    Mesh               m_unboundMesh;
    TransformComponent m_unboundMeshTransform;

    const std::string m_selectedMeshPointsBufferName  = "meshPointsBuffer";
    const std::string m_selectedMeshHeatmapBufferName = "meshHeatmapBuffer";

    // Config for the deduped position+color buffer above. Its stamp observes both position and
    // highlight revisions.
    VertexBufferUploadConfig m_selectedMeshPointsUploadConfig = {
        .vertexBufferName = m_selectedMeshPointsBufferName, .vertexAttributeNames = {"color"}, .includePosition = true};

    // Config for the corner-domain position+heatmapColors buffer; the unique-position colors are
    // gathered through positionIndices while packing.
    VertexBufferUploadConfig m_selectedMeshHeatmapUploadConfig = {.vertexBufferName = m_selectedMeshHeatmapBufferName,
                                                                  .vertexAttributeNames         = {"heatmapColors"},
                                                                  .includePosition              = true,
                                                                  .expandUniqueVertexAttributes = true};

    VertexBufferUploadResult m_selectedMeshPoints;
    VertexBufferUploadResult m_selectedMeshHeatmap;
    // Declared last so SceneGpu/Viewer callbacks disconnect before this manager's state is destroyed.
    std::vector<CallbackConnection> m_connections;
};

} // namespace lr
