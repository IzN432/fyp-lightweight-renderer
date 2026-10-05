#pragma once

#include <functional>
#include <memory>
#include <string>
#include <span>
#include <vector>

#include "SceneObject.hpp"
#include "Scene.hpp"
#include "AreaLightVisual.hpp"
#include "MeshStore.hpp"

#include "core/editor/selection/SelectionManager.hpp"
#include "core/framegraph/ResourceRegistry.hpp"
#include "core/loaders/Material.hpp"
#include "core/loaders/MaterialStore.hpp"
#include "core/loaders/SceneLoader.hpp"
#include "core/upload/CameraUploader.hpp"
#include "core/upload/LightUploader.hpp"
#include "core/upload/MaterialUploader.hpp"
#include "core/upload/MeshUploader.hpp"
#include "core/upload/SkinUploader.hpp"

// SceneManager holds all the scene objects in the scene, owns the MaterialStore, and is
// responsible for packing/uploading the GPU-facing buffers (mesh vertex/index/facegroup buffers,
// the materials SSBO + texture arrays, the lights buffer) and keeping them in sync as the scene
// changes.
namespace lr
{

class Viewer;

// The editor's current interpretation of the selected mesh. View displays the posed/skinned surface;
// Edit and Analysis operate on the unskinned rest mesh so tools and derived values agree with the
// geometry they address.
enum class EditorMode
{
    View,
    Edit,
    Analysis,
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

    // Loads OBJ, glTF, or GLB content into this manager's Scene and asset
    // stores. Returns an identity-transform container for the imported asset.
    SceneObject &load(const std::filesystem::path &path, const SceneLoaderConfig &config = {});

    // Registers renderable scene geometry. The first registered object becomes the initial selected
    // mesh; load() registers the first imported mesh automatically.
    // Registers additional static renderable geometry — e.g. procedural
    // level geometry. `object` must already have a TransformComponent and MeshComponent (see
    // AreaLightVisual.cpp for the pattern: build a Mesh, acquire a MaterialHandle, setFaceGroups,
    // then addComponent<MeshComponent>()). Its TransformComponent is applied as its model matrix at
    // draw time. Must be called before initialize()/uploadMeshes().
    // Subsequent position/attribute edits are detected from the mesh's revisions.
    void addMeshObject(SceneObject &object);
    SceneObject *removeSceneObjects(std::span<const SceneObjectId> ids);

    // The object currently targeted by vertex editing (SelectionManager, the vertex-picking
    // overlay's points buffer and heatmap analysis). The caller drives this from Scene Hierarchy
    // selection. Defaults to the first registered mesh.
    SceneObject *editedMeshObject() { return m_editedMeshObject; }
    SceneObject &selectedMeshObject() { return *m_editedMeshObject; }

    // True if `object` has what setEditedMeshObject() requires (a MeshComponent + TransformComponent).
    static bool isEditable(const SceneObject &object);

    // Repoints vertex editing at `object` (see editedMeshObject()) — rebinds the SelectionManager
    // and replaces the vertex-picking overlay's points buffer for the new mesh's vertex count,
    // which may differ arbitrarily from the previous mesh's. Resets selection, highlighting, and
    // per-vertex role classification (SelectionManager::rebind's doc comment). The caller is still
    // responsible for rebinding anything it owns directly against the old mesh (VertexManager,
    // ArapTool — see their rebind()) and for ensuring the GPU is done with the previous points
    // buffer first (see ResourceRegistry::replaceUploadedBuffer's doc comment) — e.g.
    // viewer.context().waitIdle() before calling this, the way the HDRI-reload path does.
    // Also replaces the selected heatmap buffer. Throws if `object` isn't isEditable(). No-op if
    // `object` is already selected.
    void setEditedMeshObject(SceneObject &object);

    // The camera whose Camera/TransformComponent state drives the camera UBO. Must be set before
    // initialize().
    void setDefaultCamera(SceneObject &camera) { m_defaultCamera = &camera; }

    // Updated once per frame from the current swapchain extent (window resize) — read by
    // updateCamera() the next time it runs, so there's no need to force a re-upload here.
    void setAspect(float aspect)
    {
        if (m_aspect != aspect)
        {
            m_aspect = aspect;
            m_cameraAspectDirty = true;
        }
    }

    // Performs all one-time scene setup that would otherwise have to be manually sequenced by the
    // caller: builds light visuals, uploads the initial lights/mesh/material/camera buffers, and
    // constructs the SelectionManager that operates on the initially selected mesh. Requires a
    // scene, at least one registered mesh, and a default camera.
    void initialize(const AreaLightVisualConfig &areaLightVisualConfig, const GpuMaterialLayout &materialLayout,
                    const std::vector<std::string> &vertexAttributeNames, InputHandler &input);

    // Registers the per-frame callbacks SceneManager needs — an onUpdate that tracks the
    // swapchain aspect ratio (see setAspect), an onUpdate that drives the SelectionManager's mouse/
    // drag handling, and an onLateUpdate that calls flushDirty() — so the caller doesn't need to
    // know what SceneManager wires up each frame.
    void registerCallbacks(Viewer &viewer);

    // Selection over the selected mesh's deduped-position space — constructed by initialize(), so only
    // valid to call after it. Highlight changes are wired internally to push the highlighted-vertex
    // colors into mesh CPU data; late synchronization maintains GPU views. The caller still wires a SelectionTool
    // (setSelectTool), the mouse click handoff with gizmos, and reading getSelectedIndices()/
    // getHighlightedIndices() for its own UI (translate gizmo placement, etc.).
    SelectionManager &selectionManager() { return *m_selectionManager; }

    EditorMode editorMode() const { return m_editorMode; }

    // Switches between posed viewing, rest-mesh editing, and rest-mesh analysis. Leaving Edit
    // clears the current selection. SceneManager publishes the change rather than owning render
    // passes, allowing the application to apply one consistent visibility/skinning policy.
    void setEditorMode(EditorMode mode);
    void registerEditorModeChangedCallback(std::function<void(EditorMode)> callback);

    // Flushes component edits, then synchronizes materialized mesh buffers from independent
    // per-representation revision stamps. Position/attribute mutations never upload directly.
    // Call once per frame, after every other update callback has had a chance to mutate the scene — see
    // Viewer::onLateUpdate.
    void flushDirty();

    // Builds one hidden quad MeshComponent (and one MaterialStore slot) per Light currently in the
    // scene — see AreaLightVisual.hpp for why every light gets one regardless of its current type.
    // Called by initialize(); exposed separately in case a caller needs to set up light visuals
    // without going through the full initialize() sequence.
    void createLightVisuals(const AreaLightVisualConfig &config);

    // Packs all registered geometry + every light visual into shared vertex/index/facegroup buffers and
    // uploads them, plus the initial materials SSBO/texture arrays snapshot.
    void uploadMeshes(const GpuMaterialLayout &materialLayout, const std::vector<std::string> &vertexAttributeNames);

    // Re-packs all render geometry after runtime imports/deletions. The caller must wait for the
    // GPU before calling and refresh pass draw metadata/descriptors afterward. Also replaces
    // selected overlays; a selected topology change rebinds selection (caller rebinds its tools).
    void rebuildGeometry();

    void uploadLights();

    // Replaces the analysis colors used by HeatmapPass without touching the selection-highlight
    // colors used by the points overlay. Colors are indexed by mesh.positions().
    void setSelectedMeshHeatmapColors(std::span<const glm::vec3> colors);

    // Re-uploads the materials SSBO from the MaterialStore's current contents — called by
    // flushDirty() when a rendered MeshComponent is dirty (e.g. a Scene Hierarchy slider edit,
    // see MeshComponent::onGUIImpl), so the edit reaches the GPU.
    void updateMaterials();

    // Rebuilds every light visual's CPU quad geometry + MaterialStore slot from its Light/
    // TransformComponent state. Mesh synchronization uploads changed geometry afterward. Called by flushDirty()
    // when any light-visual object's Light or TransformComponent is dirty.
    void updateLightVisuals();

    // Re-uploads the camera UBO from the default camera's current Camera/TransformComponent state and the
    // last aspect ratio set via setAspect(). Called once during initialize(), and by flushDirty()
    // when the default camera's Camera or TransformComponent is dirty.
    void updateCamera();

    // Evaluates every skin against the current joint hierarchy and mesh transform, then uploads
    // the packed joint palettes. Called every frame for the initial implementation.
    void updateSkins();

    // Advances every AnimatorComponent before skin palettes are evaluated.
    void updateAnimations(float deltaSeconds);

    const std::string &cameraBufferName() const { return m_cameraUploader.bufferName(); }

    const SkinUploadResult &skinUploadResult() const { return m_skinUploadResult; }
    const std::string &skinInfluenceEntriesBufferName() const
    {
        return m_skinUploader.influenceEntriesBufferName();
    }
    const std::string &skinInfluenceOffsetsBufferName() const
    {
        return m_skinUploader.influenceOffsetsBufferName();
    }
    const std::string &skinPositionIndicesBufferName() const { return m_skinUploader.positionIndicesBufferName(); }
    const std::string &skinJointMatricesBufferName() const { return m_skinUploader.jointMatricesBufferName(); }

    const std::string &meshPositionBufferName() const { return m_meshPositionBufferName; }
    const std::string &selectedMeshPointsBufferName() const { return m_selectedMeshPointsBufferName; }
    const std::string &meshVertexBufferName() const { return m_meshVertexBufferName; }
    const std::string &meshIndexBufferName() const { return m_meshIndexBufferName; }
    const std::string &meshFaceGroupBufferName() const { return m_meshFaceGroupBufferName; }
    // Interleaved position + color buffer, duped per UV-seam corner (unlike the selected points
    // buffer, which is deduped) — for HeatmapPass, which needs the
    // color Gouraud-interpolated across the same triangles GeometryPass draws, so it must share
    // GeometryPass's corner-indexed topology rather than the
    // deduped-position space the points overlay uses.
    const std::string &selectedMeshHeatmapBufferName() const { return m_selectedMeshHeatmapBufferName; }

    const VertexBufferUploadResult &meshPositions() const { return m_meshPositions; }
    // Selected mesh's unique/deduped position+color buffer.
    const VertexBufferUploadResult &selectedMeshPoints() const { return m_selectedMeshPoints; }
    // Selected mesh's corner-domain position+color buffer. Only
    // ever holds one mesh (singleMeshResults[0]), unlike meshPositions()/indexBuffer().
    const VertexBufferUploadResult &selectedMeshHeatmap() const { return m_selectedMeshHeatmap; }
    const IndexBufferUploadPerMeshResult &selectedMeshIndexRange() const;
    const IndexBufferUploadResult        &indexBuffer() const { return m_indexBuffer; }
    const std::vector<const TransformComponent *> &meshTransforms() const { return m_meshTransforms; }
    const std::vector<SceneObject *> &geometryObjects() const { return m_geometryObjects; }

    const MaterialUploadResult &materialUploadResult() const { return m_materialUploadResult; }

    const std::string &lightBufferName() const { return m_lightUploader.bufferName(); }
    uint32_t           numLights() const { return m_lightUploader.numLights(); }

private:
    void gatherGeometry(const std::vector<std::string> &vertexAttributeNames);

    // Rebuilds the selected mesh's "color" per-unique-vertex attribute from the SelectionManager's
    // current highlighted indices. Wired as m_selectionManager's
    // highlight-changed callback in initialize().
    void updateSelectedMeshHighlightColors();

    void synchronizeMeshes();
    void ensureSelectedMeshAttributes(Mesh &mesh);

    Scene            *m_scene = nullptr;
    ResourceRegistry &m_registry;

    std::unique_ptr<SelectionManager> m_selectionManager;
    EditorMode                        m_editorMode = EditorMode::View;
    std::vector<std::function<void(EditorMode)>> m_editorModeChangedCallbacks;

    MeshUploader     m_meshUploader;
    MaterialUploader m_materialUploader;
    LightUploader    m_lightUploader;
    CameraUploader   m_cameraUploader;
    SkinUploader     m_skinUploader;
    MaterialStore    m_materialStore;
    MeshStore        m_meshStore;

    SceneObject *m_defaultCamera  = nullptr;
    std::vector<SceneObject *> m_meshObjects;
    SceneObject *m_editedMeshObject = nullptr;
    Mesh::Revision m_selectedTopologyRevision = 0;
    // Matches Viewer::Config's default window size until setAspect() is called with the real
    // swapchain extent.
    float                      m_aspect = 1600.0f / 900.0f;
    bool                       m_cameraAspectDirty = false;
    std::vector<SceneObject *> m_lightVisualObjects;
    AreaLightVisualConfig m_areaLightVisualConfig;

    // Established by uploadMeshes()/rebuildGeometry(), polled during late synchronization so
    // every repack targets the same combined mesh list / buffer configs.
    std::vector<const Mesh *>      m_geometryMeshes;
    std::vector<const TransformComponent *> m_meshTransforms;
    std::vector<Skin *>            m_meshSkins;
    std::vector<SceneObject *>      m_geometryObjects;
    VertexBufferUploadConfig       m_meshPositionUploadConfig;
    VertexBufferUploadConfig       m_meshAttributeUploadConfig;
    GpuMaterialLayout              m_materialLayout;
    std::vector<std::string>       m_vertexAttributeNames;
    std::vector<MaterialHandle>    m_pendingTextureUpdates;

    const std::string m_meshPositionBufferName         = "meshPositionBuffer";
    const std::string m_selectedMeshPointsBufferName   = "meshPointsBuffer";
    const std::string m_meshVertexBufferName           = "meshVertexBuffer";
    const std::string m_meshIndexBufferName            = "meshIndexBuffer";
    const std::string m_meshFaceGroupBufferName        = "meshFaceGroupBuffer";
    const std::string m_selectedMeshHeatmapBufferName  = "meshHeatmapBuffer";

    // Config for the deduped position+color buffer above — same shape as m_meshPositionUploadConfig/
    // m_meshAttributeUploadConfig. Its stamp observes both position and highlight revisions.
    VertexBufferUploadConfig m_selectedMeshPointsUploadConfig = {
        .vertexBufferName = m_selectedMeshPointsBufferName, .vertexAttributeNames = {"color"}, .includePosition = true};

    // Config for the corner-domain position+heatmapColors buffer.
    VertexBufferUploadConfig m_selectedMeshHeatmapUploadConfig = {
        .vertexBufferName = m_selectedMeshHeatmapBufferName,
        .vertexAttributeNames = {"heatmapColors"},
        .includePosition = true,
        .expandUniqueVertexAttributes = true};

    VertexBufferUploadResult m_meshPositions;
    VertexBufferUploadResult m_selectedMeshPoints;
    VertexBufferUploadResult m_selectedMeshHeatmap;
    IndexBufferUploadResult  m_indexBuffer;
    MaterialUploadResult     m_materialUploadResult;
    SkinUploadResult         m_skinUploadResult;
};

} // namespace lr
