#include "EditorSession.hpp"

#include "core/app/Viewer.hpp"
#include "core/editor/DefaultVertexDragHandler.hpp"
#include "core/editor/EditorContext.hpp"
#include "core/editor/SceneObjectDragHandler.hpp"
#include "core/editor/SceneObjectRotationHandler.hpp"
#include "core/editor/SceneObjectScaleHandler.hpp"
#include "core/editor/SceneObjectTransformController.hpp"
#include "core/editor/VertexManager.hpp"
#include "core/editor/command/CommandManager.hpp"
#include "core/editor/gizmo/RotateGizmo.hpp"
#include "core/editor/gizmo/ScaleGizmo.hpp"
#include "core/editor/gizmo/TranslateGizmo.hpp"
#include "core/editor/selection/SelectionGestureTool.hpp"
#include "core/passes/geometry/GeometryPass.hpp"
#include "core/passes/heatmap/HeatmapPass.hpp"
#include "core/passes/overlaylines/OverlayLinesPass.hpp"
#include "core/passes/overlaypoints/OverlayPointsPass.hpp"
#include "core/passes/transparent/TransparentPass.hpp"
#include "core/scene/Camera.hpp"
#include "core/scene/MeshComponent.hpp"
#include "core/scene/SceneManager.hpp"
#include "features/arap/ArapTool.hpp"
#include "features/laplace_beltrami/LaplaceBeltramiTool.hpp"
#include "features/rigid_body/ColliderVisual.hpp"

#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/vec4.hpp>
#include <imgui.h>

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace lr
{

class EditorSession::Impl
{
public:
    Impl(Viewer &viewer, SceneManager &sceneManager, SceneObject &camera,
         GeometryPass &geometryPass, TransparentPass &transparentPass,
         HeatmapPass &heatmapPass, OverlayPointsPass &overlayPointsPass,
         OverlayLinesPass &overlayLinesPass)
        : m_viewer(viewer), m_sceneManager(sceneManager), m_scene(sceneManager.scene()), m_camera(camera),
          m_geometryPass(geometryPass), m_transparentPass(transparentPass),
          m_heatmapPass(heatmapPass), m_overlayPointsPass(overlayPointsPass),
          m_overlayLinesPass(overlayLinesPass), m_meshObject(requireEditedMesh(sceneManager)),
          m_vertexManager(m_meshObject->getComponent<MeshComponent>().mesh()),
          m_selectionManager(sceneManager.selectionManager()),
          m_defaultVertexHandler(m_vertexManager, m_selectionManager, m_commandManager),
          m_objectTranslationHandler(m_commandManager), m_objectRotationHandler(m_commandManager),
          m_objectScaleHandler(m_commandManager), m_translateGizmo(m_defaultVertexHandler),
          m_rotateGizmo(m_objectRotationHandler), m_scaleGizmo(m_objectScaleHandler),
          m_transformController(m_objectTranslationHandler, m_objectRotationHandler, m_objectScaleHandler),
          m_context{m_transformController, m_commandManager},
          m_arapTool(m_selectionManager, m_vertexManager, m_commandManager,
                     m_meshObject->getComponent<MeshComponent>().mesh(), m_defaultVertexHandler,
                     m_translateGizmo),
          m_laplaceBeltramiTool(m_meshObject->getComponent<MeshComponent>().mesh(), sceneManager)
    {
        m_selectionManager.setSelectTool(std::make_unique<SelectionGestureTool>(viewer.input(), camera));

        // Register what needs to be changed when editor mode is changed.
        // Specifically, we are only enabling skinning in view mode, and only displaying overlay points in edit mode,
        // and only displaying the heatmap in analysis (heatmap) mode.
        const auto applyEditorMode = [this](EditorMode mode) {
            m_geometryPass.setSkinningEnabled(mode == EditorMode::View);
            m_transparentPass.setSkinningEnabled(mode == EditorMode::View);
            m_overlayPointsPass.setEnabled(mode == EditorMode::Edit);
            m_heatmapPass.setEnabled(mode == EditorMode::Analysis);
        };
        m_sceneManager.registerEditorModeChangedCallback(applyEditorMode);
        applyEditorMode(m_sceneManager.editorMode());

        registerSceneCallbacks();
        registerInputCallbacks();
        m_viewer.onUpdate([this](float, VkExtent2D extent) {
            update(extent);
        });
    }

    EditorContext &context() { return m_context; }

    bool capturesMouse() const
    {
        return m_translateGizmo.capturesMouse() || m_rotateGizmo.capturesMouse() ||
               m_scaleGizmo.capturesMouse();
    }

    void onSceneContentChanged()
    {
        SceneObject *editedMesh = m_sceneManager.editedMeshObject();
        if (!editedMesh)
        {
            m_sceneManager.setEditorMode(EditorMode::View);
            m_transformController.setSelectedTarget(nullptr);
            return;
        }
        if (editedMesh == m_meshObject)
        {
            refreshRenderSources(*editedMesh);
            return;
        }
        rebindEditableTarget(*editedMesh, false);
    }

    void drawFeaturePanel()
    {
        if (ImGui::CollapsingHeader("Laplace-Beltrami", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Indent();
            m_laplaceBeltramiTool.onGui();
            ImGui::Unindent();
        }
        if (ImGui::CollapsingHeader("ARAP", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Indent();
            m_arapTool.onPanelGui();
            ImGui::Unindent();
        }
    }

    void drawTransformWindow()
    {
        if (!m_transformWindowOpen || !m_scene.selectedObject())
        {
            return;
        }

        const ImGuiViewport *viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + viewport->WorkSize.x * 0.5f,
                                       viewport->WorkPos.y + 20.0f),
                                ImGuiCond_Appearing, ImVec2(0.5f, 0.0f));
        ImGui::SetNextWindowSize(ImVec2(330.0f, 0.0f), ImGuiCond_Appearing);
        if (ImGui::Begin("Transform Gizmo", &m_transformWindowOpen, ImGuiWindowFlags_AlwaysAutoResize))
        {
            const bool canTransform = m_transformController.target() != nullptr;
            ImGui::BeginDisabled(!canTransform || m_transformController.hasTemporaryEdit());
            if (ImGui::RadioButton("Translate", m_transformController.tool() == TransformTool::Translate))
            {
                m_transformController.setTool(TransformTool::Translate);
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Rotate", m_transformController.tool() == TransformTool::Rotate))
            {
                m_transformController.setTool(TransformTool::Rotate);
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Scale", m_transformController.tool() == TransformTool::Scale))
            {
                m_transformController.setTool(TransformTool::Scale);
            }
            ImGui::EndDisabled();
        }
        ImGui::End();
    }

private:
    static SceneObject *requireEditedMesh(SceneManager &sceneManager)
    {
        SceneObject *mesh = sceneManager.editedMeshObject();
        if (!mesh)
        {
            throw std::logic_error("EditorSession requires an initial editable mesh");
        }
        return mesh;
    }

    void registerSceneCallbacks()
    {
        m_scene.registerSelectionChangedCallback([this](SceneObjectId id) {
            SceneObject &object = m_scene.getSceneObject(id);
            m_transformWindowOpen = true;
            m_transformController.setSelectedTarget(object.hasComponent<TransformComponent>() ? &object : nullptr);
            if (!SceneManager::isEditable(object))
            {
                if (m_sceneManager.editorMode() == EditorMode::Edit)
                {
                    m_sceneManager.setEditorMode(EditorMode::View);
                }
                return;
            }
            if (m_sceneManager.editedMeshObject() != &object)
            {
                m_viewer.context().waitIdle();
                m_sceneManager.setEditedMeshObject(object);
                rebindEditableTarget(object, true);
            }
        });

        m_scene.registerObjectsDestroyedCallback([this](std::span<const SceneObjectId> ids) {
            m_viewer.context().waitIdle();
            const SceneObject *previousEditedMesh = m_sceneManager.editedMeshObject();
            const bool editedMeshDestroyed = previousEditedMesh &&
                std::ranges::find(ids, previousEditedMesh->id()) != ids.end();

            if (m_transformController.target() &&
                std::ranges::find(ids, m_transformController.target()->id()) != ids.end())
            {
                m_transformWindowOpen = false;
            }
            m_transformController.onObjectsDestroyed(ids);

            SceneObject *replacement = m_sceneManager.removeSceneObjects(ids);
            m_sceneManager.uploadLights();
            if (!editedMeshDestroyed)
            {
                return;
            }
            if (!replacement)
            {
                m_sceneManager.setEditorMode(EditorMode::View);
                return;
            }
            rebindEditableTarget(*replacement, false);
        });
    }

    void registerInputCallbacks()
    {
        m_viewer.input().onMouseButton([this](int button, int action, bool shift, bool ctrl, bool alt) {
            if (button != GLFW_MOUSE_BUTTON_LEFT || ImGui::GetIO().WantCaptureMouse || capturesMouse() ||
                m_sceneManager.editorMode() != EditorMode::Edit)
            {
                return;
            }
            m_selectionManager.mouseButtonCallback(button, action, shift, ctrl, alt);
        });

        m_viewer.input().onKeyPress([this](int key, int action, bool, bool, bool) {
            if (key != GLFW_KEY_TAB || action != GLFW_PRESS || ImGui::GetIO().WantCaptureKeyboard)
            {
                return;
            }
            const bool nowEditing = m_sceneManager.editorMode() != EditorMode::Edit;
            if (nowEditing)
            {
                const auto selected = m_scene.selectedObject();
                if (!selected || !SceneManager::isEditable(m_scene.getSceneObject(*selected)))
                {
                    return;
                }
            }
            m_sceneManager.setEditorMode(nowEditing ? EditorMode::Edit : EditorMode::View);
        });

        m_viewer.input().onKeyPress([this](int key, int action, bool, bool ctrl, bool) {
            if (key == GLFW_KEY_Z && action == GLFW_PRESS && ctrl && !ImGui::GetIO().WantCaptureKeyboard)
            {
                m_commandManager.undo();
            }
        });

        m_viewer.input().onKeyPress([this](int key, int action, bool, bool, bool) {
            if (key == GLFW_KEY_DELETE && action == GLFW_PRESS && !ImGui::GetIO().WantCaptureKeyboard)
            {
                const auto selected = m_scene.selectedObject();
                if (selected && m_scene.canDestroySceneObject(*selected))
                {
                    m_scene.destroySceneObject(*selected);
                }
            }
        });

        m_viewer.input().onKeyPress([this](int key, int action, bool, bool, bool) {
            if (key == GLFW_KEY_A && action == GLFW_PRESS && !ImGui::GetIO().WantCaptureKeyboard)
            {
                m_arapTool.setModeActive(!m_arapTool.isModeActive());
            }
        });
    }

    void rebindEditableTarget(SceneObject &object, bool sceneManagerAlreadyRebound)
    {
        if (!sceneManagerAlreadyRebound && m_sceneManager.editedMeshObject() != &object)
        {
            m_sceneManager.setEditedMeshObject(object);
        }
        m_meshObject = &object;
        Mesh &mesh = object.getComponent<MeshComponent>().mesh();
        m_vertexManager.rebind(mesh);
        m_arapTool.rebind(mesh);
        m_laplaceBeltramiTool.rebind(mesh);
        refreshRenderSources(object);
    }

    void refreshRenderSources(SceneObject &object)
    {
        Mesh &mesh = object.getComponent<MeshComponent>().mesh();
        m_heatmapPass.setMeshSource(m_sceneManager.selectedMeshHeatmap(),
                                    m_sceneManager.selectedMeshIndexRange(),
                                    object.getComponent<TransformComponent>());
        m_overlayPointsPass.setPointsSource(m_sceneManager.selectedMeshPoints(), mesh.uniquePositionCount(),
                                            object.getComponent<TransformComponent>());
    }

    glm::vec3 worldCentroidOf(const std::unordered_set<uint32_t> &indices) const
    {
        glm::vec3 localCentroid(0.0f);
        for (uint32_t index : indices)
        {
            localCentroid += m_vertexManager.getPositions()[index];
        }
        localCentroid /= static_cast<float>(indices.size());
        return glm::vec3(m_meshObject->getComponent<TransformComponent>().worldMatrix() *
                         glm::vec4(localCentroid, 1.0f));
    }

    void update(VkExtent2D extent)
    {
        const float aspect = extent.height == 0 ? 1.0f :
            static_cast<float>(extent.width) / static_cast<float>(extent.height);
        const Camera &camera = m_camera.getComponent<Camera>();
        const auto &selected = m_selectionManager.getSelectedIndices();
        m_arapTool.onOverlayGui(camera.viewProjectionMatrix(aspect), extent,
                                selected.empty() ? glm::vec3(0.0f) : worldCentroidOf(selected));

        const bool objectTranslateActive = m_transformController.tool() == TransformTool::Translate &&
                                           m_sceneManager.editorMode() == EditorMode::View &&
                                           m_objectTranslationHandler.target();
        const bool objectRotateActive = m_transformController.tool() == TransformTool::Rotate &&
                                        m_sceneManager.editorMode() == EditorMode::View &&
                                        m_objectRotationHandler.target();
        const bool objectScaleActive = m_transformController.tool() == TransformTool::Scale &&
                                       m_sceneManager.editorMode() == EditorMode::View &&
                                       m_objectScaleHandler.target();

        if (objectTranslateActive)
        {
            if (&m_translateGizmo.dragHandler() != &m_objectTranslationHandler)
            {
                m_objectTranslateReturnHandler = &m_translateGizmo.dragHandler();
            }
            m_translateGizmo.setDragHandler(m_objectTranslationHandler);
        }
        else if (&m_translateGizmo.dragHandler() == &m_objectTranslationHandler)
        {
            m_translateGizmo.setDragHandler(*m_objectTranslateReturnHandler);
        }

        const TranslateDragHandler &activeHandler = m_translateGizmo.dragHandler();
        const auto &driven = objectTranslateActive
            ? m_selectionManager.getSelectedIndices()
            : static_cast<const VertexDragHandler &>(activeHandler).indices();
        const bool suppressedByArap = m_arapTool.isModeActive() &&
                                      &activeHandler == &m_defaultVertexHandler;
        const bool translateVisible = (objectTranslateActive || !driven.empty()) && !suppressedByArap;
        glm::vec3 translateOrigin(0.0f);
        if (translateVisible)
        {
            translateOrigin = objectTranslateActive
                ? glm::vec3(m_objectTranslationHandler.target()->worldMatrix()[3])
                : worldCentroidOf(driven);
        }

        const glm::mat4 projection = camera.projectionMatrix(aspect);
        const bool orthographic = camera.projectionType == ProjectionType::Orthographic;
        m_translateGizmo.draw(camera.viewMatrix(), projection, orthographic, extent,
                              translateOrigin, translateVisible);
        m_rotateGizmo.draw(camera.viewMatrix(), projection, orthographic, extent,
                           objectRotateActive
                               ? glm::translate(glm::mat4(1.0f), glm::vec3(m_objectRotationHandler.target()->worldMatrix()[3])) *
                                     glm::mat4_cast(m_objectRotationHandler.target()->worldRotation())
                               : glm::mat4(1.0f),
                           objectRotateActive);
        m_scaleGizmo.draw(camera.viewMatrix(), projection, orthographic, extent,
                          objectScaleActive ? m_objectScaleHandler.target()->worldMatrix() : glm::mat4(1.0f),
                          objectScaleActive);

        std::vector<OverlayLine> overlayLines = buildColliderOverlayLines(m_scene);
        if (const auto selectedObject = m_scene.selectedObject())
        {
            OverlayLineBuilder selectionGizmo;
            SelectionGizmoContext context{
                .lines = selectionGizmo,
                .cameraPosition = glm::vec3(m_camera.worldMatrix()[3]),
                .cameraForward = m_camera.worldRotation() * glm::vec3(0.0f, 0.0f, -1.0f),
                .orthographic = orthographic,
            };
            m_scene.getSceneObject(*selectedObject).onSelectGizmo(context);
            std::vector<OverlayLine> selectionLines = selectionGizmo.takeLines();
            overlayLines.insert(overlayLines.end(), selectionLines.begin(), selectionLines.end());
        }
        m_overlayLinesPass.setLines(overlayLines);
    }

    Viewer                    &m_viewer;
    SceneManager              &m_sceneManager;
    Scene                     &m_scene;
    SceneObject               &m_camera;
    GeometryPass              &m_geometryPass;
    TransparentPass           &m_transparentPass;
    HeatmapPass               &m_heatmapPass;
    OverlayPointsPass         &m_overlayPointsPass;
    OverlayLinesPass          &m_overlayLinesPass;
    SceneObject               *m_meshObject;
    VertexManager              m_vertexManager;
    CommandManager             m_commandManager;
    SelectionManager          &m_selectionManager;
    DefaultVertexDragHandler   m_defaultVertexHandler;
    SceneObjectDragHandler     m_objectTranslationHandler;
    SceneObjectRotationHandler m_objectRotationHandler;
    SceneObjectScaleHandler    m_objectScaleHandler;
    TranslateGizmo             m_translateGizmo;
    RotateGizmo                m_rotateGizmo;
    ScaleGizmo                 m_scaleGizmo;
    TranslateDragHandler      *m_objectTranslateReturnHandler = &m_defaultVertexHandler;
    SceneObjectTransformController m_transformController;
    EditorContext                  m_context;
    ArapTool                       m_arapTool;
    LaplaceBeltramiTool            m_laplaceBeltramiTool;
    bool                           m_transformWindowOpen = false;
};

EditorSession::EditorSession(Viewer &viewer, SceneManager &sceneManager, SceneObject &camera,
                             GeometryPass &geometryPass, TransparentPass &transparentPass,
                             HeatmapPass &heatmapPass, OverlayPointsPass &overlayPointsPass,
                             OverlayLinesPass &overlayLinesPass)
    : m_impl(std::make_unique<Impl>(viewer, sceneManager, camera, geometryPass, transparentPass,
                                    heatmapPass, overlayPointsPass, overlayLinesPass))
{}

EditorSession::~EditorSession() = default;

EditorContext &EditorSession::context() { return m_impl->context(); }
void EditorSession::onSceneContentChanged() { m_impl->onSceneContentChanged(); }
void EditorSession::drawFeaturePanel() { m_impl->drawFeaturePanel(); }
void EditorSession::drawTransformWindow() { m_impl->drawTransformWindow(); }
bool EditorSession::capturesMouse() const { return m_impl->capturesMouse(); }

} // namespace lr
