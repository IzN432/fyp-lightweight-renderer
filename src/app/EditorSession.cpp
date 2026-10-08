#include "EditorSession.hpp"
#include "EditorRenderBridge.hpp"
#include "core/editor/EditorStateController.hpp"

#include "core/app/Viewer.hpp"
#include "core/editor/DefaultVertexDragHandler.hpp"
#include "core/editor/EditableMeshContext.hpp"
#include "core/editor/EditorContext.hpp"
#include "core/editor/EditorFrameContext.hpp"
#include "core/editor/EditorInputRouter.hpp"
#include "core/editor/EditorShortcuts.hpp"
#include "core/editor/PointerCapture.hpp"
#include "core/editor/EditorTool.hpp"
#include "core/editor/VertexCentroid.hpp"
#include "core/editor/SceneObjectDragHandler.hpp"
#include "core/editor/SceneObjectRotationHandler.hpp"
#include "core/editor/SceneObjectScaleHandler.hpp"
#include "core/editor/SceneObjectTransformController.hpp"
#include "core/editor/VertexManager.hpp"
#include "core/editor/command/CommandManager.hpp"
#include "core/editor/camera/CameraController.hpp"
#include "core/editor/gizmo/GizmoController.hpp"
#include "core/editor/selection/SelectionGestureTool.hpp"
#include "core/scene/Camera.hpp"
#include "core/scene/MeshComponent.hpp"
#include "core/scene/SceneManager.hpp"
#include "features/arap/ArapTool.hpp"
#include "features/laplace_beltrami/LaplaceBeltramiTool.hpp"

#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/vec4.hpp>
#include <imgui.h>

#include <algorithm>
#include <unordered_set>
#include <vector>

namespace lr
{

namespace
{
constexpr std::string_view kViewState = "view";
constexpr std::string_view kEditState = "edit";
} // namespace

class EditorSession::Impl
{
public:
    Impl(Viewer &viewer, SceneManager &sceneManager, SceneObject &camera,
         EditorRenderBridge &renderBridge)
        : m_viewer(viewer), m_sceneManager(sceneManager), m_scene(sceneManager.scene()), m_camera(camera),
          m_renderBridge(renderBridge),
          m_stateController([this](const EditorStateDefinition &state) {
              m_sceneManager.setEditorPresentation(state.presentation);
          }),
          m_meshObject(sceneManager.editedMeshObject()), m_vertexManager(sceneManager.editedMesh()),
          m_selectionManager(sceneManager.selectionManager()),
          m_defaultVertexHandler(m_vertexManager, m_selectionManager, m_commandManager),
          m_objectTranslationHandler(m_commandManager), m_objectRotationHandler(m_commandManager),
          m_objectScaleHandler(m_commandManager),
          m_gizmoController(m_defaultVertexHandler, m_objectRotationHandler, m_objectScaleHandler),
          m_inputRouter(imguiCapturesPointer,
                        [this] {
                            return m_gizmoController.capturesMouse();
                        }),
          m_transformController(m_objectTranslationHandler, m_objectRotationHandler, m_objectScaleHandler),
          m_context{m_transformController, m_commandManager},
          m_arapTool(m_selectionManager, m_vertexManager, m_commandManager), m_laplaceBeltramiTool(sceneManager),
          m_tools{&m_arapTool, &m_laplaceBeltramiTool}
    {
        m_selectionManager.setSelectTool(std::make_unique<SelectionGestureTool>(viewer.input(), camera));

        const auto applyPresentation = [this](const EditorPresentation &presentation) {
            m_renderBridge.apply(presentation);
        };
        m_connections.push_back(m_sceneManager.registerEditorPresentationChangedCallback(applyPresentation));

        registerEditorStates();
        registerEditorShortcuts();
        m_stateController.setDefaultState(kViewState);

        // Tools register after the baseline states and shortcuts, so that a tool retreating to the
        // default state always finds one and cannot take over a key the editor itself needs.
        EditorServices services(m_stateController, m_shortcuts);
        for (EditorTool *tool : m_tools)
        {
            tool->registerWith(services);
        }

        m_stateController.activate(kViewState);
        applyPresentation(m_sceneManager.editorPresentation());
        // The editor always opens with no target — loading or importing geometry does not pick one,
        // only Scene Hierarchy selection does — so an empty scene is just the ordinary case.
        if (m_meshObject)
        {
            notifyTargetChanged(*m_meshObject);
        } else
        {
            notifyTargetCleared();
        }

        registerSceneCallbacks();
        registerInputLayers();
        registerInputCallbacks();
        m_connections.push_back(m_viewer.onUpdate([this](float, VkExtent2D extent) {
            update(extent);
        }));
    }

    EditorContext &context() { return m_context; }

    bool allowsViewportNavigation() const { return m_inputRouter.viewportNavigationAllowed(); }

    void onSceneContentChanged()
    {
        SceneObject *editedMesh = m_sceneManager.editedMeshObject();
        if (!editedMesh)
        {
            m_stateController.activateDefault();
            notifyTargetCleared();
            m_transformController.setSelectedTarget(nullptr);
            return;
        }
        if (editedMesh == m_meshObject)
        {
            m_renderBridge.setEditableTarget({
                .object = *editedMesh,
                .mesh   = editedMesh->getComponent<MeshComponent>().mesh(),
            });
            return;
        }
        rebindEditableTarget(*editedMesh, false);
    }

    void drawFeaturePanel()
    {
        for (EditorTool *tool : m_tools)
        {
            if (!ImGui::CollapsingHeader(tool->displayName(), ImGuiTreeNodeFlags_DefaultOpen))
            {
                continue;
            }
            ImGui::PushID(tool);
            ImGui::Indent();
            tool->drawPanel();
            ImGui::Unindent();
            ImGui::PopID();
        }
    }

    void drawTransformWindow()
    {
        if (!m_transformWindowOpen || !m_scene.selectedObject())
        {
            return;
        }

        const ImGuiViewport *viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + viewport->WorkSize.x * 0.5f, viewport->WorkPos.y + 20.0f),
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
    // Each state registers its own presentation *and* its own behavior, so neither update() nor the
    // gizmo arbitration below needs to know that ARAP, vertex editing, or analysis exist. Once
    // feature tools register themselves (EditorTool), these closures move out with them.
    void registerEditorStates()
    {
        m_stateController.registerState({
            .id           = std::string(kViewState),
            .presentation = {},
            .gizmoRequest =
                [this](const EditorFrameContext &) {
                    return objectTransformGizmoRequest();
                },
        });

        m_stateController.registerState({
            .id           = std::string(kEditState),
            .presentation = {.skinningEnabled = false, .vertexPointsVisible = true, .vertexSelectionActive = true},
            .gizmoRequest = [this](const EditorFrameContext &) -> GizmoRequest {
                if (!m_meshObject || m_defaultVertexHandler.indices().empty())
                {
                    return {};
                }
                return TranslateGizmoRequest{
                    .origin  = worldCentroidOf(m_defaultVertexHandler.indices()),
                    .handler = &m_defaultVertexHandler,
                };
            },
        });
    }

    // Target binding is one operation applied to every registered tool, so adding a feature does not
    // mean adding a line here.
    void notifyTargetChanged(SceneObject &object)
    {
        m_defaultVertexHandler.setTargetTransform(&object.getComponent<TransformComponent>());
        const EditableMeshContext target{
            .object = object,
            .mesh   = object.getComponent<MeshComponent>().mesh(),
        };
        for (EditorTool *tool : m_tools)
        {
            tool->onTargetChanged(target);
        }
        m_renderBridge.setEditableTarget(target);
    }

    void notifyTargetCleared()
    {
        m_meshObject = nullptr;
        m_defaultVertexHandler.setTargetTransform(nullptr);
        for (EditorTool *tool : m_tools)
        {
            tool->onTargetCleared();
        }
    }

    // The transform gizmo the hierarchy selection drives. Only the View state bids for it, which is
    // what previously made every object-gizmo branch test for View mode explicitly.
    GizmoRequest objectTransformGizmoRequest()
    {
        switch (m_transformController.tool())
        {
            case TransformTool::Translate:
                if (const SceneObject *target = m_objectTranslationHandler.target())
                {
                    return TranslateGizmoRequest{
                        .origin  = glm::vec3(target->worldMatrix()[3]),
                        .handler = &m_objectTranslationHandler,
                    };
                }
                break;
            case TransformTool::Rotate:
                if (const SceneObject *target = m_objectRotationHandler.target())
                {
                    return RotateGizmoRequest{
                        .worldMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(target->worldMatrix()[3])) *
                                       glm::mat4_cast(target->worldRotation()),
                        .handler     = &m_objectRotationHandler,
                    };
                }
                break;
            case TransformTool::Scale:
                if (const SceneObject *target = m_objectScaleHandler.target())
                {
                    return ScaleGizmoRequest{
                        .worldMatrix = target->worldMatrix(),
                        .handler     = &m_objectScaleHandler,
                    };
                }
                break;
        }
        return {};
    }

    void registerSceneCallbacks()
    {
        m_connections.push_back(m_scene.registerSelectionChangedCallback([this](SceneObjectId id) {
            SceneObject &object   = m_scene.getSceneObject(id);
            m_transformWindowOpen = true;
            m_transformController.setSelectedTarget(object.hasComponent<TransformComponent>() ? &object : nullptr);
            if (!SceneManager::isEditable(object))
            {
                if (m_stateController.active().presentation.vertexSelectionActive)
                {
                    m_stateController.activateDefault();
                }
                return;
            }
            if (m_sceneManager.editedMeshObject() != &object)
            {
                m_viewer.context().waitIdle();
                m_sceneManager.setEditedMeshObject(object);
                rebindEditableTarget(object, true);
            }
        }));

        m_connections.push_back(m_scene.registerObjectsDestroyedCallback([this](std::span<const SceneObjectId> ids) {
            m_viewer.context().waitIdle();
            const SceneObject *previousEditedMesh = m_sceneManager.editedMeshObject();
            const bool         editedMeshDestroyed =
                previousEditedMesh && std::ranges::find(ids, previousEditedMesh->id()) != ids.end();

            if (m_transformController.target() &&
                std::ranges::find(ids, m_transformController.target()->id()) != ids.end())
            {
                m_transformWindowOpen = false;
            }
            m_transformController.onObjectsDestroyed(ids);

            m_sceneManager.removeSceneObjects(ids);
            m_sceneManager.uploadLights();
            if (!editedMeshDestroyed)
            {
                return;
            }
            // Deleting the edited mesh leaves nothing being vertex-edited rather than silently
            // moving the target to another object: a target comes from Scene Hierarchy selection.
            m_stateController.activateDefault();
            notifyTargetCleared();
        }));
    }

    // Pointer priority, highest first. The UI and the active gizmo outrank every layer and are
    // handled by the router itself; camera navigation sits below all of them and polls rather than
    // consuming, so it is not a layer either.
    void registerInputLayers()
    {
        // The active state gets first refusal, which is how a feature claims viewport input for as
        // long as its state is the active one.
        m_inputRouter.addButtonLayer("active editor state", [this](const PointerButtonEvent &event) {
            return m_stateController.handleInput(event);
        });

        // Generic vertex picking, available to any state whose presentation asks for it.
        m_inputRouter.addButtonLayer("vertex selection", [this](const PointerButtonEvent &event) {
            if (event.button != GLFW_MOUSE_BUTTON_LEFT ||
                !m_stateController.active().presentation.vertexSelectionActive)
            {
                return false;
            }
            m_selectionManager.mouseButtonCallback(event.button, event.action, event.shift, event.ctrl, event.alt);
            return true;
        });
    }

    void registerInputCallbacks()
    {
        m_connections.push_back(m_viewer.input().onMouseButton(
            [this](int button, int action, bool shift, bool ctrl, bool alt) {
            m_inputRouter.routeButton({
                .button = button,
                .action = action,
                .shift  = shift,
                .ctrl   = ctrl,
                .alt    = alt,
            });
        }));

        // One key callback for the whole editor. The ImGui check that each of these handlers used to
        // repeat now happens here, once, and chord matching belongs to EditorShortcuts.
        m_connections.push_back(m_viewer.input().onKeyPress([this](int key, int action, bool, bool ctrl, bool) {
            if (action != GLFW_PRESS || imguiCapturesKeyboard())
            {
                return;
            }
            m_shortcuts.dispatch({.key = key, .ctrl = ctrl});
        }));
    }

    // The editor's own shortcuts. Registered before any tool's, so a feature cannot shadow them.
    void registerEditorShortcuts()
    {
        m_shortcuts.add({.key = GLFW_KEY_R}, [this] {
            SceneObject *camera = m_sceneManager.gpu().camera();
            if (camera)
            {
                if (CameraController *controller = camera->findComponent<CameraController>())
                {
                    controller->resetTransformation();
                }
            }
        });

        m_shortcuts.add({.key = GLFW_KEY_TAB}, [this] {
            const bool nowEditing = !m_stateController.isActive(kEditState);
            if (nowEditing)
            {
                const auto selected = m_scene.selectedObject();
                if (!selected || !SceneManager::isEditable(m_scene.getSceneObject(*selected)))
                {
                    return;
                }
            }
            m_stateController.activate(nowEditing ? kEditState : kViewState);
        });

        m_shortcuts.add({.key = GLFW_KEY_Z, .ctrl = true}, [this] {
            m_commandManager.undo();
        });

        m_shortcuts.add({.key = GLFW_KEY_DELETE}, [this] {
            const auto selected = m_scene.selectedObject();
            if (selected && m_scene.canDestroySceneObject(*selected))
            {
                m_scene.destroySceneObject(*selected);
            }
        });
    }

    void rebindEditableTarget(SceneObject &object, bool sceneManagerAlreadyRebound)
    {
        if (!sceneManagerAlreadyRebound)
        {
            // Unconditional: setEditedMeshObject() no-ops when `object` is already the target, and
            // because it is the only thing that sets one, already-the-target means already rebound.
            m_sceneManager.setEditedMeshObject(object);
        }
        m_meshObject = &object;
        m_vertexManager.rebind(object.getComponent<MeshComponent>().mesh());
        notifyTargetChanged(object);
    }

    glm::vec3 worldCentroidOf(const std::unordered_set<uint32_t> &indices) const
    {
        return worldVertexCentroid(m_vertexManager, m_meshObject->getComponent<TransformComponent>(), indices);
    }

    void update(VkExtent2D extent)
    {
        const Camera &camera = m_camera.getComponent<Camera>();
        const float   aspect =
            extent.height == 0 ? 1.0f : static_cast<float>(extent.width) / static_cast<float>(extent.height);

        const EditorFrameContext frame{
            .extent         = extent,
            .aspect         = aspect,
            .view           = camera.viewMatrix(),
            .projection     = camera.projectionMatrix(aspect),
            .viewProjection = camera.viewProjectionMatrix(aspect),
            .cameraPosition = glm::vec3(m_camera.worldMatrix()[3]),
            .cameraForward  = m_camera.worldRotation() * glm::vec3(0.0f, 0.0f, -1.0f),
            .orthographic   = camera.projectionType == ProjectionType::Orthographic,
        };

        m_stateController.update(frame);
        m_gizmoController.draw(frame.view, frame.projection, frame.orthographic, extent,
                               m_stateController.gizmoRequest(frame));

        m_renderBridge.updateSceneOverlays(m_scene, frame);
    }

    Viewer                        &m_viewer;
    SceneManager                  &m_sceneManager;
    Scene                         &m_scene;
    SceneObject                   &m_camera;
    EditorRenderBridge            &m_renderBridge;
    EditorStateController          m_stateController;
    EditorShortcuts                m_shortcuts;
    SceneObject                   *m_meshObject;
    VertexManager                  m_vertexManager;
    CommandManager                 m_commandManager;
    SelectionManager              &m_selectionManager;
    DefaultVertexDragHandler       m_defaultVertexHandler;
    SceneObjectDragHandler         m_objectTranslationHandler;
    SceneObjectRotationHandler     m_objectRotationHandler;
    SceneObjectScaleHandler        m_objectScaleHandler;
    GizmoController                m_gizmoController;
    EditorInputRouter              m_inputRouter;
    SceneObjectTransformController m_transformController;
    EditorContext                  m_context;
    ArapTool                       m_arapTool;
    LaplaceBeltramiTool            m_laplaceBeltramiTool;
    // Every registered tool, in panel order. Declared after the tools it points at.
    std::vector<EditorTool *> m_tools;
    bool                      m_transformWindowOpen = false;
    // Declared last so every external callback disconnects before the objects it may call into.
    std::vector<CallbackConnection> m_connections;
};

EditorSession::EditorSession(Viewer &viewer, SceneManager &sceneManager, SceneObject &camera,
                             EditorRenderBridge &renderBridge)
    : m_impl(std::make_unique<Impl>(viewer, sceneManager, camera, renderBridge))
{}

EditorSession::~EditorSession() = default;

EditorContext &EditorSession::context() { return m_impl->context(); }
void           EditorSession::onSceneContentChanged() { m_impl->onSceneContentChanged(); }
void           EditorSession::drawFeaturePanel() { m_impl->drawFeaturePanel(); }
void           EditorSession::drawTransformWindow() { m_impl->drawTransformWindow(); }
bool           EditorSession::allowsViewportNavigation() const { return m_impl->allowsViewportNavigation(); }

} // namespace lr
