#include "EditorSession.hpp"
#include "EditorRenderBridge.hpp"
#include "SceneDocument.hpp"
#include "core/editor/EditorStateController.hpp"

#include "core/app/Viewer.hpp"
#include "core/app/ImGuiWidgets.hpp"
#include "core/editor/DefaultVertexDragHandler.hpp"
#include "core/editor/Deleter.hpp"
#include "core/editor/EditableMeshContext.hpp"
#include "core/editor/EditorContext.hpp"
#include "core/editor/EditorFrameContext.hpp"
#include "core/editor/ErrorReporter.hpp"
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
#include "core/editor/ViewportPicker.hpp"
#include "core/editor/command/CommandManager.hpp"
#include "core/editor/command/RotateSceneObjectCommand.hpp"
#include "core/editor/command/TranslateSceneObjectCommand.hpp"
#include "core/editor/camera/CameraController.hpp"
#include "core/editor/gizmo/GizmoController.hpp"
#include "core/editor/selection/SelectionGestureTool.hpp"
#include "core/passes/objectpicking/ObjectPickingPass.hpp"
#include "core/passes/overlaygeometry/OverlayGeometryPass.hpp"
#include "core/scene/Camera.hpp"
#include "core/scene/ComponentCatalog.hpp"
#include "core/scene/Light.hpp"
#include "core/scene/MeshComponent.hpp"
#include "core/scene/SceneManager.hpp"
#include "features/rigid_body/ColliderComponent.hpp"
#include "features/rigid_body/RigidBodyComponent.hpp"
#include "features/fluid/FluidComponent.hpp"
#if defined(LR_ENABLE_CUDA)
#include "features/eulerian_particle/SmokeComponent.hpp"
#endif
#include "features/arap/ArapTool.hpp"
#include "features/laplace_beltrami/LaplaceBeltramiTool.hpp"
#include "features/animation/AnimationTrack.hpp"
#include "features/animation/AnimationTrackCommands.hpp"
#include "features/linear_blend_skinning/SkinComponent.hpp"
#include "features/linear_blend_skinning/editor/SkeletonManipulationAdapter.hpp"
#include "features/linear_blend_skinning/editor/SkeletonOverlay.hpp"
#include "features/linear_blend_skinning/editor/SkeletonTopology.hpp"

#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/vec4.hpp>
#include <imgui.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace lr
{

namespace
{
constexpr std::string_view kViewState = "view";
constexpr std::string_view kEditState = "edit";
constexpr std::string_view kSkeletonEditState = "skeleton-edit";

// The shapes "Add Primitive" offers, paired with the file each one imports from
// assets/runtime/primitives/. The menu shows the shape, not the filename, so the list reads as
// geometry rather than as assets — and the name the new object takes in the hierarchy is the same
// label. Adding a shape means dropping its .glb beside the others and adding a row here.
struct Primitive
{
    const char *label;
    const char *fileName;
};

constexpr std::array kPrimitives{
    Primitive{"Plane", "plane.glb"},
    Primitive{"Cube", "cube.glb"},
    Primitive{"Sphere", "sphere.glb"},
    Primitive{"Cylinder", "cylinder.glb"},
    Primitive{"Cone", "cone.glb"},
    Primitive{"Torus", "torus.glb"},
    Primitive{"Monkey", "monkey.glb"},
};

// What the Inspector's "Add Component" offers, in the order it lists them. A component earns a
// place here by having a blank state worth starting from: a mesh does not (it needs geometry, which
// arrives by import or by pasting a mesh component), and neither do skins, whose
// contents address a particular rig or set of clips. SphericalCameraController is left out
// deliberately too — it drives its object's transform, so adding one would take the object over.
ComponentCatalog makeComponentCatalog()
{
    ComponentCatalog catalog;
    catalog.add<TransformComponent>("Transform");
    catalog.add<Camera>("Camera");
    catalog.add<Light>("Light", [](SceneObject &object) {
        object.addComponent<Light>(PointLight{});
    });
    catalog.add<ColliderComponent>("Collider");
    catalog.add<RigidBodyComponent>("Rigid Body");
    catalog.add<FluidComponent>("Fluid Simulation");
#if defined(LR_ENABLE_CUDA)
    catalog.add<eulerian::SmokeComponent>("Smoke Simulation");
#endif
    return catalog;
}
} // namespace

class EditorSession::Impl final : public ComponentAddService, public TransformCommitService
{
public:
    Impl(Viewer &viewer, SceneManager &sceneManager, SceneObject &camera,
         EditorRenderBridge &renderBridge, ErrorReporter &errors)
        : m_viewer(viewer), m_sceneManager(sceneManager), m_scene(sceneManager.scene()), m_camera(camera),
          m_renderBridge(renderBridge), m_errors(errors),
          m_stateController([this](const EditorStateDefinition &state) {
              m_sceneManager.setEditorPresentation(state.presentation);
          }),
          m_meshObject(sceneManager.editedMeshObject()), m_vertexManager(sceneManager.editedMesh()),
          m_selectionManager(sceneManager.selectionManager()),
          m_defaultVertexHandler(m_vertexManager, m_selectionManager, m_commandManager),
          m_objectTranslationHandler(m_commandManager), m_objectRotationHandler(m_commandManager),
          m_objectScaleHandler(m_commandManager), m_skeletonTranslationHandler(m_commandManager),
          m_skeletonRotationHandler(m_commandManager),
          m_gizmoController(m_defaultVertexHandler, m_objectRotationHandler, m_objectScaleHandler),
          m_inputRouter(imguiCapturesPointer,
                        [this] {
                            return m_gizmoController.capturesMouse();
                        }),
          m_transformController(m_objectTranslationHandler, m_objectRotationHandler, m_objectScaleHandler),
          m_objectPicker(viewer.context(), viewer.allocator()),
          m_skeletonPicker(viewer.context(), viewer.allocator()),
          m_componentCatalog(makeComponentCatalog()),
          m_context{*this, m_commandManager, *this, m_componentCatalog},
          m_arapTool(m_selectionManager, m_vertexManager, m_commandManager), m_laplaceBeltramiTool(sceneManager),
          m_tools{&m_arapTool, &m_laplaceBeltramiTool}
    {
        m_selectionManager.setSelectTool(std::make_unique<SelectionGestureTool>(viewer.input(), camera));
        m_objectTranslationHandler.setCommitCallback([this](SceneObject &object) {
            onTransformCommitted(object, AnimationTargetProperty::Translation);
        });
        m_objectRotationHandler.setCommitCallback([this](SceneObject &object) {
            onTransformCommitted(object, AnimationTargetProperty::Rotation);
        });
        m_objectScaleHandler.setCommitCallback([this](SceneObject &object) {
            onTransformCommitted(object, AnimationTargetProperty::Scale);
        });
        m_skeletonTranslationHandler.setCommitCallback([this](SceneObject &object) {
            onTransformCommitted(object, AnimationTargetProperty::Translation);
        });
        m_skeletonRotationHandler.setCommitCallback([this](SceneObject &object) {
            onTransformCommitted(object, AnimationTargetProperty::Rotation);
        });
        const auto beginSkeletonPoseEdit = [this](SceneObject &) {
            m_sceneManager.animationSystem().stopAll();
            m_animationPlaybackWarning = "Playback stopped to edit the pose.";
        };
        m_skeletonTranslationHandler.setBeginCallback(beginSkeletonPoseEdit);
        m_skeletonRotationHandler.setBeginCallback(beginSkeletonPoseEdit);

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

    // An object the Inspector has just added a component to. An object that has only now become
    // renderable is not in SceneGpu's mesh list, and nothing else would put it there: registering it
    // is what lets the next geometry re-pack — the one the new component asks for through its
    // Geometry dirty aspect — upload it. addMeshObject ignores an object it already holds, so
    // adding anything else costs nothing.
    void onComponentAdded(SceneObject &object) override
    {
        // Both components, because the geometry gather reads the object's transform as its model
        // matrix and would throw on an object without one.
        if (object.hasComponent<MeshComponent>() && object.hasComponent<TransformComponent>())
        {
            m_sceneManager.addMeshObject(object);
        }
    }

    bool allowsViewportNavigation() const { return m_inputRouter.viewportNavigationAllowed(); }

    void onTransformCommitted(SceneObject &object, TransformTool tool) override
    {
        if (tool == TransformTool::Translate)
            onTransformCommitted(object, AnimationTargetProperty::Translation);
        else if (tool == TransformTool::Rotate)
            onTransformCommitted(object, AnimationTargetProperty::Rotation);
        else if (tool == TransformTool::Scale)
            onTransformCommitted(object, AnimationTargetProperty::Scale);
    }

    void onSceneContentChanged()
    {
        if (m_stateController.isActive(kSkeletonEditState))
        {
            m_stateController.activateDefault();
        }
        // Imports and full scene replacements can invalidate either kind of captured delete action.
        m_scene.clearHierarchySelection();
        m_deleter.clear();
        m_renamingAnimation.reset();
        m_animationRenameBuffer.clear();
        m_pendingTrack.reset();
        m_openAddTrackPopup = false;
        m_animationPlaybackWarning.clear();
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

    void drawSkinningWindow()
    {
        const bool visible = ImGui::Begin("Skinning");
        drawSkeletonEndpointPopup();
        if (!visible)
        {
            ImGui::End();
            return;
        }

        std::unordered_map<SkinId, size_t> userCounts;
        for (const auto &entry : m_scene.sceneObjects())
        {
            if (entry && m_scene.contains(entry->id()) && entry->hasComponent<SkinComponent>())
            {
                ++userCounts[entry->getComponent<SkinComponent>().skin().id()];
            }
        }

        std::optional<SkinId> selectedSkin;
        if (const std::optional<SceneObjectId> selected = m_scene.selectedObject();
            selected && m_scene.contains(*selected) && m_scene.getSceneObject(*selected).hasComponent<SkinComponent>())
        {
            selectedSkin = m_scene.getSceneObject(*selected).getComponent<SkinComponent>().skin().id();
        }

        std::unordered_set<SkinId> displayed;
        for (const auto &entry : m_scene.sceneObjects())
        {
            SceneObject *object = entry.get();
            if (!object || !m_scene.contains(object->id()) || !object->hasComponent<SkinComponent>())
            {
                continue;
            }
            const Skin &skin = object->getComponent<SkinComponent>().skin();
            if (!displayed.insert(skin.id()).second)
            {
                continue;
            }
            const bool selected = selectedSkin == skin.id();
            const std::string label = object->name.empty() ? "Unnamed Skin" : object->name;
            const std::string objectId = toString(object->id());
            ImGui::PushID(objectId.c_str());
            if (ImGui::Selectable(label.c_str(), selected))
            {
                m_scene.selectHierarchyObject(object->id());
                m_scene.revealInHierarchy(object->id());
            }
            ImGui::SameLine();
            ImGui::TextDisabled("%zu joints, %zu mesh%s", skin.joints().size(), userCounts.at(skin.id()),
                                userCounts.at(skin.id()) == 1 ? "" : "es");
            ImGui::PopID();
        }

        if (displayed.empty())
        {
            ImGui::TextDisabled("No skins in the scene.");
        }

        const std::optional<SceneObjectId> selected = m_scene.selectedObject();
        const bool canEdit = selected && m_scene.contains(*selected) &&
                             m_scene.getSceneObject(*selected).hasComponent<SkinComponent>();
        const bool editing = m_stateController.isActive(kSkeletonEditState);
        ImGui::BeginDisabled(!canEdit && !editing);
        if (ImGui::Button(editing ? "Exit Pose Mode" : "Pose Skeleton"))
        {
            m_stateController.toggle(kSkeletonEditState);
        }
        ImGui::EndDisabled();
        if (editing)
        {
            if (!m_skeletonSelection)
            {
                ImGui::TextDisabled("Click a joint, or click a bone to choose an endpoint.");
            }
            else
            {
                const Skin &skin = m_scene.getSceneObject(*selected).getComponent<SkinComponent>().skin();
                if (const std::optional<SkeletonManipulationTarget> target =
                        SkeletonManipulationAdapter::resolve(m_scene, skin, *m_skeletonSelection))
                {
                    ImGui::Text(target->selectedThroughBone ? "Selected bone (child joint %u)"
                                                            : "Selected joint %u",
                                target->joint);
                    drawSkeletonTransformControls(m_scene.getSceneObject(target->object));
                }
            }
            ImGui::SeparatorText("Display");
            ImGui::SliderFloat("Bone size", &m_skeletonOverlayStyle.boneRadiusRatio, 0.005f, 0.1f,
                               "%.3f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("Joint size", &m_skeletonOverlayStyle.jointRadiusRatio, 0.015f, 0.2f,
                               "%.3f", ImGuiSliderFlags_Logarithmic);
        }
        ImGui::End();
    }

    void drawSkeletonEndpointPopup()
    {
        if (m_openSkeletonEndpointPopup)
        {
            ImGui::OpenPopup("Select Bone Endpoint");
            m_openSkeletonEndpointPopup = false;
        }
        if (!ImGui::BeginPopup("Select Bone Endpoint"))
        {
            if (!ImGui::IsPopupOpen("Select Bone Endpoint")) m_pendingSkeletonBone.reset();
            return;
        }

        const std::optional<SceneObjectId> selected = m_scene.selectedObject();
        const Skin *skin = selected && m_scene.contains(*selected) &&
                                   m_scene.getSceneObject(*selected).hasComponent<SkinComponent>()
                               ? &m_scene.getSceneObject(*selected).getComponent<SkinComponent>().skin()
                               : nullptr;
        const SkeletonBone *bone = nullptr;
        std::optional<SkeletonTopology> topology;
        if (skin && m_pendingSkeletonBone && m_pendingSkeletonBone->skin == skin->id())
        {
            topology.emplace(m_scene, *skin);
            bone = topology->find(*m_pendingSkeletonBone);
        }
        if (!bone)
        {
            ImGui::TextDisabled("The selected bone is no longer available.");
            if (ImGui::Button("Close")) ImGui::CloseCurrentPopup();
            ImGui::EndPopup();
            return;
        }

        ImGui::TextUnformatted("Select the joint to pose:");
        if (ImGui::Button(("Parent joint " + std::to_string(bone->parentJoint)).c_str()))
        {
            selectSkeletonElement(JointRef{skin->id(), bone->parentJoint});
            m_pendingSkeletonBone.reset();
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button(("Child joint " + std::to_string(bone->childJoint)).c_str()))
        {
            selectSkeletonElement(JointRef{skin->id(), bone->childJoint});
            m_pendingSkeletonBone.reset();
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    void drawSkeletonTransformControls(SceneObject &joint)
    {
        TransformComponent &component = joint.getComponent<TransformComponent>();
        glm::vec3 position = component.transform().position();
        glm::vec3 rotation = component.transform().eulerDegrees();

        ImGui::SeparatorText("Local Transform");
        const glm::vec3 positionBeforeFrame = position;
        if (ImGui::DragFloat3("Position##SkeletonJoint", &position.x, 0.01f))
        {
            component.setPosition(position);
        }
        if (ImGui::IsItemActivated())
        {
            beginSkeletonPoseEdit();
            m_skeletonPositionBeforeEdit = positionBeforeFrame;
        }
        if (ImGui::IsItemDeactivatedAfterEdit() && m_skeletonPositionBeforeEdit)
        {
            const glm::vec3 delta = component.transform().position() - *m_skeletonPositionBeforeEdit;
            if (glm::dot(delta, delta) > 0.0f)
            {
                CommandTransaction transaction(m_commandManager);
                m_commandManager.appendCommandWithoutExecuting(
                    std::make_unique<TranslateSceneObjectCommand>(joint, delta));
                onTransformCommitted(joint, AnimationTargetProperty::Translation);
            }
            m_skeletonPositionBeforeEdit.reset();
        }

        const glm::quat rotationBeforeFrame = component.transform().rotation();
        if (ImGui::DragFloat3("Rotation (Degrees)##SkeletonJoint", &rotation.x, 0.1f))
        {
            component.setEulerDegrees(rotation);
        }
        if (ImGui::IsItemActivated())
        {
            beginSkeletonPoseEdit();
            m_skeletonRotationBeforeEdit = rotationBeforeFrame;
        }
        if (ImGui::IsItemDeactivatedAfterEdit() && m_skeletonRotationBeforeEdit)
        {
            const glm::quat after = component.transform().rotation();
            if (glm::abs(glm::dot(*m_skeletonRotationBeforeEdit, after)) < 1.0f - 1e-6f)
            {
                CommandTransaction transaction(m_commandManager);
                m_commandManager.appendCommandWithoutExecuting(
                    std::make_unique<RotateSceneObjectCommand>(joint, *m_skeletonRotationBeforeEdit, after));
                onTransformCommitted(joint, AnimationTargetProperty::Rotation);
            }
            m_skeletonRotationBeforeEdit.reset();
        }
    }

    void drawScenePanel(SceneDocument &document)
    {
        const float hierarchyHeight = ImGui::GetContentRegionAvail().y * 0.55f;
        ImGui::BeginChild("SceneHierarchy", ImVec2(0.0f, hierarchyHeight),
                          ImGuiChildFlags_Borders | ImGuiChildFlags_ResizeY);
        ImGui::SeparatorText("Hierarchy");
        if (!document.persistenceError().empty())
        {
            ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "Scene file error: %s",
                               document.persistenceError().c_str());
        }
        ImGui::Separator();
        drawSceneCreationButtons(document);
        if (!document.importError().empty())
        {
            ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "Import failed: %s", document.importError().c_str());
        }
        ImGui::Separator();
        m_scene.onHierarchyGUI(m_context);
        ImGui::EndChild();

        ImGui::BeginChild("SceneInspector", ImVec2(0.0f, 0.0f), ImGuiChildFlags_Borders);
        ImGui::SeparatorText("Inspector");
        m_scene.onInspectorGUI(m_context);
        ImGui::EndChild();
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
            const bool posing = m_stateController.isActive(kSkeletonEditState);
            const bool canTransform = posing ? m_skeletonTranslationHandler.target() != nullptr
                                             : m_transformController.target() != nullptr;
            ImGui::BeginDisabled(!canTransform);
            if (ImGui::RadioButton("Translate", m_transformController.tool() == TransformTool::Translate))
            {
                m_transformController.setTool(TransformTool::Translate);
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Rotate", m_transformController.tool() == TransformTool::Rotate))
            {
                m_transformController.setTool(TransformTool::Rotate);
            }
            if (!posing)
            {
                ImGui::SameLine();
                if (ImGui::RadioButton("Scale", m_transformController.tool() == TransformTool::Scale))
                {
                    m_transformController.setTool(TransformTool::Scale);
                }
            }
            ImGui::EndDisabled();
        }
        ImGui::End();
    }

    void drawAnimationWindow()
    {
        AnimationLibrary &library = m_sceneManager.animations();
        if (m_selectedAnimation && !library.contains(*m_selectedAnimation))
        {
            m_selectedAnimation.reset();
            m_animationSeconds = 0.0f;
        }

        if (!ImGui::Begin("Animation"))
        {
            ImGui::End();
            return;
        }

        ImGui::BeginChild("AnimationClips", ImVec2(180.0f, 0.0f), ImGuiChildFlags_Borders);
        if (ImGui::Button("Add Animation", ImVec2(-FLT_MIN, 0.0f)))
        {
            size_t suffix = library.size() + 1;
            std::string name;
            do
            {
                name = "Animation " + std::to_string(suffix++);
            } while (std::ranges::any_of(library.clips(), [&](const AnimationClip &clip) {
                return clip.name() == name;
            }));
            const AnimationClipHandle handle =
                library.add(AnimationClip(std::move(name), {}, 1.0f));
            selectAnimation(handle);
        }
        ImGui::Separator();
        std::optional<AnimationClipHandle> renameRequested;
        for (AnimationClipHandle handle = 0; handle < library.size(); ++handle)
        {
            const bool selected = m_deleter.selected(animationDeletionKey(handle));
            ImGui::PushID(static_cast<int>(handle));
            if (ImGui::Selectable(library.get(handle).name().c_str(), selected))
            {
                selectAnimation(handle);
            }
            if (ImGui::BeginPopupContextItem())
            {
                if (ImGui::MenuItem("Rename"))
                {
                    renameRequested = handle;
                }
                ImGui::EndPopup();
            }
            ImGui::PopID();
        }
        if (library.empty()) {
            ImGui::TextDisabled("No animation clips.");
        }
        if (ImGui::IsWindowHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Left) &&
            !ImGui::IsAnyItemHovered())
        {
            m_scene.clearHierarchySelection();
        }
        if (renameRequested)
        {
            m_renamingAnimation = renameRequested;
            m_animationRenameBuffer = library.get(*renameRequested).name();
            ImGui::OpenPopup("Rename Animation");
        }
        drawAnimationRenamePopup(library);
        ImGui::EndChild();
        ImGui::SameLine();

        ImGui::BeginChild("AnimationClipEditor", ImVec2(0.0f, 0.0f), ImGuiChildFlags_Borders);
        if (!m_selectedAnimation)
        {
            ImGui::TextDisabled("Select an animation clip");
        }
        else
        {
            AnimationClip &clip = library.get(*m_selectedAnimation);
            ImGui::TextUnformatted(clip.name().c_str());
            float editedDuration = clip.durationSeconds();
            ImGui::SetNextItemWidth(160.0f);
            if (ImGui::DragFloat("Duration (seconds)", &editedDuration, 0.05f, 0.0f, FLT_MAX, "%.3f"))
            {
                clip.setDurationSeconds(std::max(0.0f, editedDuration));
                m_animationSeconds = std::min(m_animationSeconds, clip.durationSeconds());
            }
            if (ImGui::IsItemActivated())
            {
                m_sceneManager.animationSystem().stop(*m_selectedAnimation);
            }
            if (ImGui::Checkbox("Auto Key", &m_autoKey) && m_autoKey)
            {
                m_sceneManager.animationSystem().stopAll();
                m_animationPlaybackWarning.clear();
            }
            ImGui::SameLine();
            if (m_sceneManager.animationSystem().isPlaying(*m_selectedAnimation))
            {
                if (ImGui::Button("Stop")) m_sceneManager.animationSystem().stop(*m_selectedAnimation);
            }
            else if (ImGui::Button("Play"))
            {
                const AnimationPlayResult result = m_sceneManager.animationSystem().play(*m_selectedAnimation);
                if (!result.started && result.conflict && result.conflictingClip)
                {
                    const char *property = result.conflict->property == AnimationTargetProperty::Translation ? "Position" :
                                           result.conflict->property == AnimationTargetProperty::Rotation ? "Rotation" : "Scale";
                    std::string target = toString(result.conflict->target);
                    if (m_scene.contains(result.conflict->target))
                    {
                        const SceneObject &object = m_scene.getSceneObject(result.conflict->target);
                        if (!object.name.empty()) target = object.name;
                    }
                    m_animationPlaybackWarning = "Cannot play '" + clip.name() + "': " + property +
                        " on '" + target + "' is already controlled by '" +
                        library.get(*result.conflictingClip).name() + "'.";
                }
                else m_animationPlaybackWarning.clear();
            }
            if (!m_animationPlaybackWarning.empty())
                ImGui::TextColored(ImVec4(1.0f, 0.65f, 0.2f, 1.0f), "%s", m_animationPlaybackWarning.c_str());
            if (const std::optional<float> playbackTime =
                    m_sceneManager.animationSystem().playbackTime(*m_selectedAnimation))
            {
                m_animationSeconds = *playbackTime;
            }
            const float duration = clip.durationSeconds();
            float progress = duration > 0.0f ? m_animationSeconds / duration : 0.0f;

            ImGui::SeparatorText("Tracks");

            if (clip.tracks().empty())
            {
                ImGui::TextDisabled("No tracks. Enable Auto Key and transform an object to add one.");
            }
            else
            {
                ImVec2 childSize = ImVec2(-FLT_MIN, 200.0f);
                ImGuiChildFlags childFlags = ImGuiChildFlags_Borders | ImGuiChildFlags_ResizeY;
                ImGui::BeginChild("AnimationTracks", childSize, childFlags);
                m_selectedAnimationTrack = std::min(m_selectedAnimationTrack, clip.tracks().size() - 1);
                for (size_t index = 0; index < clip.tracks().size(); ++index)
                {
                    const std::string label = trackLabel(clip.tracks()[index]);
                    if (ImGui::Selectable(label.c_str(),
                                          m_deleter.selected(animationTrackDeletionKey(
                                              *m_selectedAnimation, index))))
                    {
                        m_selectedAnimationTrack = index;
                        selectTrack(*m_selectedAnimation, index);
                    }
                }
                ImGui::EndChild();
                ImGui::SeparatorText("Track Editor");
                ImGui::BeginChild("TrackEditor", childSize, childFlags);
                AnimationChannel &channel = clip.tracks()[m_selectedAnimationTrack];
                enum class Action { None, Add, Edit, Delete, Apply, Cancel };
                Action action = Action::None;
                std::optional<size_t> keyAtPlayhead;
                std::visit([&](auto &track) {
                    std::vector<float> keys;
                    for (const auto &key : track.keyframes())
                    {
                        keys.push_back(duration > 0.0f ? key.seconds / duration : 0.0f);
                    }
                    const auto result = gui::animationTrack("##globalAnimationTrack", &progress, keys,
                        &m_animationTrackViewCenter, &m_animationTrackViewHalfWidth, 0.01f,
                        ImVec2(ImGui::GetContentRegionAvail().x, 42.0f));
                    if (result.progressChanged)
                    {
                        m_animationSeconds = progress * duration;
                        if (!m_sceneManager.animationSystem().seek(*m_selectedAnimation, m_animationSeconds))
                            applyAnimation(clip, m_animationSeconds);
                        refreshSkeletonOverlay();
                    }
                    for (size_t index = 0; index < track.keyframes().size(); ++index)
                    {
                        if (std::abs(track.keyframes()[index].seconds - m_animationSeconds) <= 1e-4f)
                            keyAtPlayhead = index;
                    }
                    ImGui::BeginDisabled(keyAtPlayhead.has_value());
                    if (ImGui::Button("Add Keyframe")) action = Action::Add;
                    ImGui::EndDisabled(); ImGui::SameLine();
                    ImGui::BeginDisabled(!keyAtPlayhead);
                    if (ImGui::Button("Delete Keyframe")) action = Action::Delete;
                    ImGui::EndDisabled();
                }, channel);
                switch (action)
                {
                case Action::Add:
                    if (m_scene.contains(std::visit([](const auto &track) { return track.target(); }, channel)))
                    {
                        SceneObject &target = m_scene.getSceneObject(
                            std::visit([](const auto &track) { return track.target(); }, channel));
                        const AnimationChannel before = channel;
                        setCurrentTransformKey(channel, target, m_animationSeconds);
                        recordTrackEdit(m_selectedAnimationTrack, before, channel);
                    }
                    break;
                case Action::Delete:
                {
                    const AnimationChannel before = channel;
                    std::visit([&](auto &track) { track.removeKeyframe(m_animationSeconds); }, channel);
                    recordTrackEdit(m_selectedAnimationTrack, before, channel);
                    break;
                }
                default: break;
                }
                ImGui::EndChild();
            }
        }
        ImGui::EndChild();

        drawAddTrackPopup();
        ImGui::End();
    }

private:
    void drawAnimationRenamePopup(AnimationLibrary &library)
    {
        if (!m_renamingAnimation)
        {
            return;
        }

        bool closePopup = false;
        if (ImGui::BeginPopupModal("Rename Animation", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
        {
            if (!library.contains(*m_renamingAnimation))
            {
                closePopup = true;
            }
            else
            {
                if (ImGui::IsWindowAppearing())
                {
                    ImGui::SetKeyboardFocusHere();
                }
                const bool submitted = gui::inputText("Name", m_animationRenameBuffer,
                                                      ImGuiInputTextFlags_EnterReturnsTrue);
                if (submitted || ImGui::Button("Rename"))
                {
                    library.get(*m_renamingAnimation).setName(m_animationRenameBuffer);
                    closePopup = true;
                }
                ImGui::SameLine();
                if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape))
                {
                    closePopup = true;
                }
            }

            if (closePopup)
            {
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }

        if (closePopup)
        {
            m_renamingAnimation.reset();
            m_animationRenameBuffer.clear();
        }
    }

    // Import and the built-in primitives, plus the two objects worth creating from nothing. Light is
    // capped by what the light SSBO can hold, so the button reports the limit rather than silently
    // refusing.
    void drawSceneCreationButtons(SceneDocument &document)
    {
        if (ImGui::Button("Import")) ImGui::OpenPopup("import_list");
        if (ImGui::BeginPopup("import_list"))
        {
            if (ImGui::MenuItem("Import Mesh...")) document.openImportDialog();
            if (ImGui::MenuItem("Import Fluid Simulation...")) document.openImportFluidCacheDialog();
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Import a baked Blender liquid cache directory.");
            if (ImGui::MenuItem("Import Smoke Simulation...")) document.openImportSmokeCacheDialog();
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Import a baked Blender gas cache directory (density, temperature and flame).");
            ImGui::EndPopup();
        }
        ImGui::SameLine();

        if (ImGui::Button("Add Primitive"))
        {
            ImGui::OpenPopup("add_primitive_list");
        }
        if (ImGui::BeginPopup("add_primitive_list"))
        {
            for (const auto &[label, fileName] : kPrimitives)
            {
                if (ImGui::MenuItem(label))
                {
                    document.importPrimitive(fileName, label);
                }
            }
            ImGui::EndPopup();
        }
        ImGui::SameLine();

        const uint32_t maxLights   = m_sceneManager.gpu().maxLights();
        const bool     lightsAtCap = m_sceneManager.numLights() >= maxLights;
        ImGui::BeginDisabled(lightsAtCap);
        if (ImGui::Button("Add Light"))
        {
            SceneObject &lightObject = m_scene.createSceneObject();
            lightObject.name         = "Light";
            lightObject.addComponent<TransformComponent>();
            lightObject.addComponent<Light>(PointLight{});
        }
        ImGui::EndDisabled();
        if (lightsAtCap && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        {
            ImGui::SetTooltip("The scene already contains the maximum of %u lights", maxLights);
        }

        // Added after the tooltip above, which reads the last item and would otherwise attach to
        // this button instead of the disabled one it explains.
        ImGui::SameLine();
        if (ImGui::Button("Add Scene Object"))
        {
            // Empty but for a transform, which every other component reads as its object's place in
            // the world — so the object is ready to receive one straight away. What it becomes is
            // then decided in the Inspector.
            SceneObject &emptyObject = m_scene.createSceneObject();
            emptyObject.name         = "Scene Object";
            emptyObject.addComponent<TransformComponent>();
        }
    }

    void selectAnimation(AnimationClipHandle handle)
    {
        clearSkeletonSelection();
        m_selectedAnimation = handle;
        m_scene.clearHierarchySelection();
        m_deleter.select(animationDeletionKey(handle), [this, handle] {
            AnimationLibrary &library = m_sceneManager.animations();
            if (!library.contains(handle)) return;
            m_sceneManager.animationSystem().remove(handle);
            if (m_selectedAnimation == handle)
            {
                m_selectedAnimation.reset();
                m_selectedAnimationTrack = 0;
                m_animationSeconds = 0.0f;
                m_animationPlaybackWarning.clear();
            }
            else if (m_selectedAnimation && *m_selectedAnimation > handle)
            {
                --*m_selectedAnimation;
            }
        });
        m_selectedAnimationTrack = 0;
        m_animationSeconds = 0.0f;
    }

    static AnimationTargetProperty channelProperty(const AnimationChannel &channel)
    {
        return std::visit([](const auto &track) { return track.property(); }, channel);
    }

    std::string trackLabel(const AnimationChannel &channel) const
    {
        return std::visit([&](const auto &track) {
            std::string objectName = "Missing object";
            if (m_scene.contains(track.target()))
            {
                const SceneObject &target = m_scene.getSceneObject(track.target());
                objectName = target.name.empty() ? "Scene Object " + toString(target.id()) : target.name;
            }
            const char *property = track.property() == AnimationTargetProperty::Translation ? "Position" :
                                   track.property() == AnimationTargetProperty::Rotation ? "Rotation" : "Scale";
            return objectName + " (" + property + ")";
        }, channel);
    }

    void selectTrack(AnimationClipHandle handle, size_t index)
    {
        m_scene.clearHierarchySelection();
        m_deleter.select(animationTrackDeletionKey(handle, index), [this, handle, index] {
            AnimationLibrary &library = m_sceneManager.animations();
            if (!library.contains(handle) || index >= library.get(handle).tracks().size()) return;
            m_sceneManager.animationSystem().removeTrack(handle, index);
            if (m_selectedAnimation == handle)
            {
                const size_t trackCount = library.get(handle).tracks().size();
                m_selectedAnimationTrack = trackCount == 0 ? 0 : std::min(index, trackCount - 1);
            }
            clearSkeletonSelection();
        });
        const AnimationChannel &channel = m_sceneManager.animations().get(handle).tracks()[index];
        std::visit([&](const auto &track) {
            if (!m_scene.contains(track.target())) return;
            m_scene.selectObject(track.target());
            m_transformController.setTool(track.property() == AnimationTargetProperty::Translation ? TransformTool::Translate :
                                          track.property() == AnimationTargetProperty::Rotation ? TransformTool::Rotate : TransformTool::Scale);
        }, channel);
    }

    void applyAnimation(AnimationClip &clip, float seconds)
    {
        for (const AnimationChannel &channel : clip.tracks())
            std::visit([&](const auto &track) { track.apply(m_scene, seconds); }, channel);
    }

    void onTransformCommitted(SceneObject &object, AnimationTargetProperty property)
    {
        if (!m_autoKey || !m_selectedAnimation || !object.hasComponent<TransformComponent>()) return;
        AnimationClip &clip = m_sceneManager.animations().get(*m_selectedAnimation);
        for (size_t index = 0; index < clip.tracks().size(); ++index)
        {
            AnimationChannel &channel = clip.tracks()[index];
            const bool matches = std::visit([&](const auto &track) {
                return track.target() == object.id() && track.property() == property;
            }, channel);
            if (!matches) continue;
            // Recorded, so that undoing the transform that triggered this takes the key with it.
            // Both land in the transaction the gizmo handler or the inspector opened around the
            // commit, which is what makes the pair one undo.
            const AnimationChannel before = channel;
            setCurrentTransformKey(channel, object, m_animationSeconds);
            recordTrackEdit(index, before, channel);
            m_selectedAnimationTrack = index;
            return;
        }
        m_pendingTrack = PendingTrack{object.id(), property, m_commandManager.currentTransactionId()};
        m_openAddTrackPopup = true;
    }

    // Records a keyframe edit the Track Editor's buttons have already applied. Both buttons are
    // disabled in the case where they would leave the track as it was — Add while a key sits on the
    // playhead, Delete while none does — so whatever reaches here is a real edit.
    void recordTrackEdit(size_t track, AnimationChannel before, const AnimationChannel &after)
    {
        if (!m_selectedAnimation)
        {
            return;
        }
        m_commandManager.appendCommandWithoutExecuting(std::make_unique<EditAnimationTrackCommand>(
            m_sceneManager.animations(), *m_selectedAnimation, track, std::move(before), after));
    }

    static void setCurrentTransformKey(AnimationChannel &channel, SceneObject &object, float seconds)
    {
        TransformComponent &transform = object.getComponent<TransformComponent>();
        std::visit([&](auto &track) {
            using Track = std::decay_t<decltype(track)>;
            if constexpr (std::is_same_v<Track, TranslationTrack>) track.setKeyframe(seconds, transform.transform().position());
            else if constexpr (std::is_same_v<Track, RotationTrack>) track.setKeyframe(seconds, transform.transform().rotation());
            else track.setKeyframe(seconds, transform.transform().scale());
        }, channel);
    }

    void drawAddTrackPopup()
    {
        if (m_openAddTrackPopup) { ImGui::OpenPopup("Add Animation Track?"); m_openAddTrackPopup = false; }
        if (!ImGui::BeginPopupModal("Add Animation Track?", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) return;
        if (!m_pendingTrack || !m_selectedAnimation || !m_scene.contains(m_pendingTrack->target))
        {
            ImGui::CloseCurrentPopup(); m_pendingTrack.reset(); ImGui::EndPopup(); return;
        }
        SceneObject &object = m_scene.getSceneObject(m_pendingTrack->target);
        const char *property = m_pendingTrack->property == AnimationTargetProperty::Translation ? "Position" :
                               m_pendingTrack->property == AnimationTargetProperty::Rotation ? "Rotation" : "Scale";
        ImGui::Text("Add %s track for %s?", property, object.name.c_str());
        if (ImGui::Button("Add Track"))
        {
            // Built and seeded here, then handed to the command, so that adding the track is the
            // command's own doing and redoing it produces the same track with the same first key.
            AnimationChannel channel =
                m_pendingTrack->property == AnimationTargetProperty::Translation
                    ? AnimationChannel(TranslationTrack(object.id()))
                    : m_pendingTrack->property == AnimationTargetProperty::Rotation
                          ? AnimationChannel(RotationTrack(object.id()))
                          : AnimationChannel(ScaleTrack(object.id()));
            setCurrentTransformKey(channel, object, m_animationSeconds);
            auto command = std::make_unique<AddAnimationTrackCommand>(
                m_sceneManager.animationSystem(), m_sceneManager.animations(), *m_selectedAnimation,
                std::move(channel));
            if (m_pendingTrack->originatingHistoryEntry)
                m_commandManager.executeCommandAndAppendTo(*m_pendingTrack->originatingHistoryEntry,
                                                           std::move(command));
            else
                m_commandManager.executeCommand(std::move(command));
            const size_t trackCount = m_sceneManager.animations().get(*m_selectedAnimation).tracks().size();
            m_selectedAnimationTrack = trackCount == 0 ? 0 : trackCount - 1;
            m_pendingTrack.reset(); ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) { m_pendingTrack.reset(); ImGui::CloseCurrentPopup(); }
        ImGui::EndPopup();
    }

    uint32_t readSkeletonPickingId()
    {
        return m_skeletonPicker.readIdAtCursor(m_viewer, OverlayGeometryPass::imageName);
    }

    bool handleSkeletonPointer(const PointerButtonEvent &event)
    {
        if (event.button != GLFW_MOUSE_BUTTON_LEFT)
        {
            return false;
        }
        const bool consumed = m_skeletonPicker.consumes(event.action);
        if (!m_skeletonPicker.clickCompleted(m_viewer, event.action))
        {
            // A press is consumed so the release that ends it reaches this layer too; anything
            // else (a release with no press of ours) belongs to whoever did handle the press.
            return consumed;
        }

        const uint32_t pickingId = readSkeletonPickingId();
        if (pickingId == ViewportPicker::kNoData)
        {
            return true;
        }
        if (pickingId == 0)
        {
            m_skeletonSelection.reset();
            m_pendingSkeletonBone.reset();
            clearSkeletonControlEdits();
            return true;
        }
        if (const SkeletonSelection *selection = m_skeletonOverlay.selectionForPickingId(pickingId))
        {
            if (const BoneRef *bone = std::get_if<BoneRef>(selection))
            {
                m_pendingSkeletonBone = *bone;
                m_openSkeletonEndpointPopup = true;
            }
            else
            {
                selectSkeletonElement(*selection);
                m_pendingSkeletonBone.reset();
            }
        }
        return true;
    }

    // Each state registers its own presentation *and* its own behavior, so neither update() nor the
    // gizmo arbitration below needs to know that ARAP, vertex editing, or analysis exist. Once
    // feature tools register themselves (EditorTool), these closures move out with them.
    void registerEditorStates()
    {
        m_stateController.registerState({
            .id           = std::string(kViewState),
            .presentation = {.objectSelectionActive = true},
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

        m_stateController.registerState({
            .id           = std::string(kSkeletonEditState),
            .presentation = {.skinningEnabled = true},
            .onExit       = [this] {
                m_skeletonTranslationHandler.setTarget(nullptr);
                m_skeletonRotationHandler.setTarget(nullptr);
                m_skeletonSelection.reset();
                m_pendingSkeletonBone.reset();
                m_openSkeletonEndpointPopup = false;
                clearSkeletonControlEdits();
                m_skeletonPicker.reset();
                m_skeletonHoverCursor.reset();
                m_lastSkeletonHoverReadbackSeconds = 0.0;
                m_skeletonOverlay = {};
                m_renderBridge.setHoveredOverlay(~0u);
                m_renderBridge.setOverlayGeometry({});
            },
            .update       = [this](const EditorFrameContext &) {
                const std::optional<SceneObjectId> selected = m_scene.selectedObject();
                if (!selected || !m_scene.contains(*selected) ||
                    !m_scene.getSceneObject(*selected).hasComponent<SkinComponent>())
                {
                    m_stateController.activateDefault();
                    return;
                }
                const Skin &skin = m_scene.getSceneObject(*selected).getComponent<SkinComponent>().skin();
                if (m_skeletonSelection && !std::visit([&](const auto &selection) {
                        return selection.skin == skin.id();
                    }, *m_skeletonSelection))
                {
                    m_skeletonSelection.reset();
                }
                SceneObject *jointTarget = selectedSkeletonTarget(skin);
                if (m_skeletonSelection && !jointTarget)
                {
                    m_skeletonSelection.reset();
                }
                m_skeletonTranslationHandler.setTarget(jointTarget);
                m_skeletonRotationHandler.setTarget(jointTarget);
                m_skeletonOverlay =
                    buildSkeletonOverlay(m_scene, *selected, m_skeletonSelection, m_skeletonOverlayStyle);
                m_renderBridge.setOverlayGeometry(m_skeletonOverlay.instances);

                double mouseX = 0.0, mouseY = 0.0;
                m_viewer.input().getMousePos(mouseX, mouseY);
                const glm::dvec2 cursor(mouseX, mouseY);
                const double now = glfwGetTime();
                const bool cursorMoved = !m_skeletonHoverCursor || m_skeletonHoverCursor->x != cursor.x ||
                                         m_skeletonHoverCursor->y != cursor.y;
                // Hover is visual feedback, not a precision action. Cap its synchronous GPU
                // readback rate while preserving the unthrottled click read used for selection.
                if (cursorMoved && now - m_lastSkeletonHoverReadbackSeconds >= 1.0 / 30.0)
                {
                    m_skeletonHoverCursor = cursor;
                    m_lastSkeletonHoverReadbackSeconds = now;
                    const uint32_t hovered = readSkeletonPickingId();
                    m_renderBridge.setHoveredOverlay(
                        m_skeletonOverlay.selectionForPickingId(hovered) ? hovered : ~0u);
                }
            },
            .gizmoRequest = [this](const EditorFrameContext &) -> GizmoRequest {
                switch (m_transformController.tool())
                {
                    case TransformTool::Translate:
                        if (const SceneObject *target = m_skeletonTranslationHandler.target())
                        {
                            return TranslateGizmoRequest{
                                .origin  = glm::vec3(target->worldMatrix()[3]),
                                .handler = &m_skeletonTranslationHandler,
                            };
                        }
                        break;
                    case TransformTool::Rotate:
                        if (const SceneObject *target = m_skeletonRotationHandler.target())
                        {
                            return RotateGizmoRequest{
                                .worldMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(target->worldMatrix()[3])) *
                                               glm::mat4_cast(target->worldRotation()),
                                .handler     = &m_skeletonRotationHandler,
                            };
                        }
                        break;
                    default: break;
                }
                return {};
            },
            .handleInput = [this](const PointerButtonEvent &event) { return handleSkeletonPointer(event); },
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
        m_connections.push_back(m_scene.registerHierarchySelectionChangedCallback(
            [this](std::optional<SceneObjectId> selected) {
                if (!selected)
                {
                    m_deleter.clear();
                    return;
                }
                const SceneObjectId id = *selected;
                m_deleter.select(
                    sceneObjectDeletionKey(id),
                    [this, id]() -> std::optional<std::string> {
                        const Scene::DestructionCheck check = m_scene.checkSceneObjectDestruction(id);
                        return check.allowed ? std::nullopt : std::optional<std::string>(check.reason);
                    },
                    [this, id] { m_scene.destroySceneObject(id); });
            }));
        m_connections.push_back(
            m_scene.registerSelectionChangedCallback([this](std::optional<SceneObjectId> selected) {
                if (!selected)
                {
                    // Nothing to transform and nothing to edit: the same teardown the destruction of a
                    // selected object performs, minus the destruction.
                    m_transformWindowOpen = false;
                    m_transformController.setSelectedTarget(nullptr);
                    if (m_stateController.active().presentation.vertexSelectionActive)
                    {
                        m_stateController.activateDefault();
                    }
                    if (m_stateController.isActive(kSkeletonEditState))
                    {
                        m_stateController.activateDefault();
                    }
                    return;
                }
                SceneObject &object   = m_scene.getSceneObject(*selected);
                m_transformWindowOpen = true;
                m_transformController.setSelectedTarget(object.hasComponent<TransformComponent>() ? &object : nullptr);
                if (m_stateController.isActive(kSkeletonEditState))
                {
                    if (object.hasComponent<SkinComponent>())
                    {
                        m_skeletonSelection.reset();
                        m_skeletonOverlay =
                            buildSkeletonOverlay(m_scene, object.id(), std::nullopt, m_skeletonOverlayStyle);
                        m_renderBridge.setOverlayGeometry(m_skeletonOverlay.instances);
                    }
                    else
                    {
                        m_stateController.activateDefault();
                    }
                }
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
        m_inputRouter.addButtonLayer("active editor state", [this](const PointerButtonEvent &event,
                                                                    const EditorInputContext &) {
            return m_stateController.handleInput(event);
        });

        // Generic vertex picking, available to any state whose presentation asks for it.
        m_inputRouter.addButtonLayer("vertex selection", [this](const PointerButtonEvent &event,
                                                                 const EditorInputContext &context) {
            if (event.button != GLFW_MOUSE_BUTTON_LEFT || !context.presentation.vertexSelectionActive)
            {
                return false;
            }
            m_selectionManager.mouseButtonCallback(event.button, event.action, event.shift, event.ctrl, event.alt);
            return true;
        });

        // Lowest-priority viewport interaction: select the frontmost rendered scene object. A small
        // gesture threshold prevents an orbit/drag ending over geometry from becoming a click.
        m_inputRouter.addButtonLayer("scene object selection", [this](const PointerButtonEvent &event,
                                                                       const EditorInputContext &context) {
            if (!context.presentation.objectSelectionActive)
            {
                m_objectPicker.reset();
                return false;
            }
            if (event.button != GLFW_MOUSE_BUTTON_LEFT)
            {
                return false;
            }

            const bool consumed = m_objectPicker.consumes(event.action);
            if (!m_objectPicker.clickCompleted(m_viewer, event.action))
            {
                return consumed;
            }

            const uint32_t pickingId = m_objectPicker.readIdAtCursor(m_viewer, ObjectPickingPass::imageName);
            if (pickingId == ViewportPicker::kNoData)
            {
                // Nothing was read at all. Not the same answer as "nothing is there", so the
                // selection stands.
                return true;
            }
            if (pickingId == 0)
            {
                // The picking target clears to 0, so this is empty space: deselect, as clicking the
                // background does in Blender.
                m_scene.clearSelection();
                return true;
            }
            const auto &objects = m_sceneManager.gpu().geometryObjects();
            if (pickingId <= objects.size())
            {
                SceneObject *object = objects[pickingId - 1];
                if (object && m_scene.contains(object->id()))
                {
                    m_scene.selectHierarchyObject(object->id());
                    // The click says nothing about where the object sits in the tree, so bring the
                    // panel to it rather than leaving the new selection highlighted out of sight.
                    m_scene.revealInHierarchy(object->id());
                }
            }
            return true;
        });
    }

    void registerInputCallbacks()
    {
        m_connections.push_back(m_viewer.input().onMouseButton(
            [this](int button, int action, bool shift, bool ctrl, bool alt) {
            m_inputRouter.routeButton(
                {
                    .button = button,
                    .action = action,
                    .shift  = shift,
                    .ctrl   = ctrl,
                    .alt    = alt,
                },
                {
                    .activeState = m_stateController.activeId(),
                    .presentation = m_stateController.active().presentation,
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
        // The camera reset used to own R, which the transform shortcuts below now take, as in Blender.
        m_shortcuts.add({.key = GLFW_KEY_HOME}, [this] {
            SceneObject *camera = m_sceneManager.gpu().camera();
            if (camera)
            {
                if (CameraController *controller = camera->findComponent<CameraController>())
                {
                    controller->resetTransformation();
                }
            }
        });

        // G / R / S pick the gizmo the Transform window's radio buttons pick, so they stay disabled
        // in the same case the buttons are: nothing selected to transform.
        for (const auto &[key, tool] : {std::pair{GLFW_KEY_G, TransformTool::Translate},
                                        std::pair{GLFW_KEY_R, TransformTool::Rotate},
                                        std::pair{GLFW_KEY_S, TransformTool::Scale}})
        {
            m_shortcuts.add({.key = key}, [this, tool = tool] {
                const bool posing = m_stateController.isActive(kSkeletonEditState);
                const bool targetSelected = posing ? m_skeletonTranslationHandler.target() != nullptr
                                                    : m_transformController.target() != nullptr;
                if (targetSelected && (!posing || tool != TransformTool::Scale))
                {
                    m_transformController.toggleTool(tool);
                }
            });
        }

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
            // Skeleton elements are the active selection in this mode. Bone/joint deletion is not
            // implemented, so Delete must not fall through to the stale hierarchy deletion target
            // and remove the mesh that merely represents the edited skin.
            if (m_stateController.isActive(kSkeletonEditState))
            {
                return;
            }
            const Deleter::EraseResult result = m_deleter.erase();
            if (result.status == Deleter::EraseResult::Status::Blocked)
            {
                m_errors.report("Cannot delete selection", result.message);
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

    SceneObject *selectedSkeletonTarget(const Skin &skin)
    {
        if (!m_skeletonSelection)
        {
            return nullptr;
        }
        const std::optional<SkeletonManipulationTarget> target =
            SkeletonManipulationAdapter::resolve(m_scene, skin, *m_skeletonSelection);
        if (!target)
        {
            return nullptr;
        }
        SceneObject &object = m_scene.getSceneObject(target->object);
        return object.hasComponent<TransformComponent>() ? &object : nullptr;
    }

    void beginSkeletonPoseEdit()
    {
        m_sceneManager.animationSystem().stopAll();
        m_animationPlaybackWarning = "Playback stopped to edit the pose.";
    }

    void clearSkeletonSelection()
    {
        m_skeletonSelection.reset();
        m_pendingSkeletonBone.reset();
        m_openSkeletonEndpointPopup = false;
        m_skeletonTranslationHandler.setTarget(nullptr);
        m_skeletonRotationHandler.setTarget(nullptr);
        clearSkeletonControlEdits();
        refreshSkeletonOverlay();
    }

    void refreshSkeletonOverlay()
    {
        if (!m_stateController.isActive(kSkeletonEditState))
        {
            return;
        }
        const std::optional<SceneObjectId> selected = m_scene.selectedObject();
        if (!selected || !m_scene.contains(*selected) ||
            !m_scene.getSceneObject(*selected).hasComponent<SkinComponent>())
        {
            return;
        }
        m_skeletonOverlay =
            buildSkeletonOverlay(m_scene, *selected, m_skeletonSelection, m_skeletonOverlayStyle);
        m_renderBridge.setOverlayGeometry(m_skeletonOverlay.instances);
    }

    void selectSkeletonElement(SkeletonSelection selection)
    {
        m_skeletonSelection = std::move(selection);
        clearSkeletonControlEdits();
        if (m_transformController.tool() != TransformTool::Translate &&
            m_transformController.tool() != TransformTool::Rotate)
        {
            m_transformController.setTool(TransformTool::Translate);
        }
    }

    void clearSkeletonControlEdits()
    {
        m_skeletonPositionBeforeEdit.reset();
        m_skeletonRotationBeforeEdit.reset();
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
    ErrorReporter                 &m_errors;
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
    SceneObjectDragHandler         m_skeletonTranslationHandler;
    SceneObjectRotationHandler     m_skeletonRotationHandler;
    GizmoController                m_gizmoController;
    EditorInputRouter              m_inputRouter;
    SceneObjectTransformController m_transformController;
    ViewportPicker                 m_objectPicker;
    ViewportPicker                 m_skeletonPicker;
    SkeletonOverlay                m_skeletonOverlay;
    SkeletonOverlayStyle           m_skeletonOverlayStyle;
    std::optional<SkeletonSelection> m_skeletonSelection;
    std::optional<BoneRef>         m_pendingSkeletonBone;
    std::optional<glm::vec3>       m_skeletonPositionBeforeEdit;
    std::optional<glm::quat>       m_skeletonRotationBeforeEdit;
    std::optional<glm::dvec2>      m_skeletonHoverCursor;
    double                         m_lastSkeletonHoverReadbackSeconds = 0.0;
    bool                           m_openSkeletonEndpointPopup = false;
    // Declared before m_context, which holds a reference to it.
    ComponentCatalog               m_componentCatalog;
    EditorContext                  m_context;
    ArapTool                       m_arapTool;
    LaplaceBeltramiTool            m_laplaceBeltramiTool;
    // Every registered tool, in panel order. Declared after the tools it points at.
    std::vector<EditorTool *> m_tools;
    bool                      m_transformWindowOpen = false;
    static std::string sceneObjectDeletionKey(SceneObjectId id)
    {
        return "scene:" + toString(id);
    }
    static std::string animationDeletionKey(AnimationClipHandle handle)
    {
        return "animation:" + std::to_string(handle);
    }
    static std::string animationTrackDeletionKey(AnimationClipHandle handle, size_t track)
    {
        return "animation-track:" + std::to_string(handle) + ":" + std::to_string(track);
    }

    Deleter                            m_deleter;
    std::optional<AnimationClipHandle> m_selectedAnimation;
    std::optional<AnimationClipHandle> m_renamingAnimation;
    std::string                        m_animationRenameBuffer;
    float                              m_animationSeconds = 0.0f;
    size_t                             m_selectedAnimationTrack = 0;
    float                              m_animationTrackViewCenter = 0.5f;
    float                              m_animationTrackViewHalfWidth = 0.5f;
    bool                               m_autoKey = false;
    std::string                        m_animationPlaybackWarning;
    struct PendingTrack
    {
        SceneObjectId target;
        AnimationTargetProperty property;
        std::optional<CommandManager::HistoryEntryId> originatingHistoryEntry;
    };
    std::optional<PendingTrack>        m_pendingTrack;
    bool                               m_openAddTrackPopup = false;
    // Declared last so every external callback disconnects before the objects it may call into.
    std::vector<CallbackConnection> m_connections;
};

EditorSession::EditorSession(Viewer &viewer, SceneManager &sceneManager, SceneObject &camera,
                             EditorRenderBridge &renderBridge, ErrorReporter &errors)
    : m_impl(std::make_unique<Impl>(viewer, sceneManager, camera, renderBridge, errors))
{}

EditorSession::~EditorSession() = default;

EditorContext &EditorSession::context() { return m_impl->context(); }
void           EditorSession::onSceneContentChanged() { m_impl->onSceneContentChanged(); }
void           EditorSession::drawFeaturePanel() { m_impl->drawFeaturePanel(); }
void           EditorSession::drawScenePanel(SceneDocument &document) { m_impl->drawScenePanel(document); }
void           EditorSession::drawTransformWindow() { m_impl->drawTransformWindow(); }
void           EditorSession::drawAnimationWindow() { m_impl->drawAnimationWindow(); }
void           EditorSession::drawSkinningWindow() { m_impl->drawSkinningWindow(); }
bool           EditorSession::allowsViewportNavigation() const { return m_impl->allowsViewportNavigation(); }

} // namespace lr
