// Target-binding behaviour of registered editor tools, driven the way EditorSession drives them: a
// vector<EditorTool *>, one EditorServices, and one notification per target change. Uses the real
// ArapTool rather than a stand-in, so what is verified is the feature's actual reaction to losing
// the mesh it was bound to.
//
// EditorSession itself needs a Viewer and a GPU-backed SceneManager, so the loop that issues these
// notifications is not exercised here — only that every tool reacts correctly once it is.

#include "core/app/InputHandler.hpp"
#include "core/editor/EditableMeshContext.hpp"
#include "core/editor/EditorShortcuts.hpp"
#include "core/editor/EditorStateController.hpp"
#include "core/editor/EditorTool.hpp"
#include "core/editor/VertexManager.hpp"
#include "core/editor/command/CommandManager.hpp"
#include "core/editor/selection/SelectionManager.hpp"
#include "core/loaders/MaterialStore.hpp"
#include "core/scene/Mesh.hpp"
#include "core/scene/MeshComponent.hpp"
#include "core/scene/MeshStore.hpp"
#include "core/scene/Scene.hpp"
#include "core/scene/TransformComponent.hpp"
#include "features/arap/ArapTool.hpp"

#include <cassert>
#include <string>
#include <vector>

namespace
{

constexpr std::string_view kViewState = "view";

// A quad built from two triangles — enough topology for a mesh to be a plausible edit target.
lr::Mesh makeQuad()
{
    lr::Mesh mesh;
    mesh.setTopology({{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}}, {0, 1, 2, 3},
                     {{0, 1, 2}, {0, 2, 3}});
    return mesh;
}

// Everything an editor tool needs, minus the parts that need a GPU. Mirrors how EditorSession
// composes them, so the tools under test see the same shapes they see in the real editor.
struct Harness
{
    Harness()
        : materialStore(1,
                        [] {
                            return lr::Material{};
                        }),
          firstHandle(meshStore.add(makeQuad())), secondHandle(meshStore.add(makeQuad())),
          first(makeMeshObject(firstHandle)), second(makeMeshObject(secondHandle)),
          vertexManager(first.getComponent<lr::MeshComponent>().mesh()),
          selectionManager(first.getComponent<lr::MeshComponent>().mesh().positions(),
                           first.getComponent<lr::TransformComponent>(), input),
          states([this](const lr::EditorStateDefinition &state) {
              presentation = state.presentation;
          })
    {
        states.registerState({.id = std::string(kViewState), .presentation = {}});
        states.setDefaultState(kViewState);
        states.activate(kViewState);
    }

    lr::SceneObject &makeMeshObject(lr::MeshHandle handle)
    {
        lr::SceneObject &object = scene.createSceneObject();
        object.addComponent<lr::TransformComponent>();
        object.addComponent<lr::MeshComponent>(handle, meshStore, std::vector<lr::MaterialHandle>{}, materialStore);
        return object;
    }

    static lr::EditableMeshContext contextFor(lr::SceneObject &object)
    {
        return {.object = object, .mesh = object.getComponent<lr::MeshComponent>().mesh()};
    }

    // Exactly what EditorSession::notifyTargetChanged does: rebind what the session owns, then tell
    // every registered tool, without naming any of them.
    void bindTarget(lr::SceneObject &object)
    {
        lr::Mesh &mesh = object.getComponent<lr::MeshComponent>().mesh();
        vertexManager.rebind(mesh);
        selectionManager.rebind(mesh.positions(), object.getComponent<lr::TransformComponent>());
        const lr::EditableMeshContext target = contextFor(object);
        for (lr::EditorTool *tool : tools)
        {
            tool->onTargetChanged(target);
        }
    }

    void clearTarget()
    {
        for (lr::EditorTool *tool : tools)
        {
            tool->onTargetCleared();
        }
    }

    void registerTools()
    {
        lr::EditorServices services(states, shortcuts);
        for (lr::EditorTool *tool : tools)
        {
            tool->registerWith(services);
        }
    }

    lr::Scene         scene;
    lr::MeshStore     meshStore;
    lr::MaterialStore materialStore;
    lr::MeshHandle    firstHandle;
    lr::MeshHandle    secondHandle;
    lr::SceneObject  &first;
    lr::SceneObject  &second;

    lr::InputHandler     input;
    lr::VertexManager    vertexManager;
    lr::SelectionManager selectionManager;
    lr::CommandManager   commandManager;

    lr::EditorStateController     states;
    lr::EditorShortcuts           shortcuts;
    lr::EditorPresentation        presentation;
    std::vector<lr::EditorTool *> tools;
};

// A second tool, so the notification loop is seen to reach more than one.
class RecordingTool final : public lr::EditorTool
{
public:
    void registerWith(lr::EditorServices &services) override { services.states().registerState({.id = "recording"}); }

    void onTargetChanged(const lr::EditableMeshContext &target) override
    {
        ++changes;
        boundMesh = &target.mesh;
    }

    void onTargetCleared() override
    {
        ++clears;
        boundMesh = nullptr;
    }

    void        drawPanel() override {}
    const char *displayName() const override { return "Recording"; }

    int             changes   = 0;
    int             clears    = 0;
    const lr::Mesh *boundMesh = nullptr;
};

// Changing the editable mesh must reach every registered tool from one call, and each must end up
// bound to the new mesh. This is the guarantee that adding a feature does not mean remembering to
// add it to a rebinding function.
void changingTheTargetRebindsEveryTool()
{
    Harness       harness;
    lr::ArapTool  arap(harness.selectionManager, harness.vertexManager, harness.commandManager);
    RecordingTool recording;
    harness.tools = {&arap, &recording};
    harness.registerTools();

    harness.bindTarget(harness.first);
    assert(recording.changes == 1);
    assert(recording.boundMesh == &harness.first.getComponent<lr::MeshComponent>().mesh());

    harness.bindTarget(harness.second);
    assert(recording.changes == 2);
    assert(recording.boundMesh == &harness.second.getComponent<lr::MeshComponent>().mesh());

    // Destroying the editable mesh leaves every tool unbound rather than holding a dangling mesh.
    harness.clearTarget();
    assert(recording.clears == 1);
    assert(recording.boundMesh == nullptr);
}

// A feature state is only meaningful for the mesh it was entered on. When that mesh is replaced or
// destroyed, the tool must retreat to the editor's default state by itself — the host does not know
// which states are incompatible with which targets.
void anUnsupportedTargetForcesASafeStateTransition()
{
    Harness      harness;
    lr::ArapTool arap(harness.selectionManager, harness.vertexManager, harness.commandManager);
    harness.tools = {&arap};
    harness.registerTools();
    harness.bindTarget(harness.first);

    // Entering ARAP publishes the tool's own presentation, which the host applies without knowing
    // whose it is.
    harness.states.activate(lr::ArapTool::stateId());
    assert(harness.states.isActive(lr::ArapTool::stateId()));
    assert(harness.presentation.vertexSelectionActive);
    assert(!harness.presentation.skinningEnabled);

    // Swapping the mesh under an active ARAP session drops back to the default state: a solve
    // factored for the old mesh's vertex domain must not stay reachable from the gizmo.
    harness.bindTarget(harness.second);
    assert(harness.states.isActive(kViewState));
    assert(harness.presentation == lr::EditorPresentation{});
    assert(!arap.hasDeformationTarget());

    // Losing the mesh entirely does the same.
    harness.states.activate(lr::ArapTool::stateId());
    assert(harness.states.isActive(lr::ArapTool::stateId()));
    harness.clearTarget();
    assert(harness.states.isActive(kViewState));
    assert(!arap.hasDeformationTarget());
}

// Rebinding resets vertex selection, because indices only ever meant something in the previous
// mesh's vertex domain.
void rebindingDropsSelectionFromThePreviousMesh()
{
    Harness      harness;
    lr::ArapTool arap(harness.selectionManager, harness.vertexManager, harness.commandManager);
    harness.tools = {&arap};
    harness.registerTools();
    harness.bindTarget(harness.first);

    harness.selectionManager.getSelectedIndices().insert(0u);
    harness.selectionManager.getSelectedIndices().insert(2u);
    assert(harness.selectionManager.getSelectedIndices().size() == 2);

    harness.bindTarget(harness.second);
    assert(harness.selectionManager.getSelectedIndices().empty());
}

// A tool registers its shortcut alongside its state, so the key keeps working across target changes
// without the host re-registering anything.
void aToolsShortcutSurvivesTargetChanges()
{
    Harness      harness;
    lr::ArapTool arap(harness.selectionManager, harness.vertexManager, harness.commandManager);
    harness.tools = {&arap};
    harness.registerTools();
    harness.bindTarget(harness.first);

    // GLFW_KEY_A, spelled literally so this test does not depend on a windowing header.
    const lr::KeyChord arapKey{.key = 'A'};

    assert(harness.shortcuts.dispatch(arapKey));
    assert(harness.states.isActive(lr::ArapTool::stateId()));

    harness.bindTarget(harness.second);
    assert(harness.states.isActive(kViewState));

    assert(harness.shortcuts.dispatch(arapKey));
    assert(harness.states.isActive(lr::ArapTool::stateId()));
}

} // namespace

int main()
{
    changingTheTargetRebindsEveryTool();
    anUnsupportedTargetForcesASafeStateTransition();
    rebindingDropsSelectionFromThePreviousMesh();
    aToolsShortcutSurvivesTargetChanges();
    return 0;
}
