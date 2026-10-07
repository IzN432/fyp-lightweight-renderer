#include "core/editor/EditorStateController.hpp"

#include "core/editor/EditorInputRouter.hpp"
#include "core/editor/EditorShortcuts.hpp"
#include "core/editor/EditorTool.hpp"

#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

// Minimal stand-in for whatever a real state wants the gizmo to drive. The controller only forwards
// the pointer, so a bare handler with no behavior is enough to prove the request reaches the host.
class NoopTranslateHandler final : public lr::TranslateDragHandler
{
public:
    void beginDrag() override {}
    void translate(const glm::vec3 &) override {}
    void endDrag(const glm::vec3 &) override {}
};

void registrationRejectsBadStates()
{
    lr::EditorStateController controller([](const lr::EditorStateDefinition &) {});

    bool threwOnEmptyId = false;
    try
    {
        controller.registerState({.id = ""});
    } catch (const std::invalid_argument &)
    {
        threwOnEmptyId = true;
    }
    assert(threwOnEmptyId);

    controller.registerState({.id = "view"});
    bool threwOnDuplicate = false;
    try
    {
        controller.registerState({.id = "view"});
    } catch (const std::logic_error &)
    {
        threwOnDuplicate = true;
    }
    assert(threwOnDuplicate);

    bool threwOnUnknown = false;
    try
    {
        controller.activate("nope");
    } catch (const std::out_of_range &)
    {
        threwOnUnknown = true;
    }
    assert(threwOnUnknown);
}

void activationOrdersExitPresentationEnter()
{
    std::vector<std::string>  log;
    lr::EditorStateController controller([&log](const lr::EditorStateDefinition &state) {
        log.push_back("publish:" + state.id);
    });

    controller.registerState({
        .id = "view",
        .onEnter =
            [&log] {
                log.push_back("enter:view");
            },
        .onExit =
            [&log] {
                log.push_back("exit:view");
            },
    });
    controller.registerState({
        .id = "edit",
        .onEnter =
            [&log] {
                log.push_back("enter:edit");
            },
        .onExit =
            [&log] {
                log.push_back("exit:edit");
            },
    });

    // The first activation has no outgoing state, so it must not fabricate an onExit.
    controller.activate("view");
    assert((log == std::vector<std::string>{"publish:view", "enter:view"}));

    log.clear();
    controller.activate("edit");
    assert((log == std::vector<std::string>{"exit:view", "publish:edit", "enter:edit"}));
    assert(controller.isActive("edit"));

    // Re-activating the current state is a no-op: no exit/enter churn, no presentation republish.
    log.clear();
    controller.activate("edit");
    assert(log.empty());
}

void activeStateOwnsFrameBehavior()
{
    lr::EditorStateController controller([](const lr::EditorStateDefinition &) {});
    NoopTranslateHandler      handler;

    int updates = 0;
    controller.registerState({
        .id = "quiet",
    });
    controller.registerState({
        .id = "busy",
        .update =
            [&updates](const lr::EditorFrameContext &frame) {
                assert(frame.extent.width == 800);
                ++updates;
            },
        .gizmoRequest = [&handler](const lr::EditorFrameContext &) -> lr::GizmoRequest {
            return lr::TranslateGizmoRequest{.origin = glm::vec3(1.0f, 2.0f, 3.0f), .handler = &handler};
        },
    });

    const lr::EditorFrameContext frame{.extent = {800, 600}};

    // A state that defines no hooks must still be drivable, and must bid for no gizmo.
    controller.activate("quiet");
    controller.update(frame);
    assert(updates == 0);
    assert(std::holds_alternative<std::monostate>(controller.gizmoRequest(frame)));

    controller.activate("busy");
    controller.update(frame);
    assert(updates == 1);
    const lr::GizmoRequest request = controller.gizmoRequest(frame);
    assert(std::holds_alternative<lr::TranslateGizmoRequest>(request));
    assert(std::get<lr::TranslateGizmoRequest>(request).handler == &handler);

    // Switching away withdraws both the per-frame work and the gizmo bid.
    controller.activate("quiet");
    controller.update(frame);
    assert(updates == 1);
    assert(std::holds_alternative<std::monostate>(controller.gizmoRequest(frame)));
}

void defaultStateIsTheFallbackAndToggleTarget()
{
    lr::EditorStateController controller([](const lr::EditorStateDefinition &) {});
    controller.registerState({.id = "view"});
    controller.registerState({.id = "arap"});

    // A default must name a registered state, and must exist before anything falls back to it.
    bool threwOnUnknownDefault = false;
    try
    {
        controller.setDefaultState("nope");
    } catch (const std::out_of_range &)
    {
        threwOnUnknownDefault = true;
    }
    assert(threwOnUnknownDefault);

    bool threwWithoutDefault = false;
    try
    {
        controller.activateDefault();
    } catch (const std::logic_error &)
    {
        threwWithoutDefault = true;
    }
    assert(threwWithoutDefault);

    controller.setDefaultState("view");
    controller.activateDefault();
    assert(controller.isActive("view"));

    controller.toggle("arap");
    assert(controller.isActive("arap"));
    controller.toggle("arap");
    assert(controller.isActive("view"));
}

// Stands in for a feature module: contributes a state, follows the editable target, draws a panel.
class FakeTool final : public lr::EditorTool
{
public:
    static constexpr int kToggleKey = 'F';

    void registerWith(lr::EditorServices &services) override
    {
        m_states = &services.states();
        m_states->registerState({
            .id           = "fake",
            .presentation = {.skinningEnabled = false, .vertexSelectionActive = true},
        });
        // A feature brings its own key along with its state.
        services.shortcuts().add({.key = kToggleKey}, [this] {
            m_states->toggle("fake");
        });
    }

    // Binding a real target needs a Mesh and a SceneObject, so that path is covered where the
    // session itself is exercised; here only the teardown half matters.
    void onTargetChanged(const lr::EditableMeshContext &) override { retreat(); }

    void onTargetCleared() override
    {
        ++targetClears;
        retreat();
    }

    void        drawPanel() override { ++panelDraws; }
    const char *displayName() const override { return "Fake"; }

    int targetClears = 0;
    int panelDraws   = 0;

private:
    void retreat()
    {
        if (m_states->isActive("fake"))
        {
            m_states->activateDefault();
        }
    }

    lr::EditorStateController *m_states = nullptr;
};

void aRegisteredToolContributesItsOwnState()
{
    lr::EditorPresentation    published;
    lr::EditorStateController controller([&published](const lr::EditorStateDefinition &state) {
        published = state.presentation;
    });
    controller.registerState({.id = "view"});
    controller.setDefaultState("view");

    FakeTool            tool;
    lr::EditorShortcuts shortcuts;
    lr::EditorServices  services(controller, shortcuts);
    tool.registerWith(services);

    // The host drives the tool only through the base interface, and never names its state, yet
    // activating that state publishes the tool's own policy.
    lr::EditorTool *driven = &tool;
    driven->drawPanel();
    assert(tool.panelDraws == 1);

    // The tool's own key reaches its own state without the host knowing either one.
    assert(shortcuts.dispatch({.key = FakeTool::kToggleKey}));
    assert(controller.isActive("fake"));
    assert(published.vertexSelectionActive);
    assert(!published.skinningEnabled);

    // A tool whose state is active when its target goes away retreats to the host's default, so a
    // feature state can never outlive the mesh it was meaningful for.
    driven->onTargetCleared();
    assert(tool.targetClears == 1);
    assert(controller.isActive("view"));
    assert(published == lr::EditorPresentation{});
}

void shortcutsMatchChordsAndStopAtTheFirstOwner()
{
    lr::EditorShortcuts shortcuts;

    bool emptyActionRejected = false;
    try
    {
        shortcuts.add({.key = 'X'}, {});
    } catch (const std::invalid_argument &)
    {
        emptyActionRejected = true;
    }
    assert(emptyActionRejected);

    std::vector<std::string> fired;
    shortcuts.add({.key = 'Z', .ctrl = true}, [&fired] {
        fired.push_back("undo");
    });
    shortcuts.add({.key = 'A'}, [&fired] {
        fired.push_back("editor-a");
    });
    shortcuts.add({.key = 'A'}, [&fired] {
        fired.push_back("tool-a");
    });

    // Ctrl is part of the match, so a bare Z is not undo and Ctrl+A is not the plain-A shortcut.
    assert(!shortcuts.dispatch({.key = 'Z'}));
    assert(!shortcuts.dispatch({.key = 'A', .ctrl = true}));
    assert(fired.empty());

    assert(shortcuts.dispatch({.key = 'Z', .ctrl = true}));

    // Whoever registered first owns the chord — which is why the host registers before its tools.
    assert(shortcuts.dispatch({.key = 'A'}));
    assert((fired == std::vector<std::string>{"undo", "editor-a"}));

    assert(!shortcuts.dispatch({.key = 'Q'}));
}

void routerStopsAtTheFirstLayerThatConsumes()
{
    bool uiCaptures    = false;
    bool gizmoCaptures = false;

    lr::EditorInputRouter router(
        [&uiCaptures] {
            return uiCaptures;
        },
        [&gizmoCaptures] {
            return gizmoCaptures;
        });

    std::vector<std::string> saw;
    // The upper layer only claims button 0, so button 1 must fall through to the lower one.
    router.addButtonLayer("upper", [&saw](const lr::PointerButtonEvent &event) {
        saw.push_back("upper");
        return event.button == 0;
    });
    router.addButtonLayer("lower", [&saw](const lr::PointerButtonEvent &) {
        saw.push_back("lower");
        return true;
    });

    assert(router.routeButton({.button = 0}));
    assert((saw == std::vector<std::string>{"upper"}));

    saw.clear();
    assert(router.routeButton({.button = 1}));
    assert((saw == std::vector<std::string>{"upper", "lower"}));

    // While the UI or the gizmo owns the pointer, no layer is offered anything at all.
    saw.clear();
    uiCaptures = true;
    assert(!router.routeButton({.button = 0}));
    uiCaptures    = false;
    gizmoCaptures = true;
    assert(!router.routeButton({.button = 0}));
    assert(saw.empty());
}

void navigationYieldsToUiButNotToTheGizmo()
{
    bool uiCaptures    = false;
    bool gizmoCaptures = false;

    const lr::EditorInputRouter router(
        [&uiCaptures] {
            return uiCaptures;
        },
        [&gizmoCaptures] {
            return gizmoCaptures;
        });

    assert(router.viewportNavigationAllowed());
    assert(!router.pointerCaptured());

    // A real UI window stops orbit/pan/zoom.
    uiCaptures = true;
    assert(!router.viewportNavigationAllowed());

    // A hovered gizmo also raises the UI capture flag, because ImGuizmo goes through ImGui — but the
    // navigation buttons are ones the gizmo never takes, so navigation must survive it.
    gizmoCaptures = true;
    assert(router.viewportNavigationAllowed());
    assert(router.pointerCaptured());

    bool rejectedEmptyQuery = false;
    try
    {
        lr::EditorInputRouter bad({}, [] {
            return false;
        });
    } catch (const std::invalid_argument &)
    {
        rejectedEmptyQuery = true;
    }
    assert(rejectedEmptyQuery);
}

void anActiveStateCanClaimPointerInput()
{
    lr::EditorStateController controller([](const lr::EditorStateDefinition &) {});

    int handled = 0;
    controller.registerState({.id = "passive"});
    controller.registerState({
        .id = "greedy",
        .handleInput =
            [&handled](const lr::PointerButtonEvent &event) {
                if (event.button != 0)
                {
                    return false;
                }
                ++handled;
                return true;
            },
    });

    lr::EditorInputRouter router(
        [] {
            return false;
        },
        [] {
            return false;
        });
    router.addButtonLayer("active editor state", [&controller](const lr::PointerButtonEvent &event) {
        return controller.handleInput(event);
    });

    bool fellThrough = false;
    router.addButtonLayer("fallback", [&fellThrough](const lr::PointerButtonEvent &) {
        fellThrough = true;
        return true;
    });

    // A state with no input hook is not an error; the event simply falls through.
    controller.activate("passive");
    assert(router.routeButton({.button = 0}));
    assert(handled == 0);
    assert(fellThrough);

    fellThrough = false;
    controller.activate("greedy");
    assert(router.routeButton({.button = 0}));
    assert(handled == 1);
    assert(!fellThrough);

    // The state only claims button 0, so anything else still reaches the layer below it.
    assert(router.routeButton({.button = 2}));
    assert(handled == 1);
    assert(fellThrough);
}

void activeThrowsBeforeAnyActivation()
{
    lr::EditorStateController controller([](const lr::EditorStateDefinition &) {});
    controller.registerState({.id = "view"});

    bool threw = false;
    try
    {
        controller.active();
    } catch (const std::logic_error &)
    {
        threw = true;
    }
    assert(threw);
}

} // namespace

int main()
{
    registrationRejectsBadStates();
    activationOrdersExitPresentationEnter();
    activeStateOwnsFrameBehavior();
    defaultStateIsTheFallbackAndToggleTarget();
    aRegisteredToolContributesItsOwnState();
    shortcutsMatchChordsAndStopAtTheFirstOwner();
    routerStopsAtTheFirstLayerThatConsumes();
    navigationYieldsToUiButNotToTheGizmo();
    anActiveStateCanClaimPointerInput();
    activeThrowsBeforeAnyActivation();
    return 0;
}
