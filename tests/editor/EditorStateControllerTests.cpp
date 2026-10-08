#include "core/editor/EditorStateController.hpp"

#include "core/editor/EditorInputRouter.hpp"
#include "core/editor/EditorPresentationState.hpp"
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
    const lr::EditorInputContext inputContext{.activeState = "view",
                                               .presentation = {.objectSelectionActive = true}};
    // The upper layer only claims button 0, so button 1 must fall through to the lower one.
    router.addButtonLayer("upper", [&saw](const lr::PointerButtonEvent &event, const lr::EditorInputContext &context) {
        assert(context.isState("view"));
        assert(context.presentation.objectSelectionActive);
        saw.push_back("upper");
        return event.button == 0;
    });
    router.addButtonLayer("lower", [&saw](const lr::PointerButtonEvent &, const lr::EditorInputContext &) {
        saw.push_back("lower");
        return true;
    });

    assert(router.routeButton({.button = 0}, inputContext));
    assert((saw == std::vector<std::string>{"upper"}));

    saw.clear();
    assert(router.routeButton({.button = 1}, inputContext));
    assert((saw == std::vector<std::string>{"upper", "lower"}));

    // While the UI or the gizmo owns the pointer, no layer is offered anything at all.
    saw.clear();
    uiCaptures = true;
    assert(!router.routeButton({.button = 0}, inputContext));
    uiCaptures    = false;
    gizmoCaptures = true;
    assert(!router.routeButton({.button = 0}, inputContext));
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
    router.addButtonLayer("active editor state", [&controller](const lr::PointerButtonEvent &event,
                                                                const lr::EditorInputContext &) {
        return controller.handleInput(event);
    });

    bool fellThrough = false;
    router.addButtonLayer("fallback", [&fellThrough](const lr::PointerButtonEvent &,
                                                      const lr::EditorInputContext &) {
        fellThrough = true;
        return true;
    });

    // A state with no input hook is not an error; the event simply falls through.
    controller.activate("passive");
    assert(router.routeButton({.button = 0}, {.activeState = controller.activeId(),
                                               .presentation = controller.active().presentation}));
    assert(handled == 0);
    assert(fellThrough);

    fellThrough = false;
    controller.activate("greedy");
    assert(router.routeButton({.button = 0}, {.activeState = controller.activeId(),
                                               .presentation = controller.active().presentation}));
    assert(handled == 1);
    assert(!fellThrough);

    // The state only claims button 0, so anything else still reaches the layer below it.
    assert(router.routeButton({.button = 2}, {.activeState = controller.activeId(),
                                               .presentation = controller.active().presentation}));
    assert(handled == 1);
    assert(fellThrough);
}

// Two states that both want the translate gizmo. Only one of them can ever be the active state, so
// the controller can only ever hand the host one request — the exclusivity the gizmo depends on is a
// property of state activation, not something GizmoController has to police.
void onlyTheActiveStateBidsForTheGizmo()
{
    lr::EditorStateController controller([](const lr::EditorStateDefinition &) {});
    NoopTranslateHandler      vertexHandler;
    NoopTranslateHandler      objectHandler;

    controller.registerState({
        .id           = "edit",
        .gizmoRequest = [&vertexHandler](const lr::EditorFrameContext &) -> lr::GizmoRequest {
            return lr::TranslateGizmoRequest{.origin = glm::vec3(1.0f), .handler = &vertexHandler};
        },
    });
    controller.registerState({
        .id           = "view",
        .gizmoRequest = [&objectHandler](const lr::EditorFrameContext &) -> lr::GizmoRequest {
            return lr::TranslateGizmoRequest{.origin = glm::vec3(2.0f), .handler = &objectHandler};
        },
    });
    controller.registerState({.id = "analysis"});

    const lr::EditorFrameContext frame{};

    const auto handlerOf = [](const lr::GizmoRequest &request) -> const lr::TranslateDragHandler * {
        const auto *translate = std::get_if<lr::TranslateGizmoRequest>(&request);
        return translate ? translate->handler : nullptr;
    };

    controller.activate("edit");
    assert(handlerOf(controller.gizmoRequest(frame)) == &vertexHandler);

    // Switching states withdraws the previous bid in the same breath as publishing the new one:
    // there is no frame in which both are live, which is what lets GizmoController finish the old
    // interaction by simply not being asked for it again.
    controller.activate("view");
    assert(handlerOf(controller.gizmoRequest(frame)) == &objectHandler);

    // And a state that wants no gizmo withdraws it entirely rather than leaving the last one up.
    controller.activate("analysis");
    assert(std::holds_alternative<std::monostate>(controller.gizmoRequest(frame)));
}

// The selection-lifetime policy itself, driven by the real EditorPresentationState: vertex indices
// only mean something while a state actually selects vertices, so withdrawing that capability must
// invalidate the selection, and leaving it in place must not.
void withdrawingVertexSelectionInvalidatesTheSelection()
{
    int                    invalidations = 0;
    std::vector<bool>      publishedSelectionActive;
    lr::EditorPresentation lastPublished;

    lr::EditorPresentationState  presentation([&invalidations] {
        ++invalidations;
    });
    const lr::CallbackConnection connection =
        presentation.registerChangedCallback([&](const lr::EditorPresentation &applied) {
            lastPublished = applied;
            publishedSelectionActive.push_back(applied.vertexSelectionActive);
        });

    const lr::EditorPresentation vertexEditing{
        .skinningEnabled = false, .vertexPointsVisible = true, .vertexSelectionActive = true};
    const lr::EditorPresentation analysis{.skinningEnabled = false, .heatmapVisible = true};

    assert(presentation.current() == lr::EditorPresentation{});

    presentation.set(vertexEditing);
    assert(invalidations == 0);
    assert(presentation.current().vertexSelectionActive);

    // Edit -> ARAP: a different state asking for the same capabilities is not a transition at all,
    // so nothing is published and nothing is invalidated. This is what makes handing a selection
    // from one vertex-selecting state to another safe.
    presentation.set(vertexEditing);
    assert(invalidations == 0);
    assert(publishedSelectionActive.size() == 1);

    // Leaving vertex selection is the one transition that invalidates.
    presentation.set(analysis);
    assert(invalidations == 1);
    assert(!presentation.current().vertexSelectionActive);
    assert(presentation.current().heatmapVisible);

    // Going from one non-selecting presentation to another changes what is rendered but has no
    // selection to invalidate.
    presentation.set({});
    assert(invalidations == 1);
    assert(publishedSelectionActive.size() == 3);

    // Returning to vertex editing grants the capability rather than withdrawing it.
    presentation.set(vertexEditing);
    assert(invalidations == 1);
    assert(lastPublished.vertexSelectionActive);

    bool rejectedEmptyCallback = false;
    try
    {
        lr::EditorPresentationState bad({});
    } catch (const std::invalid_argument &)
    {
        rejectedEmptyCallback = true;
    }
    assert(rejectedEmptyCallback);
}

// The state controller is what feeds that policy, so what it owes it is exactly one publish per real
// transition — and none when the active state is re-activated.
void activatingAStatePublishesItsPresentationOnce()
{
    std::vector<lr::EditorPresentation> published;
    lr::EditorStateController           controller([&published](const lr::EditorStateDefinition &state) {
        published.push_back(state.presentation);
    });

    const lr::EditorPresentation vertexEditing{
        .skinningEnabled = false, .vertexPointsVisible = true, .vertexSelectionActive = true};

    controller.registerState({.id = "view", .presentation = {}});
    controller.registerState({.id = "edit", .presentation = vertexEditing});
    // ARAP asks for the same capabilities as plain vertex editing.
    controller.registerState({.id = "arap", .presentation = vertexEditing});

    controller.activate("view");
    controller.activate("edit");
    controller.activate("arap");
    assert(published.size() == 3);
    assert(published[1] == published[2]);

    // Re-activating the active state publishes nothing, so a consumer is never handed a transition
    // the user did not make.
    controller.activate("arap");
    assert(published.size() == 3);

    controller.activate("view");
    assert(published.size() == 4);
    assert(!published.back().vertexSelectionActive);
}

// A state added later must work without the presentation consumer learning anything about it: the
// consumer only ever reads EditorPresentation fields. This is what lets a feature library add a
// state without SceneManager (or any other consumer) changing.
void aNewStateNeedsNoConsumerChange()
{
    // Stands in for every presentation consumer — it knows the flags, never the state ids.
    lr::EditorPresentation applied;
    int                    applications = 0;

    lr::EditorStateController controller([&](const lr::EditorStateDefinition &state) {
        applied = state.presentation;
        ++applications;
    });

    controller.registerState({.id = "view", .presentation = {}});
    controller.setDefaultState("view");
    controller.activate("view");
    assert(applications == 1);

    // A state invented after the consumer was written, with a combination of flags no existing state
    // uses.
    controller.registerState({
        .id           = "future-feature",
        .presentation = {.skinningEnabled = true, .vertexPointsVisible = true, .heatmapVisible = true},
    });
    controller.activate("future-feature");

    assert(applications == 2);
    assert(applied.skinningEnabled);
    assert(applied.vertexPointsVisible);
    assert(applied.heatmapVisible);
    assert(!applied.vertexSelectionActive);

    // And it takes part in the default-state fallback like any other state.
    controller.activateDefault();
    assert(controller.isActive("view"));
    assert(applied == lr::EditorPresentation{});
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
    onlyTheActiveStateBidsForTheGizmo();
    withdrawingVertexSelectionInvalidatesTheSelection();
    activatingAStatePublishesItsPresentationOnce();
    aNewStateNeedsNoConsumerChange();
    activeThrowsBeforeAnyActivation();
    return 0;
}
