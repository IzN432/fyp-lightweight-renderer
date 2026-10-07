#include "app/EditorStateController.hpp"

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
    activeThrowsBeforeAnyActivation();
    return 0;
}
