#include "core/passes/overlaylines/OverlayLine.hpp"
#include "core/scene/Light.hpp"
#include "core/scene/Scene.hpp"
#include "core/scene/TransformComponent.hpp"

#include <glm/gtc/epsilon.hpp>

#include <cassert>
#include <cmath>

namespace
{

class TestGizmoComponent : public lr::Component
{
public:
    void onSelectGizmo(lr::SelectionGizmoContext &context) const override
    {
        context.lines.addLine(context.cameraPosition, glm::vec3(20.0f));
    }
};

} // namespace

int main()
{
    lr::Scene scene;
    auto &directional = scene.createSceneObject();
    directional.addComponent<lr::TransformComponent>(glm::vec3(1.0f, 2.0f, 3.0f));
    directional.addComponent<lr::Light>(lr::DirectionalLight{});

    lr::OverlayLineBuilder builder;
    lr::SelectionGizmoContext context{.lines = builder, .cameraPosition = glm::vec3(0.0f, 0.0f, 10.0f)};
    directional.onSelectGizmo(context);
    const auto &directionalLines = builder.lines();
    assert(directionalLines.size() == 1);
    assert(glm::all(glm::epsilonEqual(directionalLines[0].start,
                                     glm::vec3(1.0f, 2.0f, 3.0f), 0.0001f)));
    assert(glm::all(glm::epsilonEqual(directionalLines[0].end,
                                     glm::vec3(1.0f, 2.0f, 1.5f), 0.0001f)));
    assert(directionalLines[0].style.visibleOpacity == 1.0f);
    assert(glm::all(glm::epsilonEqual(directionalLines[0].style.color,
                                     glm::vec3(234.0f, 145.0f, 40.0f) / 255.0f, 0.0001f)));

    auto &spot = scene.createSceneObject();
    spot.addComponent<lr::TransformComponent>(glm::vec3(-1.0f, 0.0f, 1.0f));
    lr::SpotLight spotLight{{}, 15.0f, 30.0f};
    spotLight.range = 6.0f;
    spot.addComponent<lr::Light>(spotLight);

    lr::OverlayLineBuilder spotBuilder;
    lr::SelectionGizmoContext spotContext{.lines = spotBuilder, .cameraPosition = glm::vec3(4.0f, 0.0f, 2.0f)};
    spot.onSelectGizmo(spotContext);
    const auto &spotLines = spotBuilder.lines();
    assert(spotLines.size() == 50); // Outer + inner circles, then two outer-cone tangent sides.
    assert(glm::all(glm::epsilonEqual(spotLines[0].start,
                                     glm::vec3(-1.0f + spotLight.range * glm::tan(glm::radians(30.0f)), 0.0f,
                                               1.0f - spotLight.range),
                                     0.0001f)));
    const float slope = glm::tan(glm::radians(30.0f));
    const float lowerRadius = spotLight.range * slope;
    const glm::vec3 apex(-1.0f, 0.0f, 1.0f);
    const glm::vec3 axis(0.0f, 0.0f, -1.0f);
    const glm::vec3 toCamera = spotContext.cameraPosition - apex;
    const float axial = glm::dot(toCamera, axis);
    const glm::vec3 radialView = toCamera - axis * axial;
    const glm::vec3 towardCamera = glm::normalize(radialView);
    const glm::vec3 sideways = glm::normalize(glm::cross(axis, towardCamera));
    const float offset = slope * axial / glm::length(radialView);
    const glm::vec3 expectedRadial = offset * towardCamera + std::sqrt(1.0f - offset * offset) * sideways;
    assert(glm::all(glm::epsilonEqual(spotLines[24].start,
                                     glm::vec3(-1.0f + spotLight.range * glm::tan(glm::radians(15.0f)), 0.0f,
                                               1.0f - spotLight.range),
                                     0.0001f)));
    assert(spotLines[48].start == apex);
    assert(glm::all(glm::epsilonEqual(spotLines[48].end,
                                     apex + axis * spotLight.range + expectedRadial * lowerRadius, 0.0001f)));

    // Looking down the cone axis hides only the tangent sides; both angle circles remain.
    lr::OverlayLineBuilder nestedBuilder;
    lr::SelectionGizmoContext nestedContext{
        .lines = nestedBuilder,
        .cameraPosition = glm::vec3(-1.0f, 0.0f, 5.0f),
    };
    spot.onSelectGizmo(nestedContext);
    assert(nestedBuilder.lines().size() == 48);

    // Non-UI input is normalized at the component boundary; equal angles are valid.
    lr::Light constrained(lr::SpotLight{{}, 40.0f, 20.0f});
    const auto &constrainedSpot = std::get<lr::SpotLight>(constrained.light);
    assert(glm::epsilonEqual(constrainedSpot.innerConeAngleDegrees, 20.0f, 0.0001f));
    assert(constrainedSpot.innerConeAngleDegrees == constrainedSpot.outerConeAngleDegrees);

    // Components without a selection gizmo remain no-ops.
    auto &point = scene.createSceneObject();
    point.addComponent<lr::TransformComponent>();
    point.addComponent<lr::Light>(lr::PointLight{});
    lr::OverlayLineBuilder pointBuilder;
    lr::SelectionGizmoContext pointContext{.lines = pointBuilder};
    point.onSelectGizmo(pointContext);
    assert(pointBuilder.lines().empty());

    // SceneObject dispatches the hook without knowing the concrete component type.
    auto &custom = scene.createSceneObject();
    custom.addComponent<TestGizmoComponent>();
    lr::OverlayLineBuilder customBuilder;
    lr::SelectionGizmoContext customContext{.lines = customBuilder, .cameraPosition = glm::vec3(10.0f)};
    custom.onSelectGizmo(customContext);
    assert(customBuilder.lines().size() == 1);
    assert(customBuilder.lines().front().start == glm::vec3(10.0f));
}
