#include "core/scene/Light.hpp"

#include "core/passes/overlaylines/OverlayLine.hpp"
#include "core/scene/SceneObject.hpp"

#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace lr
{
namespace
{

constexpr float kDirectionalLength = 1.5f;
constexpr int   kCircleSegments    = 24;
constexpr glm::vec3 kGizmoColor{234.0f / 255.0f, 145.0f / 255.0f, 40.0f / 255.0f};

void addCircle(OverlayLineBuilder &builder, const glm::vec3 &center, const glm::vec3 &right,
               const glm::vec3 &up, float radius, const OverlayLineStyle &style)
{
    for (int i = 0; i < kCircleSegments; ++i)
    {
        const float a0 = glm::two_pi<float>() * static_cast<float>(i) / static_cast<float>(kCircleSegments);
        const float a1 = glm::two_pi<float>() * static_cast<float>(i + 1) / static_cast<float>(kCircleSegments);
        const glm::vec3 p0 = center + radius * (right * std::cos(a0) + up * std::sin(a0));
        const glm::vec3 p1 = center + radius * (right * std::cos(a1) + up * std::sin(a1));
        builder.addLine(p0, p1, style);
    }
}

void addSpot(OverlayLineBuilder &builder, const glm::vec3 &origin, const glm::quat &rotation,
             const SelectionGizmoContext &context, const SpotLight &light, const OverlayLineStyle &style)
{
    const glm::vec3 direction = glm::normalize(rotation * glm::vec3(0.0f, 0.0f, -1.0f));
    const glm::vec3 right     = glm::normalize(rotation * glm::vec3(1.0f, 0.0f, 0.0f));
    const glm::vec3 up        = glm::normalize(rotation * glm::vec3(0.0f, 1.0f, 0.0f));
    const float spotLength = light.range;
    const glm::vec3 lowerCenter = origin + direction * spotLength;
    const float coneSlope = std::tan(glm::radians(glm::clamp(light.outerConeAngleDegrees, 0.0f, 89.0f)));
    const float lowerRadius = spotLength * coneSlope;
    const float innerRadius = spotLength *
                              std::tan(glm::radians(glm::clamp(light.innerConeAngleDegrees, 0.0f, 89.0f)));

    addCircle(builder, lowerCenter, right, up, lowerRadius, style);
    addCircle(builder, lowerCenter, right, up, innerRadius, style);

    // A silhouette generator has a surface normal perpendicular to the camera ray. Solving that
    // condition gives the two points where lines from the projected apex touch the projected circle.
    const glm::vec3 toCamera = context.orthographic ? -glm::normalize(context.cameraForward)
                                                     : context.cameraPosition - origin;
    const float axial = glm::dot(toCamera, direction);
    const glm::vec3 radialView = toCamera - direction * axial;
    const float radialViewLength = glm::length(radialView);

    if (radialViewLength <= 1e-6f)
        return;

    const glm::vec3 towardCamera = radialView / radialViewLength;
    const float tangentOffset = coneSlope * axial / radialViewLength;
    // No two real tangents means the projected apex lies inside the projected circle.
    if (std::abs(tangentOffset) >= 1.0f)
        return;

    const glm::vec3 sideways = glm::normalize(glm::cross(direction, towardCamera));
    const float tangentSide = std::sqrt(1.0f - tangentOffset * tangentOffset);
    const glm::vec3 radialA = tangentOffset * towardCamera + tangentSide * sideways;
    const glm::vec3 radialB = tangentOffset * towardCamera - tangentSide * sideways;

    builder.addLine(origin, lowerCenter + radialA * lowerRadius, style);
    builder.addLine(origin, lowerCenter + radialB * lowerRadius, style);
}

} // namespace

void Light::onSelectGizmo(SelectionGizmoContext &context) const
{
    const SceneObject &object   = getOwningObject();
    const glm::vec3    origin   = glm::vec3(object.worldMatrix()[3]);
    const glm::quat    rotation = object.worldRotation();

    std::visit([&](const auto &value) {
        using T = std::decay_t<decltype(value)>;
        const OverlayLineStyle style{
            .color            = kGizmoColor,
            .visibleOpacity   = 1.0f,
            .occludedOpacity = 0.22f,
        };
        if constexpr (std::is_same_v<T, DirectionalLight>)
        {
            const glm::vec3 direction = glm::normalize(rotation * glm::vec3(0.0f, 0.0f, -1.0f));
            context.lines.addLine(origin, origin + direction * kDirectionalLength, style);
        }
        else if constexpr (std::is_same_v<T, SpotLight>)
            addSpot(context.lines, origin, rotation, context, value, style);
    }, light);
}

} // namespace lr
