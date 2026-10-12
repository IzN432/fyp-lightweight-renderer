#include "features/fluid/ParticleBillboardPass.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include <cassert>
#include <cmath>
#include <stdexcept>

namespace
{

constexpr float kEpsilon = 1e-5f;

bool near(float actual, float expected, float epsilon = kEpsilon) { return std::abs(actual - expected) <= epsilon; }

glm::mat4 projection(float nearPlane = 0.1f)
{
    glm::mat4 result = glm::perspective(glm::radians(60.0f), 1.0f, nearPlane, 100.0f);
    result[1][1] *= -1.0f;
    glm::mat4 clip(1.0f);
    clip[2][2] = 0.5f;
    clip[3][2] = 0.5f;
    return clip * result;
}

void rejectsCountsBeyondCapacity()
{
    assert(lr::particle_billboard::checkedParticleCount(0, 4) == 0);
    assert(lr::particle_billboard::checkedParticleCount(4, 4) == 4);

    bool rejected = false;
    try
    {
        (void)lr::particle_billboard::checkedParticleCount(5, 4);
    } catch (const std::length_error &)
    {
        rejected = true;
    }
    assert(rejected);
}

void projectedRadiusMatchesPerspectiveProjection()
{
    const glm::mat4 p        = projection();
    const float     expected = (1.0f / std::tan(glm::radians(30.0f))) * 0.5f / 5.0f;

    const glm::vec2 centred = lr::particle_billboard::projectedRadiusNdc(p, {0.0f, 0.0f, -5.0f}, 0.5f);
    assert(near(centred.x, expected));
    assert(near(centred.y, expected));

    // Translating the centre toward a screen edge must not change a view-aligned quad's half-size.
    const glm::vec2 atEdge = lr::particle_billboard::projectedRadiusNdc(p, {2.0f, 1.0f, -5.0f}, 0.5f);
    assert(near(atEdge.x, centred.x));
    assert(near(atEdge.y, centred.y));

    // Close to the near plane the projection may cover more than the viewport, but must stay finite.
    const glm::vec2 nearPlane = lr::particle_billboard::projectedRadiusNdc(p, {0.0f, 0.0f, -0.15f}, 0.1f);
    assert(std::isfinite(nearPlane.x) && std::isfinite(nearPlane.y));
    assert(nearPlane.x > 1.0f && nearPlane.y > 1.0f);
}

void reconstructsNearestSphereDepth()
{
    const glm::mat4 p = projection();
    const glm::vec3 centre(0.0f, 0.0f, -5.0f);

    const auto middle = lr::particle_billboard::reconstructSurface(centre, 1.0f, {0.0f, 0.0f});
    const auto edge   = lr::particle_billboard::reconstructSurface(centre, 1.0f, {1.0f, 0.0f});
    assert(middle && edge);
    assert(near(middle->z, -4.0f));
    assert(near(edge->z, centre.z));
    assert(lr::particle_billboard::projectDepth(p, *middle) < lr::particle_billboard::projectDepth(p, *edge));

    assert(!lr::particle_billboard::reconstructSurface(centre, 1.0f, {1.01f, 0.0f}));
    assert(!lr::particle_billboard::reconstructSurface(centre, 0.0f, {0.0f, 0.0f}));

    // A sphere crossing the near plane has clipped central fragments but a finite rim. This is the
    // case most likely to expose a division, square-root, or view-space sign regression.
    const auto nearMiddle = lr::particle_billboard::reconstructSurface({0.0f, 0.0f, -0.15f}, 0.1f, {0.0f, 0.0f});
    const auto nearEdge   = lr::particle_billboard::reconstructSurface({0.0f, 0.0f, -0.15f}, 0.1f, {1.0f, 0.0f});
    assert(nearMiddle && nearEdge);
    const float middleDepth = lr::particle_billboard::projectDepth(p, *nearMiddle);
    const float edgeDepth   = lr::particle_billboard::projectDepth(p, *nearEdge);
    assert(std::isfinite(middleDepth) && std::isfinite(edgeDepth));
    assert(middleDepth < 0.0f);
    assert(edgeDepth >= 0.0f && edgeDepth <= 1.0f);
}

} // namespace

int main()
{
    rejectsCountsBeyondCapacity();
    projectedRadiusMatchesPerspectiveProjection();
    reconstructsNearestSphereDepth();
    return 0;
}
