#include "features/fluid/ScreenSpaceLiquidPass.hpp"

#include <cassert>
#include <cmath>

namespace
{

bool near(float actual, float expected, float epsilon = 1e-5f)
{
    return std::abs(actual - expected) <= epsilon;
}

void dispatchRoundsUpWithoutExtraGroups()
{
    using lr::screen_space_liquid::dispatchSize;
    assert(dispatchSize(1) == 1);
    assert(dispatchSize(16) == 1);
    assert(dispatchSize(17) == 2);
    assert(dispatchSize(1920) == 120);
}

void waterFresnelMatchesPhysicalNormalIncidence()
{
    const float f0 = lr::screen_space_liquid::fresnelF0(1.333f);
    assert(near(f0, 0.02037f, 1e-4f));
    assert(near(lr::screen_space_liquid::fresnelF0(1.0f), 0.0f));
}

void absorptionPreservesTheSelectedHue()
{
    const glm::vec3 absorption =
        lr::screen_space_liquid::absorptionCoefficient({0.25f, 0.5f, 0.95f}, 2.0f);
    // Blue light is absorbed least, so deep liquid retains the selected blue hue.
    assert(absorption.r > absorption.g);
    assert(absorption.g > absorption.b);
    assert(absorption.b > 0.0f);

    const glm::vec3 clamped = lr::screen_space_liquid::absorptionCoefficient({-1.0f, 2.0f, 0.5f}, -1.0f);
    assert(clamped == glm::vec3(0.0f));
}

void bulkTintKeepsCyanVisibleWithoutTurningGreen()
{
    const glm::vec3 tint{0.0f, 0.8f, 1.0f};
    const glm::vec3 againstBlack =
        lr::screen_space_liquid::compositeTransmission(glm::vec3(0.0f), tint, 1.35f, 6.0f);
    const glm::vec3 againstWhite =
        lr::screen_space_liquid::compositeTransmission(glm::vec3(1.0f), tint, 1.35f, 6.0f);

    // The bulk tint remains visible over an empty black environment and both backgrounds retain
    // the selected cyan ordering. The old per-channel tint weight made green exceed blue.
    assert(againstBlack.b > 0.25f);
    assert(againstBlack.b > againstBlack.g);
    assert(againstBlack.g > againstBlack.r);
    assert(againstWhite.b > againstWhite.g);
    assert(againstWhite.g > againstWhite.r);
}

// The kernel must span roughly a particle diameter for neighbouring impostors to merge into one
// surface; anything narrower leaves each sphere with its own depth profile.
void smoothingRadiusSpansAParticleDiameter()
{
    using lr::screen_space_liquid::smoothingRadiusFromParticle;
    assert(near(smoothingRadiusFromParticle(0.07f), 0.14f));
    assert(near(smoothingRadiusFromParticle(0.0f), 0.0f));
    assert(near(smoothingRadiusFromParticle(-1.0f), 0.0f));
}

void thicknessUsesTheRasterizedVolumeBoundaries()
{
    using lr::screen_space_liquid::thicknessFromDepths;
    assert(near(thicknessFromDepths(2.0f, 5.5f), 3.5f));
    assert(near(thicknessFromDepths(2.0f, 5.5f, 0.5f), 1.75f));
    assert(near(thicknessFromDepths(5.5f, 2.0f), 0.0f));
    assert(near(thicknessFromDepths(2.0f, 5.5f, -1.0f), 0.0f));
}

} // namespace

int main()
{
    dispatchRoundsUpWithoutExtraGroups();
    waterFresnelMatchesPhysicalNormalIncidence();
    absorptionPreservesTheSelectedHue();
    bulkTintKeepsCyanVisibleWithoutTurningGreen();
    smoothingRadiusSpansAParticleDiameter();
    thicknessUsesTheRasterizedVolumeBoundaries();
    return 0;
}
