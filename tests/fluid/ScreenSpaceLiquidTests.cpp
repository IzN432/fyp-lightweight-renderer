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

} // namespace

int main()
{
    dispatchRoundsUpWithoutExtraGroups();
    waterFresnelMatchesPhysicalNormalIncidence();
    absorptionPreservesTheSelectedHue();
    return 0;
}
