#include "HeatmapColors.hpp"

#include <glm/common.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace lr
{

namespace
{
glm::vec3 heatmapRamp(float value)
{
    return glm::clamp(glm::vec3(1.5f) - glm::abs(4.0f * value - glm::vec3(3.0f, 2.0f, 1.0f)),
                      glm::vec3(0.0f), glm::vec3(1.0f));
}
} // namespace

std::vector<glm::vec3> makeHeatmapColors(std::span<const float> values, float upperPercentile)
{
    std::vector<float> finiteValues;
    finiteValues.reserve(values.size());
    for (float value : values)
    {
        if (std::isfinite(value))
        {
            finiteValues.push_back(value);
        }
    }

    std::vector<glm::vec3> colors(values.size(), heatmapRamp(0.0f));
    if (finiteValues.empty())
    {
        return colors;
    }

    std::sort(finiteValues.begin(), finiteValues.end());
    upperPercentile        = std::clamp(upperPercentile, 0.0f, 1.0f);
    const size_t upperIndex = static_cast<size_t>(upperPercentile * static_cast<float>(finiteValues.size() - 1));
    const float  upper      = finiteValues[upperIndex];
    if (upper <= std::numeric_limits<float>::epsilon())
    {
        return colors;
    }

    for (size_t i = 0; i < values.size(); ++i)
    {
        if (std::isfinite(values[i]))
        {
            const float t = std::clamp(values[i] / upper, 0.0f, 1.0f);
            colors[i]     = heatmapRamp(t);
        }
    }
    return colors;
}

} // namespace lr
