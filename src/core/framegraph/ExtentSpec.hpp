#pragma once

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace lr
{

enum class ExtentRounding
{
    Down,
    Up,
};

// Symbolic image size resolved against the current swapchain extent. Integer
// ratios keep resize behaviour deterministic, including for odd dimensions.
class ExtentSpec
{
public:
    enum class Kind
    {
        Absolute,
        SwapchainRelative,
    };

    static ExtentSpec absolute(uint32_t width, uint32_t height)
    {
        if (width == 0 || height == 0)
        {
            throw std::invalid_argument("FrameGraph: absolute extent must be non-zero");
        }
        return ExtentSpec(Kind::Absolute, width, 1, height, 1, ExtentRounding::Down);
    }

    static ExtentSpec swapchain() { return ExtentSpec{}; }

    static ExtentSpec relative(uint32_t numerator, uint32_t denominator, ExtentRounding rounding = ExtentRounding::Up)
    {
        return relative(numerator, denominator, numerator, denominator, rounding);
    }

    static ExtentSpec relative(uint32_t widthNumerator, uint32_t widthDenominator, uint32_t heightNumerator,
                               uint32_t heightDenominator, ExtentRounding rounding = ExtentRounding::Up)
    {
        if (widthNumerator == 0 || heightNumerator == 0 || widthDenominator == 0 || heightDenominator == 0)
        {
            throw std::invalid_argument("FrameGraph: relative extent ratios must be non-zero");
        }
        const uint32_t widthDivisor  = std::gcd(widthNumerator, widthDenominator);
        const uint32_t heightDivisor = std::gcd(heightNumerator, heightDenominator);
        return ExtentSpec(Kind::SwapchainRelative, widthNumerator / widthDivisor, widthDenominator / widthDivisor,
                          heightNumerator / heightDivisor, heightDenominator / heightDivisor, rounding);
    }

    VkExtent2D resolve(VkExtent2D swapchainExtent) const
    {
        if (m_kind == Kind::Absolute)
        {
            return {m_widthNumerator, m_heightNumerator};
        }

        return {resolveAxis(swapchainExtent.width, m_widthNumerator, m_widthDenominator),
                resolveAxis(swapchainExtent.height, m_heightNumerator, m_heightDenominator)};
    }

    Kind kind() const { return m_kind; }

    std::string describe() const
    {
        if (m_kind == Kind::Absolute)
        {
            return std::to_string(m_widthNumerator) + "x" + std::to_string(m_heightNumerator);
        }
        if (m_widthNumerator == 1 && m_widthDenominator == 1 && m_heightNumerator == 1 && m_heightDenominator == 1)
        {
            return "swapchain";
        }
        return "swapchain*(" + std::to_string(m_widthNumerator) + "/" + std::to_string(m_widthDenominator) + "," +
               std::to_string(m_heightNumerator) + "/" + std::to_string(m_heightDenominator) + ")";
    }

    friend bool operator==(const ExtentSpec &, const ExtentSpec &) = default;

private:
    ExtentSpec() = default;
    ExtentSpec(Kind kind, uint32_t widthNumerator, uint32_t widthDenominator, uint32_t heightNumerator,
               uint32_t heightDenominator, ExtentRounding rounding)
        : m_kind(kind), m_widthNumerator(widthNumerator), m_widthDenominator(widthDenominator),
          m_heightNumerator(heightNumerator), m_heightDenominator(heightDenominator), m_rounding(rounding)
    {}

    uint32_t resolveAxis(uint32_t base, uint32_t numerator, uint32_t denominator) const
    {
        uint64_t scaled = static_cast<uint64_t>(base) * numerator;
        if (m_rounding == ExtentRounding::Up)
        {
            scaled += denominator - 1;
        }
        const uint64_t result = std::max<uint64_t>(1, scaled / denominator);
        if (result > std::numeric_limits<uint32_t>::max())
        {
            throw std::overflow_error("FrameGraph: resolved extent exceeds uint32_t");
        }
        return static_cast<uint32_t>(result);
    }

    Kind           m_kind              = Kind::SwapchainRelative;
    uint32_t       m_widthNumerator    = 1;
    uint32_t       m_widthDenominator  = 1;
    uint32_t       m_heightNumerator   = 1;
    uint32_t       m_heightDenominator = 1;
    ExtentRounding m_rounding          = ExtentRounding::Up;
};

} // namespace lr
