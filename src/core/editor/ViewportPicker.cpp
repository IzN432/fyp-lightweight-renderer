#include "ViewportPicker.hpp"

#include "core/app/Viewer.hpp"

#include <GLFW/glfw3.h>
#include <glm/geometric.hpp>
#include <imgui.h>

#include <algorithm>

namespace lr
{

ViewportPicker::ViewportPicker(const VulkanContext &ctx, Allocator &alloc) : m_readback(ctx, alloc) {}

uint32_t ViewportPicker::readIdAtCursor(Viewer &viewer, const std::string &imageName)
{
    if (!viewer.hasRenderedAtLeastOneFrame())
    {
        return kNoData;
    }

    double x = 0.0, y = 0.0;
    viewer.input().getMousePos(x, y);
    const VkExtent2D extent  = viewer.resources().getExtent();
    const ImVec2     display = ImGui::GetIO().DisplaySize;
    if (extent.width == 0 || extent.height == 0 || display.x <= 0.0f || display.y <= 0.0f || x < 0.0 || y < 0.0 ||
        x >= display.x || y >= display.y)
    {
        return kNoData;
    }

    const uint32_t pixelX = std::min(static_cast<uint32_t>(x * extent.width / display.x), extent.width - 1);
    const uint32_t pixelY = std::min(static_cast<uint32_t>(y * extent.height / display.y), extent.height - 1);
    return m_readback.readPixel(viewer.resources(), imageName, pixelX, pixelY);
}

bool ViewportPicker::consumes(int action) const
{
    return action == GLFW_PRESS || (action == GLFW_RELEASE && m_press.has_value());
}

bool ViewportPicker::clickCompleted(Viewer &viewer, int action)
{
    double x = 0.0, y = 0.0;
    viewer.input().getMousePos(x, y);

    if (action == GLFW_PRESS)
    {
        m_press = glm::dvec2(x, y);
        return false;
    }
    if (action != GLFW_RELEASE || !m_press)
    {
        return false;
    }

    const glm::dvec2 release(x, y);
    const glm::dvec2 travel = release - *m_press;
    m_press.reset();
    return glm::dot(travel, travel) <= kClickRadius * kClickRadius;
}

} // namespace lr
