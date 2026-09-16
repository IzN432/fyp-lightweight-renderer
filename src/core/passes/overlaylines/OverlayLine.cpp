#include "core/passes/overlaylines/OverlayLine.hpp"

#include <stdexcept>

namespace lr
{
namespace
{

glm::vec3 transformPoint(const glm::mat4 &model, const glm::vec3 &point)
{
    return glm::vec3(model * glm::vec4(point, 1.0f));
}

} // namespace

void OverlayLineBuilder::addLine(const glm::vec3 &start, const glm::vec3 &end, const OverlayLineStyle &style)
{
    m_lines.push_back({.start = start, .end = end, .style = style});
}

void OverlayLineBuilder::addPrimitive(const OverlayLineMeshData &outline, const glm::mat4 &model,
                                      const OverlayLineStyle &style)
{
    if (!outline.colors.empty() && outline.colors.size() != outline.positions.size())
    {
        throw std::invalid_argument("Overlay line primitive colors must be empty or match its positions");
    }
    for (const glm::uvec2 &edge : outline.edges)
    {
        if (edge.x >= outline.positions.size() || edge.y >= outline.positions.size())
        {
            throw std::out_of_range("Overlay line primitive edge index is out of range");
        }

        OverlayLineStyle edgeStyle = style;
        if (!outline.colors.empty())
        {
            edgeStyle.color *= (outline.colors[edge.x] + outline.colors[edge.y]) * 0.5f;
        }
        addLine(transformPoint(model, outline.positions[edge.x]), transformPoint(model, outline.positions[edge.y]),
                edgeStyle);
    }
}

} // namespace lr
