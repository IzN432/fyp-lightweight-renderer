#pragma once

#include "core/overlay/PrimitiveOverlayMeshes.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <utility>
#include <vector>

namespace lr
{

struct OverlayLineStyle
{
    glm::vec3 color{1.0f, 0.0f, 1.0f};
    float     visibleOpacity  = 1.0f;
    float     occludedOpacity = 0.15f;
};

struct OverlayLine
{
    glm::vec3        start{0.0f};
    glm::vec3        end{0.0f};
    OverlayLineStyle style;
};

// CPU-side helpers for producing line segments. The renderer only consumes the resulting
// OverlayLine list and remains independent of colliders, contacts, gizmos, or other producers.
class OverlayLineBuilder
{
public:
    void addLine(const glm::vec3 &start, const glm::vec3 &end, const OverlayLineStyle &style = {});
    void addPrimitive(const OverlayLineMeshData &outline, const glm::mat4 &model,
                      const OverlayLineStyle &style = {});

    const std::vector<OverlayLine> &lines() const { return m_lines; }
    std::vector<OverlayLine>        takeLines() { return std::move(m_lines); }

private:
    std::vector<OverlayLine> m_lines;
};

} // namespace lr
