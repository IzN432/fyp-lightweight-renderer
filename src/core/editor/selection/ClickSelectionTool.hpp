#pragma once

#include "SelectionTool.hpp"

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>

namespace lr
{

class ClickSelectionTool : public SelectionTool
{
public:
    ClickSelectionTool(InputHandler &input, SceneObject &camera) : SelectionTool(input, camera) {}

    void onMouseDown(double ndcX, double ndcY, double aspect) override;
    void onMouseUp(double ndcX, double ndcY, double aspect) override;

    void selectVertices(std::unordered_set<uint32_t> &highlightedVertices,
                        std::unordered_set<uint32_t> &selectedVertices,
                        const std::vector<glm::vec3> &vertices) override;

private:
    static constexpr float kPickRadiusPixels = 12.0f;

    glm::vec2 m_clickPosition{};
    glm::mat4 m_viewProjectionMatrix{1.0f};
};

} // namespace lr
