#pragma once

#include "SelectionTool.hpp"

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>

namespace lr
{

class BoxSelectionTool : public SelectionTool
{
public:
    BoxSelectionTool(InputHandler &input, SceneObject &camera) : SelectionTool(input, camera) {}

    void onMouseDown(double ndcX, double ndcY, double aspect) override;
    void onMouseUp(double ndcX, double ndcY, double aspect) override;
    void dragCallback(double ndcX, double ndcY, double dNdcX, double dNdcY, double aspect) override;


    void selectVertices(std::vector<uint32_t> &selectedVertices, const std::vector<glm::vec3> &vertices) override;
private:
    glm::vec2 m_boxStart;
    glm::vec2 m_boxEnd;
    bool m_isDragging = false;

    glm::mat4 m_viewProjectionMatrix;
};
} // namespace lr