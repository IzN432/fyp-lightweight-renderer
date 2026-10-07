#pragma once

#include "BoxSelectionTool.hpp"
#include "ClickSelectionTool.hpp"

namespace lr
{

class SelectionGestureTool : public SelectionTool
{
public:
    SelectionGestureTool(InputHandler &input, SceneObject &camera);

    void onMouseDown(double ndcX, double ndcY, double aspect) override;
    void onMouseUp(double ndcX, double ndcY, double aspect) override;
    void dragCallback(double ndcX, double ndcY, double dNdcX, double dNdcY, double aspect) override;
    void setViewportExtent(VkExtent2D extent) override;

    void selectVertices(std::unordered_set<uint32_t> &highlightedVertices,
                        std::unordered_set<uint32_t> &selectedVertices,
                        const std::vector<glm::vec3> &vertices) override;
    void highlightVertices(std::unordered_set<uint32_t> &highlightedVertices,
                           std::unordered_set<uint32_t> &selectedVertices,
                           const std::vector<glm::vec3> &vertices) override;

private:
    enum class Gesture
    {
        None,
        Pending,
        Click,
        Box
    };

    static constexpr float kDragThresholdPixels = 4.0f;

    ClickSelectionTool m_clickTool;
    BoxSelectionTool   m_boxTool;
    Gesture            m_gesture = Gesture::None;
    glm::vec2          m_start{};
};

} // namespace lr
