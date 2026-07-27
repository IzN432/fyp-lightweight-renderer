#pragma once

#include "core/app/InputHandler.hpp"

#include <vector>
#include <glm/vec3.hpp>
#include <functional>

namespace lr
{

class SceneObject;

class SelectionTool
{
public:
    SelectionTool(InputHandler &input, SceneObject &camera) : m_input(input), m_camera(camera) {}
    virtual ~SelectionTool() = default; 

    // Mouse interaction callbacks: override these in derived classes to implement tool behavior.
    // Same convention as Gizmo — ndcX/ndcY are normalized device coordinates, dNdcX/dNdcY
    // are the per-frame NDC delta.
    virtual void onMouseDown(double ndcX, double ndcY, double aspect) {}
    virtual void onMouseUp(double ndcX, double ndcY, double aspect) {}
    virtual void dragCallback(double ndcX, double ndcY, double dNdcX, double dNdcY, double aspect) {}

    // Selection callback: override this in derived classes to implement selection behavior
    virtual void selectVertices(std::vector<uint32_t> &selectedVertices, const std::vector<glm::vec3> &vertices) {}

    void registerSelectionCallback(std::function<void()> callback) { m_selectionCallback = std::move(callback); }
protected:
    SceneObject &m_camera;
    InputHandler &m_input;
    std::function<void()> m_selectionCallback;
};

} // namespace lr