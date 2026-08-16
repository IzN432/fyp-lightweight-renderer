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
    virtual void selectVertices(std::vector<uint32_t> &highlightedVertices, std::vector<uint32_t> &selectedVertices,
                                const std::vector<glm::vec3> &vertices) {};
    virtual void highlightVertices(std::vector<uint32_t> &highlightedVertices, std::vector<uint32_t> &selectedVertices,
                                const std::vector<glm::vec3> &vertices) {};

    // Color SelectionManager paints highlighted vertices with — override to vary by tool state
    // (e.g. a different color while a modifier key changes what the in-progress drag will do to
    // the selection). Queried once per highlight rebuild, applied to every highlighted vertex.
    virtual glm::vec3 highlightColor() const { return glm::vec3(1.0f, 0.8f, 0.0f); }

    void registerSelectionCallback(std::function<void()> callback) { m_selectionCallback = std::move(callback); }
    void registerHighlightCallback(std::function<void()> callback) { m_highlightCallback = std::move(callback); }

protected:
    SceneObject          &m_camera;
    InputHandler         &m_input;
    std::function<void()> m_selectionCallback;
    std::function<void()> m_highlightCallback;
};

} // namespace lr