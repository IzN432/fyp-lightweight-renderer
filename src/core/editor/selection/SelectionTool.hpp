#pragma once

#include "core/app/InputHandler.hpp"

#include <vector>
#include <unordered_set>
#include <glm/vec3.hpp>
#include <functional>
#include <vulkan/vulkan.h>

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
    virtual void setViewportExtent(VkExtent2D extent) { m_viewportExtent = extent; }

    // Selection callback: override this in derived classes to implement selection behavior
    virtual void selectVertices(std::unordered_set<uint32_t> &highlightedVertices,
                                std::unordered_set<uint32_t> &selectedVertices,
                                const std::vector<glm::vec3> &vertices) {};
    virtual void highlightVertices(std::unordered_set<uint32_t> &highlightedVertices,
                                   std::unordered_set<uint32_t> &selectedVertices,
                                   const std::vector<glm::vec3> &vertices) {};

    // Color SelectionManager paints highlighted vertices with — override to vary by tool state
    // (e.g. a different color while a modifier key changes what the in-progress drag will do to
    // the selection). Queried once per highlight rebuild, applied to every highlighted vertex.
    virtual glm::vec3 highlightColor() const { return glm::vec3(1.0f, 0.8f, 0.0f); }

    CallbackConnection registerSelectionCallback(std::function<void()> callback)
    {
        return m_selectionCallbacks.connect(std::move(callback));
    }
    CallbackConnection registerHighlightCallback(std::function<void()> callback)
    {
        return m_highlightCallbacks.connect(std::move(callback));
    }

protected:
    SceneObject          &m_camera;
    InputHandler         &m_input;
    CallbackList<>        m_selectionCallbacks;
    CallbackList<>        m_highlightCallbacks;
    VkExtent2D            m_viewportExtent{};
};

} // namespace lr
