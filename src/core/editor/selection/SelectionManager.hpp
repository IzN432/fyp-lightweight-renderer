#pragma once

#include "SelectionTool.hpp"

#include "core/app/InputHandler.hpp"

#include <functional>
#include <memory>
#include <vulkan/vulkan.h>

namespace lr
{

class SelectionManager
{
public:
    SelectionManager(const std::vector<glm::vec3> &vertices, InputHandler &input)
        : m_vertices(vertices), m_input(input) {}
    ~SelectionManager() = default;

    void setSelectTool(std::unique_ptr<SelectionTool> tool);

    // Dispatches the callback of the active selection tool
    void mouseButtonCallback(int button, int action, bool shift, bool ctrl, bool alt);

    // Updates the state of the active selection tool (e.g. drag tracking).
    // Flags set by mouseButtonCallback are consumed on the first update() after they're set.
    void updateCallback(float dt, VkExtent2D extent);

    const std::vector<uint32_t> &getSelectedIndices() const { return m_selectedVertices; }

    void clearSelection();

    // Invoked whenever the selected vertex set changes (click, box select, or clearSelection()).
    void registerSelectionChangedCallback(std::function<void()> callback) { m_selectionChangedCallback = std::move(callback); }
private:
    std::unique_ptr<SelectionTool> m_selectTool;
    std::vector<uint32_t> m_selectedVertices;
    const std::vector<glm::vec3> &m_vertices;
    InputHandler &m_input;
    bool m_mouseClickedThisFrame = false;
    bool m_mouseReleasedThisFrame = false;
    std::function<void()> m_selectionChangedCallback;
};


} // namespace lr