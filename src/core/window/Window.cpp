#include "Window.hpp"

#include <spdlog/spdlog.h>

#include <stdexcept>

namespace lr
{

// ---------------------------------------------------------------------------
// GlfwContext
// ---------------------------------------------------------------------------

GlfwContext::GlfwContext()
{
    if (!glfwInit())
    {
        throw std::runtime_error("GlfwContext: failed to initialise GLFW");
    }

    spdlog::info("GlfwContext: GLFW initialised");
}

GlfwContext::~GlfwContext()
{
    glfwTerminate();
}

std::vector<const char *> GlfwContext::getRequiredInstanceExtensions()
{
    uint32_t count = 0;
    const char **extensions = glfwGetRequiredInstanceExtensions(&count);
    if (!extensions)
    {
        throw std::runtime_error("GlfwContext: could not get required Vulkan extensions");
    }

    return std::vector<const char *>(extensions, extensions + count);
}

// ---------------------------------------------------------------------------
// Window
// ---------------------------------------------------------------------------

Window::Window(const Config &config)
{
    m_width = config.width;
    m_height = config.height;

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, config.resizable ? GLFW_TRUE : GLFW_FALSE);

    m_window = glfwCreateWindow(m_width, m_height, config.title.c_str(), nullptr, nullptr);
    if (!m_window)
    {
        throw std::runtime_error("Window: failed to create GLFW window");
    }

    glfwSetWindowUserPointer(m_window, this);
    glfwSetFramebufferSizeCallback(m_window, glfwResizeCallback);
    glfwSetKeyCallback(m_window, glfwKeyCallback);
    glfwSetCursorPosCallback(m_window, glfwCursorPosCallback);
    glfwSetMouseButtonCallback(m_window, glfwMouseButtonCallback);
    glfwSetScrollCallback(m_window, glfwScrollCallback);

    spdlog::info("Window: created ({}x{})", m_width, m_height);
}

Window::~Window()
{
    glfwDestroyWindow(m_window);
}

bool Window::shouldClose() const
{
    return glfwWindowShouldClose(m_window);
}

void Window::pollEvents()
{
    glfwPollEvents();
}

// ---------------------------------------------------------------------------
// Private
// ---------------------------------------------------------------------------

void Window::glfwResizeCallback(GLFWwindow *window, int width, int height)
{
    auto *self = static_cast<Window *>(glfwGetWindowUserPointer(window));
    self->m_width = width;
    self->m_height = height;
    self->m_resized = true;

    if (self->m_resizeCallback)
        self->m_resizeCallback(width, height);
}

void Window::glfwKeyCallback(GLFWwindow *window, int key, int /*scancode*/, int action, int /*mods*/)
{
    auto *self = static_cast<Window *>(glfwGetWindowUserPointer(window));
    if (self->m_keyCallback)
        self->m_keyCallback(key, action);
}

void Window::glfwCursorPosCallback(GLFWwindow *window, double x, double y)
{
    auto *self = static_cast<Window *>(glfwGetWindowUserPointer(window));
    if (self->m_cursorPosCallback)
        self->m_cursorPosCallback(x, y);
}

void Window::glfwMouseButtonCallback(GLFWwindow *window, int button, int action, int /*mods*/)
{
    auto *self = static_cast<Window *>(glfwGetWindowUserPointer(window));
    if (self->m_mouseButtonCallback)
        self->m_mouseButtonCallback(button, action);
}

void Window::glfwScrollCallback(GLFWwindow *window, double /*xoffset*/, double yoffset)
{
    auto *self = static_cast<Window *>(glfwGetWindowUserPointer(window));
    if (self->m_scrollCallback)
        self->m_scrollCallback(yoffset);
}

}  // namespace lr
