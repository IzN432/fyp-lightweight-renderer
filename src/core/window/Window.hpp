#pragma once

#include <GLFW/glfw3.h>

#include <functional>
#include <string>
#include <vector>

namespace lr
{

// RAII guard for the global GLFW library state.
// Construct once before any Window, let it go out of scope to terminate.
class GlfwContext
{
public:
    GlfwContext();
    ~GlfwContext();

    GlfwContext(const GlfwContext &) = delete;
    GlfwContext &operator=(const GlfwContext &) = delete;

    static std::vector<const char *> getRequiredInstanceExtensions();
};

// ---------------------------------------------------------------------------

class Window
{
public:
    struct Config
    {
        int width = 1600;
        int height = 900;
        std::string title = "Renderer";
        bool resizable = true;
    };

public:
    explicit Window(const Config &config);
    ~Window();

    Window(const Window &) = delete;
    Window &operator=(const Window &) = delete;
    Window(Window &&) = delete;
    Window &operator=(Window &&) = delete;

    bool shouldClose() const;
    void pollEvents();

    int getWidth() const { return m_width; }
    int getHeight() const { return m_height; }
    bool wasResized() const { return m_resized; }
    void clearResizedFlag() { m_resized = false; }

    GLFWwindow *getHandle() const { return m_window; }

    void setResizeCallback(std::function<void(int, int)> cb) { m_resizeCallback = cb; }

private:
    static void glfwResizeCallback(GLFWwindow *window, int width, int height);

private:
    GLFWwindow *m_window = nullptr;
    int m_width = 0;
    int m_height = 0;
    bool m_resized = false;
    std::function<void(int, int)> m_resizeCallback;
};

}  // namespace lr
