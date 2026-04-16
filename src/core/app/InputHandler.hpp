#pragma once

#include <functional>
#include <unordered_set>
#include <vector>

namespace lr
{

class InputHandler
{
public:
    InputHandler();
    ~InputHandler();

    // Register a callback for key press events
    void onKeyPress(std::function<void(int key, int action)> callback);

    // Register a callback for mouse move events
    void onMouseMove(std::function<void(double x, double y)> callback);

    // Register a callback for mouse button events
    void onMouseButton(std::function<void(int button, int action)> callback);

    // Call this each frame to poll input state
    void update();

    // Query current key state
    bool isKeyPressed(int key) const;

    // Get current mouse position
    void getMousePos(double &x, double &y) const;

    // Feed events from the window/input backend.
    void notifyKey(int key, int action);
    void notifyMouseMove(double x, double y);
    void notifyMouseButton(int button, int action);

private:
    std::vector<std::function<void(int, int)>> m_keyPressCallbacks;
    std::vector<std::function<void(double, double)>> m_mouseMoveCallbacks;
    std::vector<std::function<void(int, int)>> m_mouseButtonCallbacks;

    std::unordered_set<int> m_pressedKeys;

    double m_lastMouseX = 0.0;
    double m_lastMouseY = 0.0;
};

}  // namespace lr
