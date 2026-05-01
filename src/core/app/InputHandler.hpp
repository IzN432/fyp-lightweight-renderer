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

    // Query current key/button state
    bool isKeyPressed(int key) const;
    bool isMouseButtonPressed(int button) const;

    // Per-frame polling — valid after update() is called each frame
    void   getMouseDelta(double &dx, double &dy) const;
    double getScrollDelta() const;

    // Get current mouse position
    void getMousePos(double &x, double &y) const;

    // Feed events from the window/input backend.
    void notifyKey(int key, int action);
    void notifyMouseMove(double x, double y);
    void notifyMouseButton(int button, int action);
    void notifyScroll(double delta);

private:
    std::vector<std::function<void(int, int)>>    m_keyPressCallbacks;
    std::vector<std::function<void(double, double)>> m_mouseMoveCallbacks;
    std::vector<std::function<void(int, int)>>    m_mouseButtonCallbacks;

    std::unordered_set<int> m_pressedKeys;
    std::unordered_set<int> m_pressedButtons;

    double m_currentMouseX = 0.0, m_currentMouseY = 0.0;
    double m_prevMouseX    = 0.0, m_prevMouseY    = 0.0;
    double m_deltaMouseX   = 0.0, m_deltaMouseY   = 0.0;

    double m_scrollAccum = 0.0;
    double m_scrollDelta = 0.0;
};

}  // namespace lr
