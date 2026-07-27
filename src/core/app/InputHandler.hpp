#pragma once

#include <functional>
#include <unordered_set>
#include <vector>
#include <GLFW/glfw3.h>

namespace lr
{

class InputHandler
{
public:
    InputHandler();
    ~InputHandler();

    // Register a callback for key press events
    void onKeyPress(std::function<void(int key, int action, bool shift, bool ctrl, bool alt)> callback);

    // Register a callback for mouse button events. Query getMousePos()/getMouseDelta()
    // in an update callback if position is needed — mouse move no longer fires events.
    void onMouseButton(std::function<void(int button, int action, bool shift, bool ctrl, bool alt)> callback);

    // Call this each frame to poll input state
    void update();

    // Query current key/button state
    bool isKeyPressed(int key) const;
    bool isMouseButtonPressed(int button) const;

    bool isShiftPressed() const {
        return isKeyPressed(GLFW_KEY_LEFT_SHIFT) || isKeyPressed(GLFW_KEY_RIGHT_SHIFT);
    }
    bool isCtrlPressed() const {
        return isKeyPressed(GLFW_KEY_LEFT_CONTROL) || isKeyPressed(GLFW_KEY_RIGHT_CONTROL);
    }
    bool isAltPressed() const {
        return isKeyPressed(GLFW_KEY_LEFT_ALT) || isKeyPressed(GLFW_KEY_RIGHT_ALT);
    }
    
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
    std::vector<std::function<void(int, int, bool, bool, bool)>>    m_keyPressCallbacks;
    std::vector<std::function<void(int, int, bool, bool, bool)>>    m_mouseButtonCallbacks;

    std::unordered_set<int> m_pressedKeys;
    std::unordered_set<int> m_pressedButtons;

    double m_currentMouseX = 0.0, m_currentMouseY = 0.0;
    double m_prevMouseX    = 0.0, m_prevMouseY    = 0.0;
    double m_deltaMouseX   = 0.0, m_deltaMouseY   = 0.0;

    double m_scrollAccum = 0.0;
    double m_scrollDelta = 0.0;
};

}  // namespace lr
