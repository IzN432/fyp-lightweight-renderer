#include "InputHandler.hpp"

namespace lr
{

InputHandler::InputHandler() = default;

InputHandler::~InputHandler() = default;

void InputHandler::onKeyPress(std::function<void(int, int)> callback)
{
    m_keyPressCallbacks.push_back(callback);
}

void InputHandler::onMouseMove(std::function<void(double, double)> callback)
{
    m_mouseMoveCallbacks.push_back(callback);
}

void InputHandler::onMouseButton(std::function<void(int, int)> callback)
{
    m_mouseButtonCallbacks.push_back(callback);
}

void InputHandler::update()
{
    // This will be called each frame to poll input state and fire callbacks
    // For now, this is a placeholder. In a full implementation, you'd check
    // GLFW state here and fire callbacks based on state changes.
}

bool InputHandler::isKeyPressed(int key) const
{
    return m_pressedKeys.find(key) != m_pressedKeys.end();
}

void InputHandler::getMousePos(double &x, double &y) const
{
    x = m_lastMouseX;
    y = m_lastMouseY;
}

void InputHandler::notifyKey(int key, int action)
{
    if (action != 0)
        m_pressedKeys.insert(key);
    else
        m_pressedKeys.erase(key);

    for (auto &cb : m_keyPressCallbacks)
        cb(key, action);
}

void InputHandler::notifyMouseMove(double x, double y)
{
    m_lastMouseX = x;
    m_lastMouseY = y;

    for (auto &cb : m_mouseMoveCallbacks)
        cb(x, y);
}

void InputHandler::notifyMouseButton(int button, int action)
{
    for (auto &cb : m_mouseButtonCallbacks)
        cb(button, action);
}

}  // namespace lr
