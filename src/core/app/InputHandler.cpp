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
    m_deltaMouseX = m_currentMouseX - m_prevMouseX;
    m_deltaMouseY = m_currentMouseY - m_prevMouseY;
    m_prevMouseX  = m_currentMouseX;
    m_prevMouseY  = m_currentMouseY;

    m_scrollDelta = m_scrollAccum;
    m_scrollAccum = 0.0;
}

bool InputHandler::isKeyPressed(int key) const
{
    return m_pressedKeys.contains(key);
}

bool InputHandler::isMouseButtonPressed(int button) const
{
    return m_pressedButtons.contains(button);
}

void InputHandler::getMouseDelta(double &dx, double &dy) const
{
    dx = m_deltaMouseX;
    dy = m_deltaMouseY;
}

double InputHandler::getScrollDelta() const
{
    return m_scrollDelta;
}

void InputHandler::getMousePos(double &x, double &y) const
{
    x = m_currentMouseX;
    y = m_currentMouseY;
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
    m_currentMouseX = x;
    m_currentMouseY = y;

    for (auto &cb : m_mouseMoveCallbacks)
        cb(x, y);
}

void InputHandler::notifyMouseButton(int button, int action)
{
    if (action != 0)
        m_pressedButtons.insert(button);
    else
        m_pressedButtons.erase(button);

    for (auto &cb : m_mouseButtonCallbacks)
        cb(button, action);
}

void InputHandler::notifyScroll(double delta)
{
    m_scrollAccum += delta;
}

}  // namespace lr
