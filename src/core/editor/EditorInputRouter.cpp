#include "EditorInputRouter.hpp"

#include <stdexcept>
#include <utility>

namespace lr
{

EditorInputRouter::EditorInputRouter(CaptureQuery uiCapturesPointer, CaptureQuery gizmoCapturesPointer)
    : m_uiCapturesPointer(std::move(uiCapturesPointer)), m_gizmoCapturesPointer(std::move(gizmoCapturesPointer))
{
    if (!m_uiCapturesPointer || !m_gizmoCapturesPointer)
    {
        throw std::invalid_argument("EditorInputRouter: capture queries cannot be empty");
    }
}

void EditorInputRouter::addButtonLayer(std::string name, ButtonLayer layer)
{
    if (!layer)
    {
        throw std::invalid_argument("EditorInputRouter: layer handler cannot be empty for " + name);
    }
    m_layers.push_back({std::move(name), std::move(layer)});
}

bool EditorInputRouter::pointerCaptured() const { return m_uiCapturesPointer() || m_gizmoCapturesPointer(); }

bool EditorInputRouter::routeButton(const PointerButtonEvent &event) const
{
    if (pointerCaptured())
    {
        return false;
    }
    for (const Layer &layer : m_layers)
    {
        if (layer.handler(event))
        {
            return true;
        }
    }
    return false;
}

bool EditorInputRouter::viewportNavigationAllowed() const { return !m_uiCapturesPointer() || m_gizmoCapturesPointer(); }

} // namespace lr
