#pragma once

#include <string>
#include <imgui.h>
#include <vector>
#include <functional>

namespace lr
{

class Component
{
private:
friend class SceneObject;
    SceneObject *m_owningObject = nullptr;
    std::string m_name;
    std::vector<std::function<void()>> m_listeners;
protected:
    const SceneObject& getOwningObject() const { return *m_owningObject; }
    void notifyChanged() { for (const auto& listener : m_listeners) listener(); }
public:
    Component(std::string name = "")
        : m_name(std::move(name)) {}
    virtual ~Component() = default;

    void onGUI() 
    { 
        ImGui::Text("Component: %s", m_name.c_str());
        return onGUIImpl();
    }

    virtual void onGUIImpl() {}

    void addChangeListener(std::function<void()> listener)
    {
        m_listeners.push_back(std::move(listener));
    }
};

}