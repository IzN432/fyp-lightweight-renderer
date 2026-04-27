#pragma once

#include <string>
#include <imgui.h>

namespace lr
{

class Component
{
private:
friend class SceneObject;
    SceneObject *m_owningObject = nullptr;
    std::string m_name;
protected:
    const SceneObject& getOwningObject() const { return *m_owningObject; }
public:
    Component(std::string name = "") : m_name(std::move(name)) {}
    virtual ~Component() = default;

    bool onGUI() 
    { 
        ImGui::Text("Component: %s", m_name.c_str());
        return onGUIImpl();
    }

    virtual bool onGUIImpl() 
    {
        return false; // return true if this component's data was changed
    };
};

}