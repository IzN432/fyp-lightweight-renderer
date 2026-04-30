#pragma once

#include "Component.hpp"

#include <unordered_map>
#include <typeindex>
#include <memory>
#include <stdexcept>
#include <imgui.h>

namespace lr
{

class SceneObject
{
    std::unordered_map<std::type_index, std::unique_ptr<Component>> components;
public:
    std::string name;
    
    template<typename T, typename... Args>
    T& addComponent(Args&&... args)
    {
        if (components.contains(std::type_index(typeid(T))))
        {
            throw std::runtime_error("Component of this type already exists on this object");
        }
        auto component = std::make_unique<T>(std::forward<Args>(args)...);
        T& ref = *component;
        ref.m_owningObject = this;
        components[std::type_index(typeid(T))] = std::move(component);
        return ref;
    }

    template<typename T>
    T& getComponent() const
    {
        auto it = components.find(std::type_index(typeid(T)));
        if (it == components.end()) throw std::runtime_error("Component not found");
        return static_cast<T&>(*it->second);
    }

    template<typename T>
    bool hasComponent() const
    {
        return components.contains(std::type_index(typeid(T)));
    };

    void onGUI()
    {
        int id = 0;
        if (ImGui::CollapsingHeader(name.c_str()))
        {
            for (auto& [type, component] : components)
            {
                ImGui::PushID(id++);
                component->onGUI();
                ImGui::PopID();
            }
        }
    }
};

}