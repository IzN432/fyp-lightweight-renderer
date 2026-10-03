#pragma once

#include "Component.hpp"
#include "SceneObjectId.hpp"

#include <glm/glm.hpp>

#include <unordered_map>
#include <cstdint>
#include <optional>
#include <vector>
#include <typeindex>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <imgui.h>

namespace lr
{

class Scene;

class SceneObject
{
    friend class Scene;

    explicit SceneObject(Scene &scene, SceneObjectId id) : m_scene(&scene), m_id(id) {}

    std::unordered_map<std::type_index, std::unique_ptr<Component>> components;
    Scene                                                    *m_scene;
    SceneObjectId                                             m_id;
    std::optional<SceneObjectId>                              m_parent;
    std::vector<SceneObjectId>                                m_children;
    bool                                                       m_alive = true;

public:
    std::string name;

    SceneObjectId id() const { return m_id; }
    Scene &scene() const { return *m_scene; }
    std::optional<SceneObjectId> parent() const { return m_parent; }
    const std::vector<SceneObjectId> &children() const { return m_children; }

    glm::mat4 worldMatrix() const;

    template <typename T, typename... Args> T &addComponent(Args &&...args)
    {
        static_assert(std::is_base_of_v<Component, T>, "T must derive from Component");

        if (components.contains(std::type_index(typeid(T))))
        {
            throw std::runtime_error("Component of this type already exists on this object");
        }

        auto component                         = std::make_unique<T>(std::forward<Args>(args)...);
        T   &ref                               = *component;
        ref.m_owningObject                     = this;
        components[std::type_index(typeid(T))] = std::move(component);
        return ref;
    }

    template <typename T> T &getComponent() const
    {
        auto it = components.find(std::type_index(typeid(T)));
        if (it == components.end())
        {
            throw std::runtime_error("Component not found");
        }
        return static_cast<T &>(*it->second);
    }

    template <typename T> bool hasComponent() const { return components.contains(std::type_index(typeid(T))); };

    void onGUI(EditorContext &context)
    {
        int  id = 0;
        bool first = true;
        for (auto &[type, component] : components)
        {
            if (!first)
            {
                ImGui::Separator();
            }
            first = false;
            ImGui::PushID(id++);
            component->onGUI(context);
            ImGui::PopID();
        }
    }
};

} // namespace lr
