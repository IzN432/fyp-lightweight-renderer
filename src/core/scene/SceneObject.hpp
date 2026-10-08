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
class OverlayLineBuilder;

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
    glm::quat worldRotation() const;

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

    // For code holding a component type it cannot name, such as the inspector asking whether this
    // object already has whatever is on the component clipboard.
    bool hasComponent(std::type_index type) const { return components.contains(type); }

    // Finds a component by interface as well as by its concrete stored type. Components are keyed by
    // their concrete type, so editor-facing contracts such as CameraController need the polymorphic
    // fallback when the object actually stores a SphericalCameraController.
    template <typename T> T *findComponent()
    {
        static_assert(std::is_base_of_v<Component, T>, "T must derive from Component");
        for (auto &entry : components)
        {
            if (auto *match = dynamic_cast<T *>(entry.second.get()))
            {
                return match;
            }
        }
        return nullptr;
    }

    template <typename T> const T *findComponent() const
    {
        static_assert(std::is_base_of_v<Component, T>, "T must derive from Component");
        for (const auto &entry : components)
        {
            if (const auto *match = dynamic_cast<const T *>(entry.second.get()))
            {
                return match;
            }
        }
        return nullptr;
    }

    size_t componentCount() const { return components.size(); }

    // Runtime component types, primarily for persistence/diagnostics. Components remain owned and
    // accessed through the typed API above.
    std::vector<std::type_index> componentTypes() const
    {
        std::vector<std::type_index> result;
        result.reserve(components.size());
        for (const auto &[type, component] : components)
        {
            result.push_back(type);
        }
        return result;
    }

    void onGUI(EditorContext &context)
    {
        int                            id = 0;
        std::optional<std::type_index> removalRequest;
        for (auto &[type, component] : components)
        {
            ImGui::PushID(id++);
            component->onGUI(context, removalRequest);
            ImGui::PopID();
        }
        // Erased only once the loop is over. A component's own context menu is what asks for this,
        // so erasing where it was asked would invalidate this iteration and destroy the component
        // whose method is still running.
        if (removalRequest)
        {
            components.erase(*removalRequest);
        }
    }

    // Runs Component::onLoaded on every component. Called by scene loading once the whole scene
    // exists, so a component may reach its siblings here.
    void onLoaded()
    {
        for (auto &[type, component] : components)
        {
            component->onLoaded();
        }
    }

    // Gives every component an opportunity to visualize itself when this object is selected.
    void onSelectGizmo(SelectionGizmoContext &context) const
    {
        for (const auto &[type, component] : components)
        {
            component->onSelectGizmo(context);
        }
    }
};

} // namespace lr
