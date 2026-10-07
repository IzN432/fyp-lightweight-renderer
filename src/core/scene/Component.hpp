#pragma once

#include <glm/glm.hpp>

#include <string>
#include <imgui.h>

namespace lr
{

struct EditorContext;
class SceneObject;
class OverlayLineBuilder;

struct SelectionGizmoContext
{
    OverlayLineBuilder &lines;
    glm::vec3           cameraPosition{0.0f};
    glm::vec3           cameraForward{0.0f, 0.0f, -1.0f};
    bool                orthographic = false;
};

class Component
{
private:
    friend class SceneObject;
    SceneObject *m_owningObject = nullptr;
    std::string  m_name;
    // Set by markDirty() whenever this component's state changes, and cleared by whoever
    // consumes it (see SceneManager::flushDirty) once its GPU-facing data has been re-uploaded.
    // This is a poll-based replacement for the old push-listener model: setters just flag
    // themselves dirty instead of eagerly triggering a re-upload, so N edits to the same
    // component within a frame collapse into a single upload at flush time.
    bool m_dirty = false;
    EditorContext *m_editorContext = nullptr;

protected:
    const SceneObject &getOwningObject() const { return *m_owningObject; }
    // Null until SceneObject::addComponent adopts this component. Components that may legitimately
    // live detached from the hierarchy (see TransformComponent::worldMatrix) ask through this.
    const SceneObject *findOwningObject() const { return m_owningObject; }
    void               markDirty() { m_dirty = true; }
    EditorContext     *editorContext() const { return m_editorContext; }

public:
    Component(std::string name = "") : m_name(std::move(name)) {}
    virtual ~Component() = default;

    void onGUI(EditorContext &context)
    {
        m_editorContext = &context;
        ImGui::Text("Component: %s", m_name.c_str());
        onGUIImpl();
        m_editorContext = nullptr;
    }

    virtual void onGUIImpl() {}

    // Appends editor-only geometry when this component's owning object is selected.
    // Components without a selection visualization keep the default no-op.
    virtual void onSelectGizmo(SelectionGizmoContext &) const {}

    // Runs once a whole scene has finished loading, when every object, every component and every
    // parent link in it exists. This is the only place during a load where a component may reach
    // outside itself.
    //
    // A component's deserialization must restore that component's own state and nothing else — it
    // must not call SceneObject::getComponent or hasComponent, because the components of an object
    // are created in manifest order and the sibling being asked for may not exist yet. Anything
    // that needs a sibling, or another object, goes here instead. See SceneSerializer for the full
    // rule, and SphericalCameraController for the worked example.
    //
    // Not called for components added at runtime: addComponent takes the component's full state,
    // and whatever it needs to reach is already in place by then.
    virtual void onLoaded() {}

    bool isDirty() const { return m_dirty; }
    void clearDirty() { m_dirty = false; }
};

} // namespace lr
