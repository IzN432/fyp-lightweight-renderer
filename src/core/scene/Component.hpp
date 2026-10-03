#pragma once

#include <string>
#include <imgui.h>

namespace lr
{

struct EditorContext;

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

    bool isDirty() const { return m_dirty; }
    void clearDirty() { m_dirty = false; }
};

} // namespace lr
