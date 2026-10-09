#pragma once

#include "core/scene/ComponentClipboard.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <typeindex>
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
    //
    // A bitmask rather than a flag, so a component whose parts need different work can say which
    // part moved — see MeshComponent::Aspect, where a material edit costs a buffer update and a
    // mesh swap costs a full geometry re-pack. The bits mean whatever the component that defines
    // them says; a consumer asks with the enum of the concrete type it is holding.
    uint32_t m_dirtyAspects = 0;
    EditorContext *m_editorContext = nullptr;

protected:
    const SceneObject &getOwningObject() const { return *m_owningObject; }
    // Null until SceneObject::addComponent adopts this component. Components that may legitimately
    // live detached from the hierarchy (see TransformComponent::worldMatrix) ask through this.
    const SceneObject *findOwningObject() const { return m_owningObject; }
    // Flags every aspect, which is what a component that does not divide itself into aspects wants.
    void markDirty(uint32_t aspects = allAspects) { m_dirtyAspects |= aspects; }
    EditorContext     *editorContext() const { return m_editorContext; }

public:
    // Passed to markDirty() by a component that does not divide itself into aspects, and the set
    // isDirty()/clearDirty() cover when asked without an argument.
    static constexpr uint32_t allAspects = ~0u;

    Component(std::string name = "") : m_name(std::move(name)) {}
    virtual ~Component() = default;

    // `removalRequest` is how this component's menu asks to be deleted: it is set rather than acted
    // on, because the caller is iterating its object's components and a component cannot erase
    // itself from inside its own method. SceneObject::onGUI performs it after the loop.
    void onGUI(EditorContext &context, std::optional<std::type_index> &removalRequest);

    virtual void onGUIImpl() {}

    // Editor copy/paste of this component's values, offered from a right-click menu over the
    // component's own block in the inspector. A component opts in by overriding both halves:
    // copyValues takes a detached snapshot of whatever its inspector edits, pasteValues applies one
    // back and marks itself dirty.
    //
    // The default copyValues returns null, which keeps the component out of the menu altogether —
    // a component joins only once pasting it is actually meaningful, so there is no menu entry for
    // components whose values belong to one object alone. The clipboard only ever hands pasteValues
    // a snapshot produced by this same component type, so an override may cast straight to its own
    // payload type.
    virtual std::unique_ptr<ComponentValues> copyValues() const { return nullptr; }
    virtual void                             pasteValues(const ComponentValues &values) { (void)values; }

    // The snapshot the inspector records for undo, and the way it is put back. These default to the
    // clipboard pair above, which for most components is exactly right: what can be copied off a
    // component is what its inspector edits.
    //
    // A component overrides them where the two differ. MeshComponent copies its mesh but edits its
    // materials, which live in the shared store; RigidBodyComponent deliberately leaves its force
    // accumulators out of a copy, yet its inspector drives them; SkinComponent has a checkbox but no
    // business being pasted onto another object at all. Returning null (the default for a component
    // that implements neither) keeps the component out of the undo history, the same way returning
    // null from copyValues keeps it out of the clipboard menu.
    virtual std::unique_ptr<ComponentValues> undoValues() const { return copyValues(); }
    virtual void restoreUndoValues(const ComponentValues &values) { pasteValues(values); }

    // How to put a copy of this component's values onto an object that has no component of this
    // type yet, for the inspector's "Paste component". Null (the default) keeps a type out of that
    // menu entry while leaving its copy/paste onto an existing component alone — for a component
    // that only makes sense where something else already put it.
    virtual ComponentValuesAdder valuesAdder() const { return nullptr; }

    // Whether the Inspector may delete this component off its object. False by default, so a
    // component takes part only once removing it is known to leave the rest of the engine in a
    // state it can handle — the component types nothing else depends on, in practice.
    virtual bool allowsRemoval() const { return false; }

    // Whether this component can take the clipboard's values as things stand. Type agreement is
    // already settled by ComponentClipboard; this is for a component whose answer depends on its own
    // situation rather than its type — MeshComponent refuses while a sibling SkinComponent is
    // matched to the mesh it would replace. A false answer disables the paste entry without
    // disabling the copy one.
    virtual bool acceptsPastedValues() const { return true; }

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

    // Whether anything is dirty, and whether any of `aspects` is.
    bool isDirty() const { return m_dirtyAspects != 0; }
    bool isDirty(uint32_t aspects) const { return (m_dirtyAspects & aspects) != 0; }

    void clearDirty() { m_dirtyAspects = 0; }
    void clearDirty(uint32_t aspects) { m_dirtyAspects &= ~aspects; }

private:
    void drawComponentMenu(EditorContext &context, std::optional<std::type_index> &removalRequest);

    // Turns whatever this component's inspector did this frame into at most one undo command. See
    // Component.cpp for how a drag spanning many frames still becomes a single one.
    void recordValueEdit(EditorContext &context, std::unique_ptr<ComponentValues> beforeThisFrame,
                         bool beingRemoved);

    // The values this component held when the interaction currently in progress started. Null while
    // nothing is being edited, which is also what it is reset to once an edit has been recorded.
    std::unique_ptr<ComponentValues> m_valuesBeforeEdit;
};

} // namespace lr
