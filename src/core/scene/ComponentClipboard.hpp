#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <typeindex>
#include <utility>

namespace lr
{

class SceneObject;

// A detached copy of one component's editable values, taken for the inspector's copy/paste.
//
// The concrete payload type stays private to the component type that produced it: only that same
// type ever reads one back, because ComponentClipboard refuses a mismatched paste. That keeps
// copy/paste out of the serialization schema in ComponentCodec, which has a different job — a
// codec has to survive being written to disk and read back by a later build, while a snapshot only
// has to survive until the next paste in this session.
//
// Because a snapshot is short-lived, it keeps shared data as the handle the component already
// stores rather than duplicating it, so pasting re-points the target at the same mesh, material or
// skeleton instead of cloning one.
class ComponentValues
{
public:
    virtual ~ComponentValues() = default;
};

// Payload for the common case, which is most of them: a component whose editable state is one
// copyable value — a Transform, a light variant, a small struct of the fields its inspector drives.
// Nothing keys off the payload type, because the clipboard already keys off the component type, so
// two components sharing a T never see each other's snapshots.
template <typename T> struct ComponentValueSnapshot final : public ComponentValues
{
    explicit ComponentValueSnapshot(T value) : value(std::move(value)) {}

    T value;
};

// Reads a snapshot back inside pasteValues. Sound by construction: ComponentClipboard only offers a
// paste to the component type that produced the snapshot, so the payload is the one that type wrote.
template <typename T> const T &componentValuesAs(const ComponentValues &values)
{
    return static_cast<const ComponentValueSnapshot<T> &>(values).value;
}

// Puts a component carrying copied values onto an object that has no component of that type yet,
// which is what the inspector's "Paste component" does. Bound at the moment the values are copied,
// so a component that cannot be built from its values alone — MeshComponent, which resolves its
// mesh through the stores — can capture what it needs while it still has it.
using ComponentValuesAdder = std::function<void(SceneObject &, const ComponentValues &)>;

// Holds at most one copied snapshot, tagged with the component type it came from. Owned by
// EditorContext, so it outlives both the inspector GUI that filled it and the object copied from.
class ComponentClipboard
{
public:
    void store(std::type_index componentType, std::unique_ptr<ComponentValues> values,
               ComponentValuesAdder adder = {})
    {
        m_componentType = componentType;
        m_values        = std::move(values);
        m_adder         = std::move(adder);
    }

    // The copied component's type, or nothing while the clipboard is empty.
    std::optional<std::type_index> componentType() const
    {
        return m_values ? m_componentType : std::nullopt;
    }

    const ComponentValues *values() const { return m_values.get(); }

    // Empty unless the copied component type can be added to an object that lacks one.
    const ComponentValuesAdder &adder() const { return m_adder; }

    // Null unless the clipboard holds a snapshot taken from this very component type. Pasting
    // across component types would mean inventing a translation between two unrelated sets of
    // values, so the inspector simply does not offer it.
    const ComponentValues *valuesFor(std::type_index componentType) const
    {
        if (!m_values || m_componentType != componentType)
        {
            return nullptr;
        }
        return m_values.get();
    }

    bool holdsValuesFor(std::type_index componentType) const
    {
        return valuesFor(componentType) != nullptr;
    }

private:
    std::optional<std::type_index>   m_componentType;
    std::unique_ptr<ComponentValues> m_values;
    ComponentValuesAdder             m_adder;
};

} // namespace lr
