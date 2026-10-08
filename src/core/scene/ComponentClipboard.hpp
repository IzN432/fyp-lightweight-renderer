#pragma once

#include <memory>
#include <optional>
#include <typeindex>
#include <utility>

namespace lr
{

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

// Holds at most one copied snapshot, tagged with the component type it came from. Owned by
// EditorContext, so it outlives both the inspector GUI that filled it and the object copied from.
class ComponentClipboard
{
public:
    void store(std::type_index componentType, std::unique_ptr<ComponentValues> values)
    {
        m_componentType = componentType;
        m_values        = std::move(values);
    }

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
};

} // namespace lr
