#pragma once

#include "core/scene/SceneObject.hpp"

#include <functional>
#include <string>
#include <typeindex>
#include <utility>
#include <vector>

namespace lr
{

// The component types the Inspector offers to add, in the order it lists them.
//
// Composed explicitly by the editor rather than discovered from the component types that exist:
// which components are worth offering, and what a blank one of each should look like, is an editor
// decision. It follows ComponentCodec's registration approach for the same reasons — the list stays
// deterministic, and each feature library the editor reaches into stays visible at the call site.
//
// A component with no sensible blank state is simply left out. MeshComponent is the clearest case:
// it needs a mesh, which means importing or picking one, so geometry arrives through an import or
// through pasting a mesh component instead.
class ComponentCatalog
{
public:
    struct Entry
    {
        std::string                        name;
        std::type_index                    type;
        std::function<void(SceneObject &)> add;
    };

    // Adds `T` as its default constructor leaves it.
    template <typename T> void add(std::string name)
    {
        addEntry(std::move(name), std::type_index(typeid(T)),
                 [](SceneObject &object) { object.addComponent<T>(); });
    }

    // For a component whose blank state has to be chosen — a light has to be some kind of light.
    template <typename T> void add(std::string name, std::function<void(SceneObject &)> factory)
    {
        addEntry(std::move(name), std::type_index(typeid(T)), std::move(factory));
    }

    const std::vector<Entry> &entries() const { return m_entries; }

private:
    void addEntry(std::string name, std::type_index type, std::function<void(SceneObject &)> factory)
    {
        m_entries.push_back({std::move(name), type, std::move(factory)});
    }

    std::vector<Entry> m_entries;
};

} // namespace lr
