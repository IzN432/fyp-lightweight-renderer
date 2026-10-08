#pragma once

#include "core/scene/Component.hpp"

#include "core/scene/MeshStore.hpp"
#include "core/loaders/MaterialStore.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace lr
{

class MeshComponent : public Component
{
private:
    // The distinct materials of the mesh this component references.
    void collectMaterialHandles();

    // Adds a MeshComponent for `values`' mesh to `object`, which must have none. A static member so
    // it can flag the new component dirty; bound into the adder with the stores it resolves through.
    static void addPasted(SceneObject &object, MeshStore &meshStore, MaterialStore &materialStore,
                          const ComponentValues &values);

    MeshHandle m_meshHandle;
    MeshStore *m_meshStore;
    // The distinct materials this mesh's faceGroups reference, in first-encounter order — derived
    // at construction, not owned here, just referenced so onGUIImpl can offer them up for editing.
    std::vector<MaterialHandle> m_materialHandles;
    MaterialStore              *m_materialStore;

public:
    // The material list is read off the mesh's faceGroups, which are the only authority on what the
    // mesh actually uses: loaders bake global MaterialHandles into them, and deserialization remaps
    // them to live handles before any component is built.
    explicit MeshComponent(MeshHandle meshHandle, MeshStore &meshStore, MaterialStore &materialStore);

    // Which part of this component a consumer has to react to: a material edit is a buffer update,
    // while a different mesh invalidates the gathered Mesh* list and every shared buffer packed
    // from it (see SceneGpu::flushDirty).
    enum Aspect : uint32_t
    {
        Materials = 1u << 0,
        Geometry  = 1u << 1,
    };

    // Carries the mesh as its MeshId, not its MeshHandle: a handle only addresses the store for
    // this session and MeshStore::clear() invalidates it, while the clipboard outlives a scene
    // load. The id is the identity built to survive one, and resolving it hands back the very same
    // mesh — pasting gives another object the same shared geometry, it never duplicates any.
    std::unique_ptr<ComponentValues> copyValues() const override;
    ComponentValuesAdder             valuesAdder() const override;

    // A mesh can be given to an object that has none, but never swapped on an object that already
    // has one. Replacing it would have to re-pack all shared geometry, rebind whatever the editor
    // has bound against the outgoing Mesh, and break the vertex-group-to-joint pairing any sibling
    // SkinComponent depends on — none of which an inspector paste should be quietly doing.
    bool acceptsPastedValues() const override { return false; }

    void onGUIImpl() override;

    const std::vector<MaterialHandle> &materialHandles() const;
    MeshHandle                         meshHandle() const { return m_meshHandle; }
    Mesh                              &mesh();
    const Mesh                        &mesh() const;
    const MeshLayout                  &layout() const;
};

} // namespace lr
