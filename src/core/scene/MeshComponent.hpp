#pragma once

#include "core/scene/Component.hpp"

#include "core/scene/MeshStore.hpp"
#include "core/loaders/MaterialStore.hpp"

#include <vector>

namespace lr
{

class MeshComponent : public Component
{
private:
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

    void onGUIImpl() override;

    const std::vector<MaterialHandle> &materialHandles() const;
    MeshHandle                         meshHandle() const { return m_meshHandle; }
    Mesh                              &mesh();
    const Mesh                        &mesh() const;
    const MeshLayout                  &layout() const;
};

} // namespace lr
