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
    // Handles into the MaterialStore this mesh's faceGroups index into — not owned here, just
    // referenced so onGUIImpl can offer them up for editing.
    std::vector<MaterialHandle> m_materialHandles;
    MaterialStore              *m_materialStore;
    // Set for meshes that are an implementation detail of another component (e.g. a light's visual
    // quad, whose mesh/material are derived from that light and overwritten on every update) rather
    // than user-editable scene content — keeps them out of the Scene Hierarchy.
    bool m_hideFromGui;

public:
    explicit MeshComponent(MeshHandle meshHandle, MeshStore &meshStore,
                           std::vector<MaterialHandle> materialHandles, MaterialStore &materialStore,
                           bool hideFromGui = false);

    void onGUIImpl() override;

    const std::vector<MaterialHandle> &materialHandles() const;
    MeshHandle                         meshHandle() const { return m_meshHandle; }
    Mesh                              &mesh();
    const Mesh                        &mesh() const;
    const MeshLayout                  &layout() const;
};

} // namespace lr
