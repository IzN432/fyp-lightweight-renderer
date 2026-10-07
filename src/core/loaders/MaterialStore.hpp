#pragma once

#include "Material.hpp"

#include "core/utility/Uuid.hpp"

#include <functional>
#include <optional>
#include <unordered_map>
#include <vector>

namespace lr
{

using MaterialHandle = uint32_t;

// The identity a material keeps across a save and a load, as opposed to MaterialHandle, which is a
// GPU-facing slot index valid only for the lifetime of the session.
using MaterialId = Uuid;

// Fixed-capacity store of Materials, handing out stable handles that a mesh's faceGroups can
// index directly (see AreaLightVisual.hpp / geometry.frag, which reads faceGroup values straight
// into the materials SSBO and texture arrays with no extra indirection). Capacity is reserved up
// front rather than grown dynamically: growing it would mean resizing the GPU-side material
// buffer and texture arrays, which requires rebuilding the frame graph's descriptor sets (see
// FrameGraph::compile()/buildDescriptorSets()) — out of scope for now.
class MaterialStore
{
public:
    MaterialStore(uint32_t capacity, std::function<Material()> defaultMaterialFactory);

    // Parameters the material lacks are filled in from the default material.
    MaterialHandle acquire(Material material);
    // Adopts an identity a material already had, for loading a saved scene. The handle is still
    // whichever slot happens to be free — only the identity carries over. Throws if that identity
    // is nil or already present in this store.
    MaterialHandle acquire(Material material, MaterialId id);
    void           release(MaterialHandle handle);

    Material       &get(MaterialHandle handle);
    const Material &get(MaterialHandle handle) const;

    // Handles are slot indices that a mesh's faceGroups feed straight to the GPU, so they cannot
    // carry identity across a save. Each occupied slot also has a MaterialId, which can:
    // persistence names materials by MaterialId and resolves back to a handle through find().
    MaterialId                    idOf(MaterialHandle handle) const;
    std::optional<MaterialHandle> find(MaterialId id) const;

    // A single shared "no material assigned" slot, reserved at construction and never handed out
    // by acquire() — loaders use this for primitives/faces that don't reference a real material,
    // instead of building and registering their own throwaway default on every load() call.
    MaterialHandle defaultMaterialHandle() const { return m_defaultMaterialHandle; }

    uint32_t capacity() const { return static_cast<uint32_t>(m_materials.size()); }
    // Restores every slot to its default value and makes all non-default handles available again.
    void clear();

    // Capacity-length, in handle order — feeds MaterialUploader::upload()/update() directly.
    std::vector<const Material *> snapshot() const;

private:
    // Identity of the material in each slot, parallel to m_materials. A free slot's entry is nil.
    void assignId(MaterialHandle handle, MaterialId id);

    std::vector<Material>                          m_materials;
    std::vector<MaterialId>                        m_ids;
    std::unordered_map<MaterialId, MaterialHandle> m_handlesById;
    std::vector<MaterialHandle>                    m_freeList;
    std::function<Material()>                      m_defaultMaterialFactory;
    MaterialHandle                                 m_defaultMaterialHandle = 0;
};

} // namespace lr
