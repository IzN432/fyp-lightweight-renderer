#pragma once

#include "Material.hpp"

#include <functional>
#include <vector>

namespace lr
{

using MaterialHandle = uint32_t;

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

    MaterialHandle acquire(Material material);
    void           release(MaterialHandle handle);

    Material       &get(MaterialHandle handle);
    const Material &get(MaterialHandle handle) const;

    // A single shared "no material assigned" slot, reserved at construction and never handed out
    // by acquire() — loaders use this for primitives/faces that don't reference a real material,
    // instead of building and registering their own throwaway default on every load() call.
    MaterialHandle defaultMaterialHandle() const { return m_defaultMaterialHandle; }

    uint32_t capacity() const { return static_cast<uint32_t>(m_materials.size()); }

    // Capacity-length, in handle order — feeds MaterialUploader::upload()/update() directly.
    std::vector<const Material *> snapshot() const;

private:
    std::vector<Material>       m_materials;
    std::vector<MaterialHandle> m_freeList;
    std::function<Material()>   m_defaultMaterialFactory;
    MaterialHandle              m_defaultMaterialHandle = 0;
};

} // namespace lr
