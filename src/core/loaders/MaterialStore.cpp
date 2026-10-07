#include "MaterialStore.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace lr
{

MaterialStore::MaterialStore(uint32_t capacity, std::function<Material()> defaultMaterialFactory)
    : m_defaultMaterialFactory(std::move(defaultMaterialFactory))
{
    if (capacity == 0)
    {
        throw std::runtime_error("MaterialStore: capacity must be greater than zero");
    }

    m_materials.reserve(capacity);
    for (uint32_t i = 0; i < capacity; ++i)
    {
        m_materials.push_back(m_defaultMaterialFactory());
    }
    m_ids.resize(capacity);

    // Slot 0 is permanently reserved as the shared default material — excluded from the free list
    // so acquire() never hands it out. It still gets an identity, because a mesh may reference it
    // and that reference has to survive being saved.
    m_defaultMaterialHandle = 0;
    assignId(m_defaultMaterialHandle, generateUuid());

    m_freeList.reserve(capacity - 1);
    for (uint32_t i = capacity; i-- > 1;)
    {
        m_freeList.push_back(i);
    }
}

MaterialHandle MaterialStore::acquire(Material material)
{
    return acquire(std::move(material), generateUuid());
}

MaterialHandle MaterialStore::acquire(Material material, MaterialId id)
{
    if (id.is_nil())
    {
        throw std::invalid_argument("A material cannot be stored under a nil ID");
    }
    if (m_handlesById.contains(id))
    {
        throw std::invalid_argument("A material with ID " + toString(id) + " already exists");
    }
    if (m_freeList.empty())
    {
        throw std::runtime_error("MaterialStore: capacity exhausted");
    }

    // Producers that predate a parameter (OBJ files, light visuals, scenes saved before it existed)
    // don't write it; the default material's value fills the gap, so the GPU upload finds every scalar.
    for (auto &[name, value] : m_defaultMaterialFactory().parameters)
    {
        material.parameters.try_emplace(name, std::move(value));
    }

    MaterialHandle handle = m_freeList.back();
    m_freeList.pop_back();
    m_materials[handle] = std::move(material);
    assignId(handle, id);
    return handle;
}

void MaterialStore::release(MaterialHandle handle)
{
    m_materials.at(handle) = m_defaultMaterialFactory();
    assignId(handle, MaterialId{});
    m_freeList.push_back(handle);
}

void MaterialStore::assignId(MaterialHandle handle, MaterialId id)
{
    const MaterialId previous = m_ids.at(handle);
    if (!previous.is_nil())
    {
        m_handlesById.erase(previous);
    }
    m_ids[handle] = id;
    if (!id.is_nil())
    {
        m_handlesById.emplace(id, handle);
    }
}

MaterialId MaterialStore::idOf(MaterialHandle handle) const { return m_ids.at(handle); }

std::optional<MaterialHandle> MaterialStore::find(MaterialId id) const
{
    const auto found = m_handlesById.find(id);
    if (found == m_handlesById.end())
    {
        return std::nullopt;
    }
    return found->second;
}

Material       &MaterialStore::get(MaterialHandle handle) { return m_materials.at(handle); }
const Material &MaterialStore::get(MaterialHandle handle) const { return m_materials.at(handle); }

std::vector<const Material *> MaterialStore::snapshot() const
{
    std::vector<const Material *> result;
    result.reserve(m_materials.size());
    for (const auto &material : m_materials)
    {
        result.push_back(&material);
    }
    return result;
}

void MaterialStore::clear()
{
    for (Material &material : m_materials)
    {
        material = m_defaultMaterialFactory();
    }
    std::ranges::fill(m_ids, MaterialId{});
    m_handlesById.clear();
    m_defaultMaterialHandle = 0;
    assignId(m_defaultMaterialHandle, generateUuid());
    m_freeList.clear();
    for (uint32_t i = static_cast<uint32_t>(m_materials.size()); i-- > 1;)
    {
        m_freeList.push_back(i);
    }
}

} // namespace lr
