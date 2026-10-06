#include "MaterialStore.hpp"

#include <stdexcept>

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

    // Slot 0 is permanently reserved as the shared default material — excluded from the free list
    // so acquire() never hands it out.
    m_defaultMaterialHandle = 0;

    m_freeList.reserve(capacity - 1);
    for (uint32_t i = capacity; i-- > 1;)
    {
        m_freeList.push_back(i);
    }
}

MaterialHandle MaterialStore::acquire(Material material)
{
    if (m_freeList.empty())
    {
        throw std::runtime_error("MaterialStore: capacity exhausted");
    }

    MaterialHandle handle = m_freeList.back();
    m_freeList.pop_back();
    m_materials[handle] = std::move(material);
    return handle;
}

void MaterialStore::release(MaterialHandle handle)
{
    m_materials.at(handle) = m_defaultMaterialFactory();
    m_freeList.push_back(handle);
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
    for (Material &material : m_materials) material = m_defaultMaterialFactory();
    m_defaultMaterialHandle = 0;
    m_freeList.clear();
    for (uint32_t i = static_cast<uint32_t>(m_materials.size()); i-- > 1;)
    {
        m_freeList.push_back(i);
    }
}

} // namespace lr
