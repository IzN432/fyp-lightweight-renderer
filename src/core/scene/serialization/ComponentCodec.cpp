#include "ComponentCodec.hpp"

#include <stdexcept>
#include <string>

namespace lr
{

void ComponentCodecRegistry::add(std::unique_ptr<ComponentCodec> codec)
{
    for (const auto &existing : m_codecs)
    {
        if (existing->key() == codec->key())
            throw std::runtime_error("Duplicate scene component codec key '" + std::string(codec->key()) + "'");
        if (existing->componentType() == codec->componentType())
            throw std::runtime_error("Duplicate scene component codec type '" +
                                     std::string(codec->componentType().name()) + "'");
    }
    m_codecs.push_back(std::move(codec));
}

const ComponentCodec &ComponentCodecRegistry::forType(std::type_index type) const
{
    for (const auto &codec : m_codecs)
        if (codec->componentType() == type) return *codec;
    throw std::runtime_error("SceneSerializer: unsupported component '" + std::string(type.name()) +
                             "' in format version 2");
}

const ComponentCodec &ComponentCodecRegistry::forKey(std::string_view key) const
{
    if (const ComponentCodec *codec = findKey(key)) return *codec;
    throw std::runtime_error("SceneSerializer: unknown component '" + std::string(key) + "'");
}

const ComponentCodec *ComponentCodecRegistry::findKey(std::string_view key) const
{
    for (const auto &codec : m_codecs)
        if (codec->key() == key) return codec.get();
    return nullptr;
}

ComponentCodecRegistry makeSceneComponentCodecs()
{
    ComponentCodecRegistry registry;
    addCoreComponentCodecs(registry);
    addSkinComponentCodecs(registry);
    addRigidBodyComponentCodecs(registry);
    return registry;
}

} // namespace lr
