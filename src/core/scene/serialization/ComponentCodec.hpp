#pragma once

#include <nlohmann/json_fwd.hpp>

#include <memory>
#include <string_view>
#include <typeindex>
#include <vector>


namespace lr
{

class MaterialStore;
class MeshStore;
class Scene;
class SceneObject;

struct ComponentSaveContext
{
    const MeshStore &meshes;
    const MaterialStore &materials;
};

struct ComponentLoadContext
{
    Scene &scene;
    MeshStore &meshes;
    MaterialStore &materials;
};

class ComponentCodec
{
public:
    virtual ~ComponentCodec() = default;

    virtual std::string_view key() const = 0;
    virtual std::type_index componentType() const = 0;
    virtual nlohmann::json encode(const SceneObject &object,
                                  const ComponentSaveContext &context) const = 0;
    virtual void decode(const nlohmann::json &value, SceneObject &object,
                        ComponentLoadContext &context, const std::string &where) const = 0;
};

class ComponentCodecRegistry
{
public:
    void add(std::unique_ptr<ComponentCodec> codec);
    const ComponentCodec &forType(std::type_index type) const;
    const ComponentCodec &forKey(std::string_view key) const;
    const ComponentCodec *findKey(std::string_view key) const;

private:
    std::vector<std::unique_ptr<ComponentCodec>> m_codecs;
};

ComponentCodecRegistry makeSceneComponentCodecs();

// Each domain contributes its codecs explicitly. This keeps registration deterministic and makes
// the link dependency from scene serialization to optional feature libraries visible in CMake.
void addCoreComponentCodecs(ComponentCodecRegistry &registry);
void addSkinComponentCodecs(ComponentCodecRegistry &registry);
void addRigidBodyComponentCodecs(ComponentCodecRegistry &registry);

} // namespace lr
