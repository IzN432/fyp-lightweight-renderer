#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace lr::framegraph
{

struct ResourceId
{
    uint32_t value = std::numeric_limits<uint32_t>::max();

    friend bool operator==(ResourceId, ResourceId) = default;
    explicit operator bool() const { return value != std::numeric_limits<uint32_t>::max(); }
};

struct PassId
{
    uint32_t value = std::numeric_limits<uint32_t>::max();

    friend bool operator==(PassId, PassId) = default;
    explicit operator bool() const { return value != std::numeric_limits<uint32_t>::max(); }
};

enum class ResourceKind
{
    Unknown,
    Image,
    Buffer,
};

enum class PassKind
{
    Graphics,
    Compute,
    External,
};

enum class AccessMode
{
    Read,
    Write,
    ReadWrite,
};

// Backend-independent intent. Descriptor slots, layouts, pipeline stages and
// native resource handles are frontend/backend concerns and deliberately do
// not appear in the logical graph.
enum class ResourceUsage
{
    Unknown,
    SampledImage,
    StorageImage,
    UniformBuffer,
    StorageBuffer,
    ColorOrDepthAttachment,
    VertexBuffer,
    IndexBuffer,
};

struct ResourceAccess
{
    ResourceId    resource;
    AccessMode    mode  = AccessMode::Read;
    ResourceUsage usage = ResourceUsage::Unknown;
};

struct ResourceNode
{
    ResourceId   id;
    std::string  name;
    ResourceKind kind = ResourceKind::Unknown;
};

struct PassNode
{
    PassId                      id;
    std::string                 name;
    PassKind                    kind = PassKind::Graphics;
    std::vector<ResourceAccess> accesses;
    std::vector<PassId>         explicitDependencies;
};

// This is the stable, backend-independent frame-graph input. It can be built
// directly by another frontend (including Python) or populated by the current
// C++ PassDesc adapter.
class GraphDefinition
{
public:
    PassId addPass(std::string name, PassKind kind);
    ResourceId addResource(std::string name, ResourceKind kind);
    void addAccess(PassId pass, ResourceAccess access);
    void addDependency(PassId pass, PassId dependency);

    const std::vector<ResourceNode> &resources() const { return m_resources; }
    const std::vector<PassNode> &passes() const { return m_passes; }

    const ResourceNode &resource(ResourceId id) const;
    const PassNode &pass(PassId id) const;

    ResourceId findResource(std::string_view name) const;
    PassId findPass(std::string_view name) const;

private:
    std::vector<ResourceNode> m_resources;
    std::vector<PassNode> m_passes;
    std::unordered_map<std::string, ResourceId> m_resourceIds;
    std::unordered_map<std::string, PassId> m_passIds;
};

} // namespace lr::framegraph
