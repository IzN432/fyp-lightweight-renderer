#pragma once

#include <atomic>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace lr
{

inline constexpr uint32_t invalidHandleIndex = std::numeric_limits<uint32_t>::max();

struct ImageHandle
{
    uint32_t index = invalidHandleIndex;
    uint64_t owner = 0;

    explicit    operator bool() const { return index != invalidHandleIndex; }
    friend bool operator==(ImageHandle, ImageHandle) = default;
};

struct BufferHandle
{
    uint32_t index = invalidHandleIndex;
    uint64_t owner = 0;

    explicit    operator bool() const { return index != invalidHandleIndex; }
    friend bool operator==(BufferHandle, BufferHandle) = default;
};

struct PassHandle
{
    uint32_t index = invalidHandleIndex;
    uint64_t owner = 0;

    explicit    operator bool() const { return index != invalidHandleIndex; }
    friend bool operator==(PassHandle, PassHandle) = default;
};

// Retrieves or adds frontend resource names while keeping image and buffer identity
// distinct. Names remain available for the current ResourceRegistry backend and
// for diagnostics, but pass declarations can use type-safe handles.
class ResourceHandleRegistry
{
public:
    ResourceHandleRegistry() : m_owner(nextOwner()) {}

    ResourceHandleRegistry(const ResourceHandleRegistry &)            = delete;
    ResourceHandleRegistry &operator=(const ResourceHandleRegistry &) = delete;

    uint64_t owner() const { return m_owner; }

    ImageHandle image(std::string_view name)
    {
        return ImageHandle{fetchOrAdd(name, m_imageNames, m_imageHandles), m_owner};
    }

    BufferHandle buffer(std::string_view name)
    {
        return BufferHandle{fetchOrAdd(name, m_bufferNames, m_bufferHandles), m_owner};
    }

    const std::string &name(ImageHandle handle) const
    {
        return checkedName(handle.index, handle.owner, m_imageNames, "image");
    }

    const std::string &name(BufferHandle handle) const
    {
        return checkedName(handle.index, handle.owner, m_bufferNames, "buffer");
    }

private:
    static uint64_t nextOwner()
    {
        static uint64_t nextId = 1;
        return nextId++;
    }

    static uint32_t fetchOrAdd(std::string_view name, std::vector<std::string> &names,
                               std::unordered_map<std::string, uint32_t> &handles)
    {
        if (name.empty())
        {
            throw std::invalid_argument("FrameGraph: resource name cannot be empty");
        }

        const auto existing = handles.find(std::string(name));
        if (existing != handles.end())
        {
            return existing->second;
        }

        const uint32_t index = static_cast<uint32_t>(names.size());
        names.emplace_back(name);
        handles.emplace(names.back(), index);
        return index;
    }

    const std::string &checkedName(uint32_t index, uint64_t owner, const std::vector<std::string> &names,
                                   std::string_view kind) const
    {
        if (owner != m_owner)
        {
            throw std::invalid_argument("FrameGraph: " + std::string(kind) + " handle belongs to another graph");
        }
        if (index >= names.size())
        {
            throw std::out_of_range("FrameGraph: invalid " + std::string(kind) + " handle");
        }
        return names[index];
    }

    uint64_t                                  m_owner;
    std::vector<std::string>                  m_imageNames;
    std::vector<std::string>                  m_bufferNames;
    std::unordered_map<std::string, uint32_t> m_imageHandles;
    std::unordered_map<std::string, uint32_t> m_bufferHandles;
};

} // namespace lr
