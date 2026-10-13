#include "features/fluid/import/BlenderUniImporter.hpp"

#include <zlib.h>

#include <algorithm>
#include <bit>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace lr
{
namespace
{

class Cursor
{
public:
    explicit Cursor(const std::vector<std::byte> &bytes) : m_bytes(bytes) {}

    template <typename T> T read()
    {
        static_assert(std::is_trivially_copyable_v<T>);
        if (m_offset + sizeof(T) > m_bytes.size())
        {
            throw std::runtime_error("truncated Blender UNI configuration");
        }
        T result;
        std::memcpy(&result, m_bytes.data() + m_offset, sizeof(T));
        m_offset += sizeof(T);
        return result;
    }

    glm::vec3   readVec3() { return {read<float>(), read<float>(), read<float>()}; }
    glm::ivec3  readIvec3() { return {read<int32_t>(), read<int32_t>(), read<int32_t>()}; }
    std::size_t remaining() const { return m_bytes.size() - m_offset; }

private:
    const std::vector<std::byte> &m_bytes;
    std::size_t                   m_offset = 0;
};

} // namespace

std::string BlenderFluidDomainConfig::versionString() const
{
    const auto end = std::find(version.begin(), version.end(), '\0');
    return std::string(version.begin(), end);
}

BlenderFluidDomainConfig BlenderUniImporter::importFile(const std::filesystem::path &path)
{
    if constexpr (std::endian::native != std::endian::little)
    {
        throw std::runtime_error("BlenderUniImporter: only little-endian hosts are supported");
    }

    gzFile input = gzopen(path.string().c_str(), "rb");
    if (!input)
    {
        throw std::runtime_error("BlenderUniImporter: cannot open '" + path.string() + "'");
    }
    std::vector<std::byte> bytes(204);
    const int              readBytes = gzread(input, bytes.data(), static_cast<unsigned int>(bytes.size()));
    std::byte              extra{};
    const int              extraBytes  = gzread(input, &extra, 1);
    const int              closeResult = gzclose(input);
    if (readBytes != static_cast<int>(bytes.size()) || extraBytes != 0 || closeResult != Z_OK)
    {
        throw std::runtime_error("BlenderUniImporter: expected a 204-byte configuration in '" + path.string() + "'");
    }

    Cursor                   cursor(bytes);
    BlenderFluidDomainConfig result;
    result.activeFields      = cursor.read<int32_t>();
    result.resolution        = cursor.readIvec3();
    result.cellSize          = cursor.read<float>();
    result.timeStep          = cursor.read<float>();
    result.boundsMin         = cursor.readVec3();
    result.boundsMax         = cursor.readVec3();
    result.previousBoundsMin = cursor.readVec3();
    result.shift             = cursor.readIvec3();
    result.objectShift       = cursor.readVec3();
    // Blender stores a row-vector matrix row-major, with translation in row 3. GLM indexes
    // [column][row] and uses column vectors. Keeping the two file indices in this assignment
    // performs the semantic transpose and places that translation in GLM column 3.
    for (int blenderRow = 0; blenderRow < 4; ++blenderRow)
    {
        for (int blenderColumn = 0; blenderColumn < 4; ++blenderColumn)
        {
            result.objectTransform[blenderRow][blenderColumn] = cursor.read<float>();
        }
    }
    result.baseResolution  = cursor.readIvec3();
    result.resolutionMin   = cursor.readIvec3();
    result.resolutionMax   = cursor.readIvec3();
    result.activeColor     = cursor.readVec3();
    const int32_t timeBits = cursor.read<int32_t>();
    std::memcpy(&result.totalTime, &timeBits, sizeof(result.totalTime));
    for (char &byte : result.version)
    {
        byte = cursor.read<char>();
    }
    if (cursor.remaining() != 0)
    {
        throw std::runtime_error("BlenderUniImporter: unexpected trailing data in '" + path.string() + "'");
    }
    if (result.versionString().empty())
    {
        throw std::runtime_error("BlenderUniImporter: missing cache version in '" + path.string() + "'");
    }
    return result;
}

} // namespace lr
