#include "core/loaders/ObjExporter.hpp"

#include <fstream>
#include <ostream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace lr
{
namespace
{

// Reads a per-vertex attribute only if it is registered with the expected type. A Mesh's
// getPerVertexArray throws on both a missing name and a type mismatch; here either just means
// "the OBJ omits this attribute", so the layout is consulted first and the span left empty.
template <typename T> std::span<const T> optionalPerVertexArray(const Mesh &mesh, const std::string &name)
{
    if (name.empty())
    {
        return {};
    }
    const MeshLayout::AttributeDesc *desc = mesh.layout().findPerVertexAttr(name);
    if (!desc || desc->type != std::type_index(typeid(T)))
    {
        return {};
    }
    return mesh.getPerVertexArray<T>(name);
}

// OBJ indices are 1-based.
uint32_t objIndex(uint32_t zeroBased) { return zeroBased + 1; }

void writeFaceCorner(std::ostream &out, uint32_t positionIndex, uint32_t cornerIndex, bool hasUvs, bool hasNormals)
{
    out << ' ' << objIndex(positionIndex);
    if (hasUvs)
    {
        out << '/' << objIndex(cornerIndex);
    } else if (hasNormals)
    {
        // "v//vn" — the empty texcoord field is what lets a normal follow without a uv.
        out << '/';
    }
    if (hasNormals)
    {
        out << '/' << objIndex(cornerIndex);
    }
}

} // namespace

void ObjExporter::write(const Mesh &mesh, std::ostream &out, const ObjExporterConfig &config)
{
    const auto &positions       = mesh.positions();
    const auto &positionIndices = mesh.positionIndices();
    const auto &faces           = mesh.faces();
    const auto &faceGroups      = mesh.faceGroups();

    if (faces.empty())
    {
        throw std::invalid_argument("ObjExporter: mesh has no faces");
    }
    if (!faceGroups.empty() && faceGroups.size() != faces.size())
    {
        throw std::invalid_argument("ObjExporter: faceGroups size does not match face count");
    }

    const std::span<const glm::vec3> normals = optionalPerVertexArray<glm::vec3>(mesh, config.normalAttributeName);
    const std::span<const glm::vec2> uvs     = optionalPerVertexArray<glm::vec2>(mesh, config.uvAttributeName);

    // Both attribute domains are sized by vertexCount(), so one vt/vn line per render vertex
    // keeps a face corner's position index and its attribute index independent of each other.
    const bool hasNormals = normals.size() == positionIndices.size();
    const bool hasUvs     = uvs.size() == positionIndices.size();

    const std::streamsize previousPrecision = out.precision(config.precision);

    if (!config.objectName.empty())
    {
        out << "o " << config.objectName << '\n';
    }

    for (const glm::vec3 &position : positions)
    {
        out << "v " << position.x << ' ' << position.y << ' ' << position.z << '\n';
    }
    if (hasUvs)
    {
        for (const glm::vec2 &uv : uvs)
        {
            out << "vt " << uv.x << ' ' << uv.y << '\n';
        }
    }
    if (hasNormals)
    {
        for (const glm::vec3 &normal : normals)
        {
            out << "vn " << normal.x << ' ' << normal.y << ' ' << normal.z << '\n';
        }
    }

    // Faces are emitted in their existing order and a usemtl is written whenever the group
    // changes, rather than sorting faces by group. Reordering would renumber faces and so
    // invalidate any per-face data a caller holds alongside the exported file.
    std::unordered_map<uint32_t, std::string> groupNames;
    auto                                      groupName = [&](uint32_t group) -> const std::string                                      &{
        auto [it, inserted] = groupNames.try_emplace(group);
        if (inserted)
        {
            if (config.materialStore && group < config.materialStore->capacity())
            {
                it->second = config.materialStore->get(group).name;
            }
            // Positional fallback, also used for a real material whose name is empty: a usemtl
            // with no argument is not valid OBJ, and a stable name keeps groups distinguishable.
            if (it->second.empty())
            {
                it->second = "group_" + std::to_string(group);
            }
        }
        return it->second;
    };

    bool     haveGroup   = false;
    uint32_t activeGroup = 0;

    for (size_t faceIdx = 0; faceIdx < faces.size(); ++faceIdx)
    {
        if (!faceGroups.empty())
        {
            const uint32_t group = faceGroups[faceIdx];
            if (!haveGroup || group != activeGroup)
            {
                out << "usemtl " << groupName(group) << '\n';
                activeGroup = group;
                haveGroup   = true;
            }
        }

        const glm::uvec3 &face = faces[faceIdx];
        out << 'f';
        for (int corner = 0; corner < 3; ++corner)
        {
            const uint32_t cornerIndex = face[corner];
            if (cornerIndex >= positionIndices.size())
            {
                throw std::invalid_argument("ObjExporter: face references a vertex outside the mesh");
            }
            writeFaceCorner(out, positionIndices[cornerIndex], cornerIndex, hasUvs, hasNormals);
        }
        out << '\n';
    }

    out.precision(previousPrecision);
}

void ObjExporter::write(const Mesh &mesh, const std::filesystem::path &path, const ObjExporterConfig &config)
{
    if (path.empty())
    {
        throw std::invalid_argument("ObjExporter: empty path");
    }

    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file)
    {
        throw std::runtime_error("ObjExporter: failed to open '" + path.string() + "' for writing");
    }

    write(mesh, file, config);

    file.flush();
    if (!file)
    {
        throw std::runtime_error("ObjExporter: failed to write '" + path.string() + "'");
    }
}

} // namespace lr
