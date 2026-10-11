#include "core/loaders/ObjExporter.hpp"

#include <fstream>
#include <ostream>
#include <stdexcept>

namespace lr
{
namespace
{

// Writes one object's `v` and `f` lines, shifting face indices by `vertexOffset` -- the number of
// vertices already written by earlier objects -- so several objects share one index space in the
// file. Returns the vertex count this object contributed, which the caller adds to the offset.
// A mesh with no faces writes nothing and contributes nothing.
uint32_t writeObject(const Mesh &mesh, const glm::mat4 &transform, const std::string &name, std::ostream &out,
                     uint32_t vertexOffset)
{
    const auto &positions       = mesh.positions();
    const auto &positionIndices = mesh.positionIndices();
    const auto &faces           = mesh.faces();

    if (faces.empty())
    {
        return 0;
    }

    if (!name.empty())
    {
        out << "o " << name << '\n';
    }

    for (const glm::vec3 &position : positions)
    {
        // Baked rather than written as-is: OBJ carries no per-object transform, so an object's
        // placement in the scene can only survive in the coordinates themselves.
        const glm::vec3 world = glm::vec3(transform * glm::vec4(position, 1.0f));
        out << "v " << world.x << ' ' << world.y << ' ' << world.z << '\n';
    }

    for (const glm::uvec3 &face : faces)
    {
        out << 'f';
        for (int corner = 0; corner < 3; ++corner)
        {
            // A face corner is a render vertex, so it resolves to a position through
            // positionIndices even here, where no per-corner attributes are written.
            const uint32_t cornerIndex = face[corner];
            if (cornerIndex >= positionIndices.size())
            {
                throw std::invalid_argument("ObjExporter: face references a vertex outside the mesh");
            }
            // OBJ indices are 1-based, and counted from the start of the file across all objects.
            out << ' ' << vertexOffset + positionIndices[cornerIndex] + 1;
        }
        out << '\n';
    }

    return static_cast<uint32_t>(positions.size());
}

// Guards the stream's precision so a throw mid-write cannot leave the caller's stream reconfigured.
class PrecisionGuard
{
public:
    PrecisionGuard(std::ostream &out, int precision) : m_out(out), m_previous(out.precision(precision)) {}
    ~PrecisionGuard() { m_out.precision(m_previous); }

    PrecisionGuard(const PrecisionGuard &)            = delete;
    PrecisionGuard &operator=(const PrecisionGuard &) = delete;

private:
    std::ostream   &m_out;
    std::streamsize m_previous;
};

// Shared by both path overloads: opening, delegating, and turning stream failure into an exception.
template <typename WriteBody> void writeToPath(const std::filesystem::path &path, WriteBody writeBody)
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

    writeBody(file);

    file.flush();
    if (!file)
    {
        throw std::runtime_error("ObjExporter: failed to write '" + path.string() + "'");
    }
}

} // namespace

void ObjExporter::write(const Mesh &mesh, std::ostream &out, const ObjExporterConfig &config)
{
    if (mesh.faces().empty())
    {
        throw std::invalid_argument("ObjExporter: mesh has no faces");
    }

    const PrecisionGuard precision(out, config.precision);
    writeObject(mesh, glm::mat4(1.0f), config.objectName, out, 0);
}

void ObjExporter::write(const Mesh &mesh, const std::filesystem::path &path, const ObjExporterConfig &config)
{
    writeToPath(path, [&](std::ostream &out) {
        write(mesh, out, config);
    });
}

void ObjExporter::write(std::span<const ObjExportObject> objects, std::ostream &out, const ObjExporterConfig &config)
{
    if (objects.empty())
    {
        throw std::invalid_argument("ObjExporter: no objects to export");
    }

    const PrecisionGuard precision(out, config.precision);

    uint32_t vertexOffset = 0;
    for (const ObjExportObject &object : objects)
    {
        if (!object.mesh)
        {
            throw std::invalid_argument("ObjExporter: null mesh in export object list");
        }
        vertexOffset += writeObject(*object.mesh, object.transform, object.name, out, vertexOffset);
    }

    // Every object was empty, so the file would have no geometry at all. Reported like the
    // single-mesh no-faces case rather than written out, since an empty OBJ is not a useful export.
    if (vertexOffset == 0)
    {
        throw std::invalid_argument("ObjExporter: none of the objects have any faces");
    }
}

void ObjExporter::write(std::span<const ObjExportObject> objects, const std::filesystem::path &path,
                        const ObjExporterConfig &config)
{
    writeToPath(path, [&](std::ostream &out) {
        write(objects, out, config);
    });
}

} // namespace lr
