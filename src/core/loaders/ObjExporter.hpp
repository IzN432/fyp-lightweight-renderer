#pragma once

#include "core/scene/Mesh.hpp"

#include <glm/mat4x4.hpp>

#include <filesystem>
#include <iosfwd>
#include <span>
#include <string>

namespace lr
{
struct ObjExporterConfig
{
    // Decimal digits written per float. OBJ is a text format, so this trades file size against
    // precision; 9 is max_digits10 for IEEE-754 binary32, the point at which a written float
    // reads back bit-exact.
    int precision = 9;

    // Emitted as a single `o` statement before the face list. Empty omits it. Only consulted by the
    // single-mesh overloads -- a multi-object export names each object from its own entry.
    std::string objectName;
};

// One object in a multi-object export: a mesh, where it sits in the world, and what to call it.
struct ObjExportObject
{
    // Borrowed for the duration of the write call only.
    const Mesh *mesh = nullptr;

    // Applied to every position as it is written, so the objects keep their relative placement in
    // the file. Pass SceneObject::worldMatrix() to export the scene as it is being rendered, or
    // identity to write each mesh in its own object space.
    glm::mat4 transform = glm::mat4(1.0f);

    // Emitted as this object's `o` statement. Empty omits it.
    std::string name;
};

class ObjExporter
{
public:
    // Writes `mesh` as Wavefront OBJ, geometry only: one `v` line per Mesh::positions() entry and
    // one `f` line per face, whose indices address that unique-position domain. Normals, uvs and
    // materials are deliberately not written -- a face corner's own attributes live in the
    // render-vertex domain, and dropping them is what keeps `f` a plain list of position indices.
    //
    // Throws std::invalid_argument if the mesh has no faces, and std::runtime_error if the file
    // cannot be opened or written.
    static void write(const Mesh &mesh, const std::filesystem::path &path, const ObjExporterConfig &config = {});

    // Same, to an already-open stream. The caller owns flushing and error checking; use the path
    // overload to have write failures turned into exceptions.
    static void write(const Mesh &mesh, std::ostream &out, const ObjExporterConfig &config = {});

    // Writes several objects into one OBJ file, each preceded by its own `o` statement and with its
    // positions baked through its `transform` -- so a whole scene exports as it is composed rather
    // than with every mesh collapsed onto the origin. Face indices are offset by the vertices
    // already written, which is what makes the per-object index spaces concatenate into one file.
    // The same Mesh may appear more than once under different transforms; its positions are written
    // again per entry, since OBJ has no instancing.
    //
    // Objects with no faces are skipped rather than rejected, so one empty mesh in a scene does not
    // fail the export. Throws std::invalid_argument if `objects` is empty, if any entry has a null
    // mesh, or if no entry contributed any geometry; std::runtime_error on file errors.
    static void write(std::span<const ObjExportObject> objects, const std::filesystem::path &path,
                      const ObjExporterConfig &config = {});

    static void write(std::span<const ObjExportObject> objects, std::ostream &out,
                      const ObjExporterConfig &config = {});
};

} // namespace lr
