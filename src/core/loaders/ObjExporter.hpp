#pragma once

#include "core/loaders/MaterialStore.hpp"
#include "core/scene/Mesh.hpp"

#include <filesystem>
#include <iosfwd>
#include <string>

namespace lr
{
struct ObjExporterConfig
{
    // Per-vertex attributes read off the Mesh. An attribute that is absent, or registered with a
    // type other than the expected one, is skipped rather than failing the export — OBJ treats
    // texcoords and normals as optional, and a Mesh built by another loader may carry neither.
    std::string normalAttributeName = "normal";
    std::string uvAttributeName     = "uv";

    // Decimal digits written per float. OBJ is a text format, so this trades file size against
    // precision; 9 is max_digits10 for IEEE-754 binary32, the point at which a written float
    // reads back bit-exact.
    int precision = 9;

    // When set, each distinct faceGroups value is looked up in this store and emitted as
    // `usemtl <material name>`. Without it the groups are still emitted, named positionally
    // (see ObjExporter::write). The materials themselves are never written: an .mtl sidecar
    // would have to re-encode every texture, which is out of scope here.
    const MaterialStore *materialStore = nullptr;

    // Emitted as a single `o` statement before the face list. Empty omits it.
    std::string objectName;
};

class ObjExporter
{
public:
    // Writes `mesh` as Wavefront OBJ, one `v` line per Mesh::positions() entry, so `f` indices
    // address the unique-position domain directly. The render-vertex domain (positionIndices)
    // only supplies the vt/vn index of each face corner, which is exactly OBJ's own split
    // between shared positions and per-corner normals/texcoords.
    //
    // Throws std::invalid_argument if the mesh has no faces, or if its faceGroups length does
    // not match its face count, and std::runtime_error if the file cannot be opened or written.
    static void write(const Mesh &mesh, const std::filesystem::path &path, const ObjExporterConfig &config = {});

    // Same, to an already-open stream. The caller owns flushing and error checking; use the path
    // overload to have write failures turned into exceptions.
    static void write(const Mesh &mesh, std::ostream &out, const ObjExporterConfig &config = {});
};

} // namespace lr
