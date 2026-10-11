#include "core/loaders/ObjExporter.hpp"
#include "core/loaders/ObjLoader.hpp"

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace lr;

namespace
{

template <typename Exception, typename Action> void expectThrow(Action action)
{
    bool threw = false;
    try
    {
        action();
    } catch (const Exception &)
    {
        threw = true;
    }
    assert(threw);
}

std::string exportToString(const Mesh &mesh, const ObjExporterConfig &config = {})
{
    std::ostringstream out;
    ObjExporter::write(mesh, out, config);
    return out.str();
}

bool contains(const std::string &haystack, const std::string &needle)
{
    return haystack.find(needle) != std::string::npos;
}

size_t countLinesStartingWith(const std::string &text, const std::string &prefix)
{
    size_t             count = 0;
    std::istringstream in(text);
    for (std::string line; std::getline(in, line);)
    {
        if (line.rfind(prefix, 0) == 0)
        {
            ++count;
        }
    }
    return count;
}

// Two triangles sharing an edge, with per-corner normals and uvs. positionIndices is deliberately
// longer than positions so that the two vertex domains cannot be confused for one another: render
// vertex 3 is a second corner at position 0, the way a uv seam duplicates a corner.
Mesh makeMesh()
{
    Mesh mesh;
    mesh.setTopology({{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}}, {0, 1, 2, 0},
                     {{0, 1, 2}, {3, 1, 2}});
    mesh.setPerVertexArray<glm::vec3>("normal", std::vector<glm::vec3>{{0, 0, 1}, {0, 1, 0}, {1, 0, 0}, {0, 0, -1}});
    mesh.setPerVertexArray<glm::vec2>("uv", std::vector<glm::vec2>{{0, 0}, {1, 0}, {0, 1}, {0.5f, 0.5f}});
    return mesh;
}

std::filesystem::path makeScratchDir()
{
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "lr_obj_exporter_tests";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

// ---------------------------------------------------------------------------

void testIndicesAddressTheUniquePositionDomain()
{
    const std::string text = exportToString(makeMesh());

    // One v per unique position, one vt/vn per render vertex — not one of each per corner.
    assert(countLinesStartingWith(text, "v ") == 3);
    assert(countLinesStartingWith(text, "vt ") == 4);
    assert(countLinesStartingWith(text, "vn ") == 4);
    assert(countLinesStartingWith(text, "f ") == 2);

    // Face 0's corners are render vertices 0,1,2 at positions 0,1,2; face 1's first corner is
    // render vertex 3, which shares position 0 while keeping its own uv/normal. All 1-based.
    assert(contains(text, "f 1/1/1 2/2/2 3/3/3\n"));
    assert(contains(text, "f 1/4/4 2/2/2 3/3/3\n"));
}

void testOptionalAttributes()
{
    Mesh mesh;
    mesh.setTopology({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}, {0, 1, 2}, {{0, 1, 2}});

    // Neither attribute present: positions only, so bare indices with no slashes.
    const std::string plain = exportToString(mesh);
    assert(countLinesStartingWith(plain, "vt ") == 0);
    assert(countLinesStartingWith(plain, "vn ") == 0);
    assert(contains(plain, "f 1 2 3\n"));

    // Normals but no uvs must use OBJ's empty-texcoord form, v//vn.
    mesh.setPerVertexArray<glm::vec3>("normal", std::vector<glm::vec3>(3, {0, 0, 1}));
    const std::string normalsOnly = exportToString(mesh);
    assert(countLinesStartingWith(normalsOnly, "vn ") == 3);
    assert(countLinesStartingWith(normalsOnly, "vt ") == 0);
    assert(contains(normalsOnly, "f 1//1 2//2 3//3\n"));

    // A name that is not in the layout, or is registered with another type, is skipped rather
    // than throwing: OBJ treats both attributes as optional.
    ObjExporterConfig missing;
    missing.normalAttributeName = "nope";
    assert(countLinesStartingWith(exportToString(mesh, missing), "vn ") == 0);

    Mesh wrongType;
    wrongType.setTopology({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}, {0, 1, 2}, {{0, 1, 2}});
    wrongType.setPerVertexArray<glm::vec2>("normal", std::vector<glm::vec2>(3, {0, 0}));
    assert(countLinesStartingWith(exportToString(wrongType), "vn ") == 0);
}

void testFaceGroupsBecomeUsemtl()
{
    Mesh mesh;
    mesh.setTopology({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}}, {0, 1, 2, 3}, {{0, 1, 2}, {1, 3, 2}, {0, 2, 3}});
    // Groups 7,7,9: one usemtl for the run, a second at the change, and none repeated.
    mesh.setFaceGroups({7, 7, 9});

    const std::string unnamed = exportToString(mesh);
    assert(countLinesStartingWith(unnamed, "usemtl ") == 2);
    assert(contains(unnamed, "usemtl group_7\n"));
    assert(contains(unnamed, "usemtl group_9\n"));

    MaterialStore store(16, [] {
        return Material{};
    });
    Material      steel;
    steel.name                  = "steel";
    const MaterialHandle handle = store.acquire(std::move(steel));
    mesh.setFaceGroups({handle, handle, handle});

    ObjExporterConfig named;
    named.materialStore         = &store;
    const std::string withNames = exportToString(mesh, named);
    assert(countLinesStartingWith(withNames, "usemtl ") == 1);
    assert(contains(withNames, "usemtl steel\n"));

    // A handle past the store's capacity falls back to the positional name instead of a
    // out-of-range read, and an unnamed real material does too — "usemtl" alone is not valid OBJ.
    mesh.setFaceGroups({999, 999, 999});
    assert(contains(exportToString(mesh, named), "usemtl group_999\n"));
}

void testObjectNameAndPrecision()
{
    Mesh mesh;
    mesh.setTopology({{0.1f, 0, 0}, {1, 0, 0}, {0, 1, 0}}, {0, 1, 2}, {{0, 1, 2}});

    ObjExporterConfig config;
    config.objectName      = "fluid_surface";
    const std::string text = exportToString(mesh, config);
    assert(text.rfind("o fluid_surface\n", 0) == 0);

    // 0.1f is not representable in binary32, so max_digits10 exposes the real stored value
    // while a low precision rounds it back to a short decimal.
    assert(contains(text, "0.100000001"));
    config.precision = 3;
    assert(contains(exportToString(mesh, config), "v 0.1 0 0\n"));
}

void testInvalidMeshesThrow()
{
    Mesh empty;
    expectThrow<std::invalid_argument>([&] {
        exportToString(empty);
    });

    expectThrow<std::invalid_argument>([&] {
        ObjExporter::write(makeMesh(), std::filesystem::path{});
    });
    expectThrow<std::runtime_error>([&] {
        ObjExporter::write(makeMesh(), std::filesystem::temp_directory_path() / "lr_no_such_dir" / "x.obj");
    });
}

// The real check: the engine's own loader must read back what the exporter wrote, with the same
// geometry. Authoring the source OBJ here rather than committing a fixture keeps the two from
// drifting apart.
void testRoundTripThroughObjLoader()
{
    const std::filesystem::path dir      = makeScratchDir();
    const std::filesystem::path original = dir / "original.obj";
    const std::filesystem::path exported = dir / "exported.obj";

    {
        std::ofstream source(original, std::ios::binary);
        source << "o quad\n"
                  "v 0 0 0\n"
                  "v 1 0 0\n"
                  "v 1 1 0\n"
                  "v 0 1 0\n"
                  "vt 0 0\n"
                  "vt 1 0\n"
                  "vt 1 1\n"
                  "vt 0 1\n"
                  "vn 0 0 1\n"
                  "f 1/1/1 2/2/1 3/3/1\n"
                  "f 1/1/1 3/3/1 4/4/1\n";
    }

    MaterialStore  materials(16, [] {
        return Material{};
    });
    MeshLoadResult loaded = ObjLoader::load(original, materials);
    assert(loaded.sequence.frames.size() == 1);
    const Mesh &mesh = loaded.sequence.frames.front();

    ObjExporter::write(mesh, exported);

    MaterialStore  reloadedMaterials(16, [] {
        return Material{};
    });
    MeshLoadResult reloaded = ObjLoader::load(exported, reloadedMaterials);
    const Mesh    &result   = reloaded.sequence.frames.front();

    assert(result.faceCount() == mesh.faceCount());
    assert(result.uniquePositionCount() == mesh.uniquePositionCount());
    assert(result.vertexCount() == mesh.vertexCount());

    // Positions survive exactly: max_digits10 makes the decimal round-trip bit-exact, and the
    // loader dedupes by position value, so the unique-position domain is rebuilt identically.
    for (uint32_t i = 0; i < mesh.uniquePositionCount(); ++i)
    {
        assert(result.positionAt(i) == mesh.positionAt(i));
    }

    // Each face corner must still resolve to the same point in space after the trip through
    // text, whatever order the loader happened to assign its render vertices.
    for (uint32_t face = 0; face < mesh.faceCount(); ++face)
    {
        for (int corner = 0; corner < 3; ++corner)
        {
            const uint32_t before = mesh.faces()[face][corner];
            const uint32_t after  = result.faces()[face][corner];
            assert(result.positionAt(result.positionIndices()[after]) ==
                   mesh.positionAt(mesh.positionIndices()[before]));
            assert(result.perVertexAt<glm::vec3>("normal", after) == mesh.perVertexAt<glm::vec3>("normal", before));
            assert(result.perVertexAt<glm::vec2>("uv", after) == mesh.perVertexAt<glm::vec2>("uv", before));
        }
    }

    std::filesystem::remove_all(dir);
}

} // namespace

int main()
{
    testIndicesAddressTheUniquePositionDomain();
    testOptionalAttributes();
    testFaceGroupsBecomeUsemtl();
    testObjectNameAndPrecision();
    testInvalidMeshesThrow();
    testRoundTripThroughObjLoader();
    return 0;
}
