#include "core/loaders/ObjExporter.hpp"
#include "core/loaders/ObjLoader.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <span>
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

std::string exportToString(const std::vector<ObjExportObject> &objects, const ObjExporterConfig &config = {})
{
    std::ostringstream out;
    ObjExporter::write(std::span<const ObjExportObject>(objects), out, config);
    return out.str();
}

std::vector<std::string> linesStartingWith(const std::string &text, const std::string &prefix)
{
    std::vector<std::string> matches;
    std::istringstream       in(text);
    for (std::string line; std::getline(in, line);)
    {
        if (line.rfind(prefix, 0) == 0)
        {
            matches.push_back(line);
        }
    }
    return matches;
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

// Two triangles sharing an edge. positionIndices is deliberately longer than positions so the two
// vertex domains cannot be confused for one another: render vertex 3 is a second corner at
// position 0, the way a uv seam duplicates a corner in a loaded mesh.
Mesh makeMesh()
{
    Mesh mesh;
    mesh.setTopology({{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}}, {0, 1, 2, 0},
                     {{0, 1, 2}, {3, 1, 2}});
    return mesh;
}

// A triangle whose three positions are each influenced by one joint, so a palette can move them
// independently and the test can see which joint reached which vertex.
Mesh makeSkinnedMesh()
{
    Mesh mesh;
    mesh.enableVertexGroups();
    mesh.setTopology({{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}}, {0, 1, 2}, {{0, 1, 2}});

    const VertexGroupEntry joint0{0, 1.0f};
    const VertexGroupEntry joint1{1, 1.0f};
    mesh.setVertexGroups(0, std::span<const VertexGroupEntry>(&joint0, 1));
    mesh.setVertexGroups(1, std::span<const VertexGroupEntry>(&joint1, 1));
    mesh.setVertexGroups(2, std::span<const VertexGroupEntry>(&joint0, 1));
    return mesh;
}

glm::mat4 translation(float x, float y, float z)
{
    glm::mat4 m(1.0f);
    m[3] = glm::vec4(x, y, z, 1.0f);
    return m;
}

std::filesystem::path makeScratchDir()
{
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "lr_obj_exporter_tests";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

// ---------------------------------------------------------------------------

void testGeometryOnly()
{
    const std::string text = exportToString(makeMesh());

    // One v per unique position, and nothing else: no vt/vn even though the face corners live in
    // the longer render-vertex domain, and no usemtl.
    assert(countLinesStartingWith(text, "v ") == 3);
    assert(countLinesStartingWith(text, "f ") == 2);
    assert(countLinesStartingWith(text, "vt ") == 0);
    assert(countLinesStartingWith(text, "vn ") == 0);
    assert(countLinesStartingWith(text, "usemtl ") == 0);
    assert(countLinesStartingWith(text, "mtllib ") == 0);

    // Bare 1-based position indices, no slashes. Face 1's first corner is render vertex 3, which
    // resolves back through positionIndices to position 0 -- so both faces come out identical,
    // which is exactly what dropping the per-corner attributes means.
    assert(countLinesStartingWith(text, "f 1 2 3") == 2);
    assert(!contains(text, "/"));
}

// Attributes present on the mesh must still be left out of the file.
void testAttributesAreIgnored()
{
    Mesh mesh = makeMesh();
    mesh.setPerVertexArray<glm::vec3>("normal", std::vector<glm::vec3>(4, {0, 0, 1}));
    mesh.setPerVertexArray<glm::vec2>("uv", std::vector<glm::vec2>(4, {0.5f, 0.5f}));
    mesh.setFaceGroups({3, 7});

    const std::string text = exportToString(mesh);
    assert(countLinesStartingWith(text, "vn ") == 0);
    assert(countLinesStartingWith(text, "vt ") == 0);
    assert(countLinesStartingWith(text, "usemtl ") == 0);
    assert(countLinesStartingWith(text, "v ") == 3);
    assert(countLinesStartingWith(text, "f ") == 2);
}

void testObjectNameAndPrecision()
{
    Mesh mesh;
    mesh.setTopology({{0.1f, 0, 0}, {1, 0, 0}, {0, 1, 0}}, {0, 1, 2}, {{0, 1, 2}});

    ObjExporterConfig config;
    config.objectName      = "fluid_surface";
    const std::string text = exportToString(mesh, config);
    assert(text.rfind("o fluid_surface\n", 0) == 0);

    // 0.1f is not representable in binary32, so max_digits10 exposes the real stored value while
    // a low precision rounds it back to a short decimal.
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

// The real check: the engine's own loader must read back the geometry the exporter wrote.
// Authoring the source OBJ here rather than committing a fixture keeps the two from drifting.
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
                  "f 1 2 3\n"
                  "f 1 3 4\n";
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

    // Positions survive exactly: max_digits10 makes the decimal round-trip bit-exact, and the
    // loader dedupes by position value, so the unique-position domain is rebuilt identically.
    for (uint32_t i = 0; i < mesh.uniquePositionCount(); ++i)
    {
        assert(result.positionAt(i) == mesh.positionAt(i));
    }

    // Each face corner must still resolve to the same point in space after the trip through text,
    // whatever order the loader happened to assign its render vertices.
    for (uint32_t face = 0; face < mesh.faceCount(); ++face)
    {
        for (int corner = 0; corner < 3; ++corner)
        {
            const uint32_t before = mesh.faces()[face][corner];
            const uint32_t after  = result.faces()[face][corner];
            assert(result.positionAt(result.positionIndices()[after]) ==
                   mesh.positionAt(mesh.positionIndices()[before]));
        }
    }

    std::filesystem::remove_all(dir);
}

// Several objects in one file: each gets its own `o` group, and the face indices of every object
// after the first must be shifted past the vertices already written, or they would address the
// first object's vertices.
void testMultipleObjectsShareOneIndexSpace()
{
    const Mesh mesh = makeMesh();

    const std::vector<ObjExportObject> objects = {
        {&mesh, glm::mat4(1.0f), "first"},
        {&mesh, glm::mat4(1.0f), "second"},
    };
    const std::string text = exportToString(objects);

    assert(countLinesStartingWith(text, "o ") == 2);
    assert(contains(text, "o first\n"));
    assert(contains(text, "o second\n"));

    // Two objects of 3 unique positions and 2 faces each.
    assert(countLinesStartingWith(text, "v ") == 6);
    assert(countLinesStartingWith(text, "f ") == 4);

    // The first object's faces stay 1-based from the file start; the second's are offset by its 3
    // vertices. Both faces of makeMesh() collapse to the same triple (see testGeometryOnly).
    const std::vector<std::string> faces = linesStartingWith(text, "f ");
    assert(faces.size() == 4);
    assert(faces[0] == "f 1 2 3");
    assert(faces[1] == "f 1 2 3");
    assert(faces[2] == "f 4 5 6");
    assert(faces[3] == "f 4 5 6");
}

// The per-object transform is what keeps a scene composed, so it has to reach the coordinates.
void testTransformIsBakedIntoPositions()
{
    Mesh mesh;
    mesh.setTopology({{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}}, {0, 1, 2}, {{0, 1, 2}});

    glm::mat4 translated(1.0f);
    translated[3] = glm::vec4(10.0f, 20.0f, 30.0f, 1.0f);
    glm::mat4 scaled(1.0f);
    scaled[0][0] = 2.0f;

    const std::vector<ObjExportObject> objects = {
        {&mesh, translated, "moved"},
        {&mesh, scaled, "scaled"},
    };
    ObjExporterConfig config;
    config.precision       = 6;
    const std::string text = exportToString(objects, config);

    // Translation reaches every vertex of its own object.
    assert(contains(text, "v 10 20 30\n"));
    assert(contains(text, "v 11 20 30\n"));
    assert(contains(text, "v 10 21 30\n"));

    // The second object is scaled, not translated -- the transforms do not accumulate across
    // objects, which a single offset tracked across the stream would have done.
    assert(contains(text, "v 0 0 0\n"));
    assert(contains(text, "v 2 0 0\n"));
    assert(contains(text, "v 0 1 0\n"));
}

// An empty mesh among populated ones must not fail the whole export, since one empty object in a
// scene is ordinary. Its `o` statement goes too: a group with no geometry is noise.
void testEmptyObjectsAreSkipped()
{
    const Mesh mesh = makeMesh();
    const Mesh empty;

    const std::vector<ObjExportObject> objects = {
        {&empty, glm::mat4(1.0f), "nothing"},
        {&mesh, glm::mat4(1.0f), "something"},
    };
    const std::string text = exportToString(objects);

    assert(countLinesStartingWith(text, "o ") == 1);
    assert(contains(text, "o something\n"));
    assert(!contains(text, "nothing"));
    assert(countLinesStartingWith(text, "v ") == 3);

    // Skipping must not consume index space either, or the surviving object's faces would be
    // shifted past vertices that were never written.
    assert(countLinesStartingWith(text, "f 1 2 3") == 2);
}

void testInvalidObjectListsThrow()
{
    const Mesh mesh = makeMesh();

    expectThrow<std::invalid_argument>([&] {
        exportToString(std::vector<ObjExportObject>{});
    });

    expectThrow<std::invalid_argument>([&] {
        exportToString(std::vector<ObjExportObject>{{nullptr, glm::mat4(1.0f), "null"}});
    });

    // Nothing to write at all is rejected like the single-mesh no-faces case.
    const Mesh empty;
    expectThrow<std::invalid_argument>([&] {
        exportToString(std::vector<ObjExportObject>{{&empty, glm::mat4(1.0f), "a"}, {&empty, glm::mat4(1.0f), "b"}});
    });

    expectThrow<std::invalid_argument>([&] {
        const std::vector<ObjExportObject> objects = {{&mesh, glm::mat4(1.0f), "m"}};
        ObjExporter::write(std::span<const ObjExportObject>(objects), std::filesystem::path{});
    });
}

// The loader must read a multi-object export back as one combined mesh with every object's geometry
// at its exported place -- the end-to-end check that the index offsetting is right.
void testMultiObjectRoundTripThroughObjLoader()
{
    const std::filesystem::path dir      = makeScratchDir();
    const std::filesystem::path exported = dir / "scene.obj";

    Mesh triangle;
    triangle.setTopology({{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}}, {0, 1, 2}, {{0, 1, 2}});

    glm::mat4 offset(1.0f);
    offset[3] = glm::vec4(5.0f, 0.0f, 0.0f, 1.0f);

    const std::vector<ObjExportObject> objects = {
        {&triangle, glm::mat4(1.0f), "left"},
        {&triangle, offset, "right"},
    };
    ObjExporter::write(std::span<const ObjExportObject>(objects), exported);

    MaterialStore  materials(16, [] {
        return Material{};
    });
    MeshLoadResult reloaded = ObjLoader::load(exported, materials);
    const Mesh    &result   = reloaded.sequence.frames.front();

    assert(result.faceCount() == 2);
    // Six distinct points: the two instances do not overlap, so none of them dedupe together.
    assert(result.uniquePositionCount() == 6);

    bool foundOrigin = false;
    bool foundOffset = false;
    for (uint32_t i = 0; i < result.uniquePositionCount(); ++i)
    {
        foundOrigin = foundOrigin || result.positionAt(i) == glm::vec3(0.0f, 0.0f, 0.0f);
        foundOffset = foundOffset || result.positionAt(i) == glm::vec3(5.0f, 0.0f, 0.0f);
    }
    assert(foundOrigin);
    assert(foundOffset);

    std::filesystem::remove_all(dir);
}

// The palette has to reach the written coordinates, per vertex, through each vertex's own joint.
void testJointPaletteDeformsPositions()
{
    const Mesh mesh = makeSkinnedMesh();

    // Joint 0 moves along +x, joint 1 along +y, so each vertex lands somewhere only its own joint
    // could have put it.
    const std::vector<glm::mat4> palette = {translation(10.0f, 0.0f, 0.0f), translation(0.0f, 100.0f, 0.0f)};

    const std::vector<ObjExportObject> objects = {{&mesh, glm::mat4(1.0f), "posed", palette}};
    ObjExporterConfig                  config;
    config.precision       = 6;
    const std::string text = exportToString(objects, config);

    // Positions 0 and 2 follow joint 0; position 1 follows joint 1.
    assert(contains(text, "v 10 0 0\n"));  // position 0 (0,0,0) through joint 0
    assert(contains(text, "v 1 100 0\n")); // position 1 (1,0,0) through joint 1
    assert(contains(text, "v 10 1 0\n"));  // position 2 (0,1,0) through joint 0

    // Exactly the three vertices, none left at the bind pose.
    assert(countLinesStartingWith(text, "v ") == 3);
    assert(!contains(text, "v 0 0 0\n"));
}

// An empty palette is the "no pose" signal, and must leave the mesh at its bind pose rather than
// collapsing it -- this is the path an unskinned object and a skinning-disabled one both take.
void testEmptyPaletteWritesBindPose()
{
    const Mesh mesh = makeSkinnedMesh();

    const std::vector<ObjExportObject> objects = {{&mesh, glm::mat4(1.0f), "bind", {}}};
    ObjExporterConfig                  config;
    config.precision       = 6;
    const std::string text = exportToString(objects, config);

    assert(contains(text, "v 0 0 0\n"));
    assert(contains(text, "v 1 0 0\n"));
    assert(contains(text, "v 0 1 0\n"));
}

// The order the vertex shader uses: skin matrix first (into mesh-local posed space), model matrix
// second. Composed the other way round the translation would be scaled, so this pins the order.
void testTransformAppliesOnTopOfSkinning()
{
    Mesh mesh;
    mesh.enableVertexGroups();
    mesh.setTopology({{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}}, {0, 1, 2}, {{0, 1, 2}});
    const VertexGroupEntry joint0{0, 1.0f};
    for (uint32_t i = 0; i < 3; ++i)
    {
        mesh.setVertexGroups(i, std::span<const VertexGroupEntry>(&joint0, 1));
    }

    // Skin translates by +1 on x; the object transform scales x by 2.
    const std::vector<glm::mat4> palette = {translation(1.0f, 0.0f, 0.0f)};
    glm::mat4                    scaled(1.0f);
    scaled[0][0] = 2.0f;

    const std::vector<ObjExportObject> objects = {{&mesh, scaled, "scaled", palette}};
    ObjExporterConfig                  config;
    config.precision       = 6;
    const std::string text = exportToString(objects, config);

    // (1,0,0) -> skin -> (2,0,0) -> scale -> (4,0,0). Skinning after scaling would give (3,0,0).
    assert(contains(text, "v 4 0 0\n"));
    assert(!contains(text, "v 3 0 0\n"));
}

// Weights are normalized by their sum, matching skinning.glslh, so weights that do not add to 1
// still land the vertex between its joints rather than somewhere scaled by the total.
void testWeightsAreNormalized()
{
    Mesh mesh;
    mesh.enableVertexGroups();
    mesh.setTopology({{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}}, {0, 1, 2}, {{0, 1, 2}});

    // Deliberately summing to 0.5, not 1: equal halves of that sum must still mean "midway".
    const std::vector<VertexGroupEntry> influences = {{0, 0.25f}, {1, 0.25f}};
    mesh.setVertexGroups(0, influences);

    const std::vector<glm::mat4> palette = {translation(0.0f, 0.0f, 0.0f), translation(10.0f, 0.0f, 0.0f)};

    const std::vector<ObjExportObject> objects = {{&mesh, glm::mat4(1.0f), "blended", palette}};
    ObjExporterConfig                  config;
    config.precision       = 6;
    const std::string text = exportToString(objects, config);

    // Midway between staying put and moving 10 along x. Un-normalized it would be (2.5,0,0).
    assert(contains(text, "v 5 0 0\n"));
    assert(!contains(text, "v 2.5 0 0\n"));
}

// A position with no influences at all keeps its bind-pose place: the shader's identity fallback,
// not the zero matrix the weighted sum starts from.
void testUninfluencedVerticesKeepBindPose()
{
    Mesh mesh;
    mesh.enableVertexGroups();
    mesh.setTopology({{7.0f, 8.0f, 9.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}}, {0, 1, 2}, {{0, 1, 2}});

    // Only position 1 is influenced; 0 and 2 are left with no entries.
    const VertexGroupEntry joint0{0, 1.0f};
    mesh.setVertexGroups(1, std::span<const VertexGroupEntry>(&joint0, 1));

    const std::vector<glm::mat4>       palette = {translation(0.0f, 0.0f, 50.0f)};
    const std::vector<ObjExportObject> objects = {{&mesh, glm::mat4(1.0f), "partial", palette}};
    ObjExporterConfig                  config;
    config.precision       = 6;
    const std::string text = exportToString(objects, config);

    // Untouched, rather than at the origin.
    assert(contains(text, "v 7 8 9\n"));
    assert(contains(text, "v 0 1 0\n"));
    // The one influenced vertex did move.
    assert(contains(text, "v 1 0 50\n"));
}

// A palette too short for the mesh's influences is a rig/mesh mismatch, not something to write out
// with a silently wrong pose.
void testOutOfRangeJointThrows()
{
    const Mesh mesh = makeSkinnedMesh();

    // makeSkinnedMesh references joints 0 and 1, so a one-entry palette is one short.
    const std::vector<glm::mat4> shortPalette = {glm::mat4(1.0f)};
    expectThrow<std::invalid_argument>([&] {
        exportToString(std::vector<ObjExportObject>{{&mesh, glm::mat4(1.0f), "short", shortPalette}});
    });
}

// A skinned and an unskinned object in one file: the pose must not leak across entries, and the
// index offsetting still has to hold.
void testSkinnedAndUnskinnedObjectsCoexist()
{
    const Mesh skinnedMesh = makeSkinnedMesh();
    const Mesh plainMesh   = makeMesh();

    const std::vector<glm::mat4> palette = {translation(0.0f, 0.0f, 20.0f), translation(0.0f, 0.0f, 20.0f)};

    const std::vector<ObjExportObject> objects = {
        {&skinnedMesh, glm::mat4(1.0f), "posed", palette},
        {&plainMesh, glm::mat4(1.0f), "plain", {}},
    };
    ObjExporterConfig config;
    config.precision       = 6;
    const std::string text = exportToString(objects, config);

    // The skinned object moved along z; the unskinned one stayed at its bind pose.
    assert(contains(text, "v 0 0 20\n"));
    assert(contains(text, "v 0 0 0\n"));

    assert(countLinesStartingWith(text, "v ") == 6);
    const std::vector<std::string> faces = linesStartingWith(text, "f ");
    assert(faces.size() == 3);
    assert(faces[0] == "f 1 2 3");
    assert(faces[1] == "f 4 5 6");
    assert(faces[2] == "f 4 5 6");
}

} // namespace

int main()
{
    testGeometryOnly();
    testAttributesAreIgnored();
    testObjectNameAndPrecision();
    testInvalidMeshesThrow();
    testRoundTripThroughObjLoader();
    testMultipleObjectsShareOneIndexSpace();
    testTransformIsBakedIntoPositions();
    testEmptyObjectsAreSkipped();
    testInvalidObjectListsThrow();
    testMultiObjectRoundTripThroughObjLoader();
    testJointPaletteDeformsPositions();
    testEmptyPaletteWritesBindPose();
    testTransformAppliesOnTopOfSkinning();
    testWeightsAreNormalized();
    testUninfluencedVerticesKeepBindPose();
    testOutOfRangeJointThrows();
    testSkinnedAndUnskinnedObjectsCoexist();
    return 0;
}
