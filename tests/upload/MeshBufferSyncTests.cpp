#include "core/editor/VertexManager.hpp"
#include "core/upload/MeshBufferPacking.hpp"

#include <cassert>
#include <cstring>
#include <stdexcept>
#include <type_traits>

using namespace lr;

static_assert(std::is_same_v<decltype(std::declval<Mesh &>().positionAt(0)), const glm::vec3 &>);
static_assert(std::is_same_v<decltype(std::declval<Mesh &>().perVertexAt<glm::vec3>("normal", 0)),
                             const glm::vec3 &>);

template <typename Exception, typename Action> void expectThrow(Action action)
{
    bool threw = false;
    try { action(); } catch (const Exception &) { threw = true; }
    assert(threw);
}

Mesh makeMesh()
{
    Mesh mesh;
    mesh.setTopology({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}, {0, 1, 0, 2}, {{0, 1, 3}, {2, 1, 3}});
    mesh.setPerVertexArray<glm::vec3>("normal", std::vector<glm::vec3>(4, {0, 0, 1}));
    mesh.setPerUniqueVertexArray<glm::vec3>("color", std::vector<glm::vec3>(3, {1, 1, 1}));
    mesh.setPerUniqueVertexArray<glm::vec3>("heatmap", std::vector<glm::vec3>{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
    return mesh;
}

glm::vec3 readVec3(const std::vector<std::byte> &bytes, size_t offset)
{
    glm::vec3 value;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

void testMutationRevisions()
{
    auto mesh = makeMesh();
    const auto topology = mesh.topologyRevision();
    const auto normal = mesh.perVertexRevision("normal");
    auto positions = mesh.positionsRevision();
    const auto *address = mesh.positions().data();
    mesh.setPositionAt(0, mesh.positionAt(0));
    mesh.setPositions(mesh.positions());
    assert(mesh.positionsRevision() == positions);

    const std::vector<uint32_t> indices{0, 1};
    const std::vector<glm::vec3> values{{2, 0, 0}, {3, 0, 0}};
    mesh.setPositions(indices, values);
    assert(mesh.positionsRevision() > positions);
    assert(mesh.positions().data() == address); // Selection's position span stays valid.
    assert(mesh.topologyRevision() == topology);
    assert(mesh.perVertexRevision("normal") == normal); // Authored normals aren't regenerated.
    positions = mesh.positionsRevision();
    expectThrow<std::out_of_range>([&] { mesh.setPositions(std::vector<uint32_t>{0, 99}, values); });
    expectThrow<std::invalid_argument>([&] { mesh.setPositions(std::vector<glm::vec3>(2)); });
    assert(mesh.positionsRevision() == positions);
    assert(mesh.positionAt(0) == values[0]); // Invalid batches have not partially written.
    const auto oldPositions = mesh.positions();
    mesh.setPositions(std::vector<uint32_t>{1, 0, 2}, mesh.positions());
    assert(mesh.positionAt(0) == oldPositions[1]);
    assert(mesh.positionAt(1) == oldPositions[0]);
    positions = mesh.positionsRevision();
    VertexManager editor(mesh);
    editor.translateSelectedVertices({0, 1}, {0, 0, 1});
    assert(mesh.positionsRevision() == positions + 1); // One publication for an editor batch.
    assert(mesh.topologyRevision() == topology);

    mesh.setPerVertexAt<glm::vec3>("normal", 0, {0, 1, 0});
    assert(mesh.perVertexRevision("normal") > normal);
    const auto updatedNormal = mesh.perVertexRevision("normal");
    expectThrow<std::out_of_range>([&] { mesh.setPerVertexAt<glm::vec3>("normal", 99, {0, 1, 0}); });
    assert(mesh.perVertexRevision("normal") == updatedNormal);
    auto colors = mesh.perUniqueVertexRevision("color");
    mesh.setPerUniqueVertexAt<glm::vec3>("color", 0, {1, 0, 1});
    assert(mesh.perUniqueVertexRevision("color") > colors);

    mesh.setPerFaceArray<uint32_t>("face", std::vector<uint32_t>{0, 1});
    auto face = mesh.perFaceRevision("face");
    mesh.setPerFaceAt<uint32_t>("face", 1, 2);
    assert(mesh.perFaceRevision("face") > face);
    mesh.setFaceGroupAttributeArray<float>("material", std::vector<float>{1});
    auto material = mesh.faceGroupAttributeRevision("material");
    mesh.setFaceGroupAttributeAt<float>("material", 0, 2);
    assert(mesh.faceGroupAttributeRevision("material") > material);
    mesh.enableVertexGroups();
    mesh.setVertexGroupAttributeArray<float>("joint", std::vector<float>{1});
    auto joint = mesh.vertexGroupAttributeRevision("joint");
    mesh.setVertexGroupAttributeAt<float>("joint", 0, 2);
    assert(mesh.vertexGroupAttributeRevision("joint") > joint);
    auto groups = mesh.vertexGroupsRevision();
    mesh.setVertexGroups(0, std::vector<VertexGroupEntry>{{0, 1}});
    assert(mesh.vertexGroupsRevision() > groups);
    groups = mesh.vertexGroupsRevision();
    assert(mesh.rawGroupEntries().size() == 1);
    assert(mesh.rawGroupOffsets().size() == 4);
    assert(mesh.vertexGroupsRevision() == groups); // Lazy CSR reads aren't source mutations.
}

void testIndependentConsumersAndPacking()
{
    auto mesh = makeMesh();
    auto other = makeMesh();
    std::vector<const Mesh *> scene{&mesh, &other};
    std::vector<const Mesh *> selected{&mesh};
    VertexBufferUploadConfig positions{.vertexBufferName = "positions", .includePosition = true};
    VertexBufferUploadConfig normals{.vertexBufferName = "normals", .vertexAttributeNames = {"normal"}};
    VertexBufferUploadConfig points{.vertexBufferName = "points", .vertexAttributeNames = {"color"}, .includePosition = true};
    VertexBufferUploadConfig heatmap{.vertexBufferName = "heatmap", .vertexAttributeNames = {"heatmap"},
                                    .includePosition = true, .expandUniqueVertexAttributes = true};
    MeshBufferCache cache;
    cache.remember("positions", MeshBufferCache::vertices(scene, positions));
    cache.remember("normals", MeshBufferCache::vertices(scene, normals));
    cache.remember("points", MeshBufferCache::vertices(selected, points, true));
    cache.remember("heatmap", MeshBufferCache::vertices(selected, heatmap));
    cache.remember("indices", MeshBufferCache::indices(scene));
    cache.remember("groups", MeshBufferCache::faceGroups(scene));
    int queued = 0;
    auto sync = [&](const auto &sources, const auto &config, bool unique = false) {
        return cache.synchronize(config.vertexBufferName, MeshBufferCache::vertices(sources, config, unique), [&] {
            const auto bytes = packMeshVertexData(sources, config, unique);
            assert(!bytes.empty());
            ++queued;
        });
    };
    assert(!sync(scene, positions));
    mesh.setPositionAt(0, {1, 2, 3});
    mesh.setPositionAt(0, {4, 5, 6});
    assert(sync(scene, positions));
    assert(!sync(scene, positions));
    assert(!sync(scene, normals));
    assert(!cache.synchronize("indices", MeshBufferCache::indices(scene), [&] { ++queued; }));
    assert(!cache.synchronize("groups", MeshBufferCache::faceGroups(scene), [&] { ++queued; }));
    assert(sync(selected, points, true));
    // An inactive overlay has not consumed the edit; it catches up when next synchronized.
    assert(sync(selected, heatmap));
    assert(queued == 3);

    auto bytes = packMeshVertexData(selected, heatmap);
    const auto stride = 2 * sizeof(glm::vec3);
    GpuMeshLayout layout(mesh.layout());
    layout.mapPosition(0, 0, VK_FORMAT_R32G32B32_SFLOAT);
    layout.mapUniqueVertex("heatmap", 0, 1, VK_FORMAT_R32G32B32_SFLOAT);
    assert(layout.bindingDescriptions()[0].stride == stride);
    assert(layout.attributeDescriptions()[0].offset == 0);
    assert(layout.attributeDescriptions()[1].offset == sizeof(glm::vec3));
    assert(bytes.size() == 4 * stride);
    assert(readVec3(bytes, 0) == glm::vec3(4, 5, 6));
    assert(readVec3(bytes, 2 * stride) == glm::vec3(4, 5, 6));
    assert(readVec3(bytes, sizeof(glm::vec3)) == glm::vec3(1, 0, 0));
    assert(readVec3(bytes, 2 * stride + sizeof(glm::vec3)) == glm::vec3(1, 0, 0));
    assert(readVec3(bytes, 3 * stride + sizeof(glm::vec3)) == glm::vec3(0, 0, 1));

    mesh.setPerUniqueVertexAt<glm::vec3>("heatmap", 0, {0.5f, 0.5f, 0});
    assert(sync(selected, heatmap));
    assert(!sync(selected, points, true));
    assert(!sync(scene, positions));
    mesh.setPerUniqueVertexAt<glm::vec3>("color", 0, {0, 0, 0});
    assert(sync(selected, points, true));
    assert(!sync(selected, heatmap));
    other.setPositionAt(0, {9, 8, 7}); // Non-selected mesh edits are observed too.
    assert(sync(scene, positions));
    assert(!sync(selected, points, true));
    bytes = packMeshVertexData(scene, positions);
    assert(readVec3(bytes, 4 * sizeof(glm::vec3)) == glm::vec3(9, 8, 7));
    mesh.setFaceGroups({1, 2});
    assert(cache.synchronize("groups", MeshBufferCache::faceGroups(scene), [&] { ++queued; }));
    assert(!sync(scene, positions));
}

void testFailureReplacementAndTopology()
{
    auto mesh = makeMesh();
    VertexBufferUploadConfig config{.vertexBufferName = "positions", .includePosition = true};
    MeshBufferCache cache;
    cache.remember("positions", MeshBufferCache::vertices({&mesh}, config));
    mesh.setPositionAt(0, {4, 0, 0});
    expectThrow<std::runtime_error>([&] {
        cache.synchronize("positions", MeshBufferCache::vertices({&mesh}, config), [] { throw std::runtime_error("queue failed"); });
    });
    assert(cache.synchronize("positions", MeshBufferCache::vertices({&mesh}, config), [] {}));
    cache.remember("indices", MeshBufferCache::indices({&mesh}));
    mesh.setTopology(mesh.positions(), {0, 1, 2, 0}, {{0, 2, 1}, {3, 2, 1}});
    assert(cache.synchronize("indices", MeshBufferCache::indices({&mesh}), [] {}));
    assert(cache.synchronize("positions", MeshBufferCache::vertices({&mesh}, config), [] {}));
    auto differentAsset = makeMesh();
    // Identity is part of the key, not just the asset-local revision numbers.
    cache.remember("positions", MeshBufferCache::vertices({&differentAsset}, config));
    auto equalRevisions = makeMesh();
    assert(cache.synchronize("positions", MeshBufferCache::vertices({&equalRevisions}, config), [] {}));
    equalRevisions.setTopology({{0, 0, 0}}, {0}, {});
    expectThrow<std::logic_error>([&] {
        cache.synchronize("positions", MeshBufferCache::vertices({&equalRevisions}, config), [] { assert(false); });
    });
    cache.remember("positions", MeshBufferCache::vertices({&equalRevisions}, config));
    assert(!cache.synchronize("positions", MeshBufferCache::vertices({&equalRevisions}, config), [] { assert(false); }));
}

int main()
{
    testMutationRevisions();
    testIndependentConsumersAndPacking();
    testFailureReplacementAndTopology();
}
