#include "features/fluid/import/BlenderFluidCacheImporter.hpp"
#include "features/fluid/import/BlenderUniImporter.hpp"
#include "features/fluid/import/VdbFluidImporter.hpp"
#include "features/eulerian_particle/import/BlenderSmokeCacheImporter.hpp"

#include <zlib.h>

#include <array>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

namespace
{

template <typename T> void append(std::vector<std::byte> &bytes, const T &value)
{
    const auto *first = reinterpret_cast<const std::byte *>(&value);
    bytes.insert(bytes.end(), first, first + sizeof(T));
}

std::filesystem::path writeUniFixture()
{
    std::vector<std::byte> bytes;
    append(bytes, int32_t(16));
    for (int value : {32, 32, 32})
    {
        append(bytes, int32_t(value));
    }
    append(bytes, 0.03125f);
    append(bytes, 0.1f);
    for (float value : {-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f})
    {
        append(bytes, value);
    }
    for (int value : {0, 0, 0})
    {
        append(bytes, int32_t(value));
    }
    for (float value : {0.0f, 0.0f, 0.0f})
    {
        append(bytes, value);
    }
    const std::array<float, 16> blenderTransform{
        2.0f, 0.0f, 0.0f, 0.0f, 0.0f, 3.0f, 0.0f, 0.0f, 0.0f, 0.0f, 4.0f, 0.0f, 1.0f, 2.0f, 3.0f, 1.0f,
    };
    for (float value : blenderTransform)
    {
        append(bytes, value);
    }
    for (int value : {32, 32, 32, 0, 0, 0, 32, 32, 32})
    {
        append(bytes, int32_t(value));
    }
    for (float value : {0.2f, 0.3f, 0.4f})
    {
        append(bytes, value);
    }
    append(bytes, 2.5f); // Blender writes the float's four bytes through an int-sized write.
    for (char value : {'C', '0', '1', '\0'})
    {
        append(bytes, value);
    }
    assert(bytes.size() == 204);

    const auto path   = std::filesystem::temp_directory_path() / "lr_blender_uni_import_test.uni";
    gzFile     output = gzopen(path.string().c_str(), "wb");
    assert(output);
    assert(gzwrite(output, bytes.data(), static_cast<unsigned int>(bytes.size())) == bytes.size());
    assert(gzclose(output) == Z_OK);
    return path;
}

void importsUniConfiguration()
{
    const auto path   = writeUniFixture();
    const auto config = lr::BlenderUniImporter::importFile(path);
    std::filesystem::remove(path);
    assert(config.activeFields == 16);
    assert(config.resolution == glm::ivec3(32));
    assert(config.baseResolution == glm::ivec3(32));
    assert(config.cellSize == 0.03125f);
    assert(config.totalTime == 2.5f);
    assert(config.versionString() == "C01");
    assert(config.objectTransform[0][0] == 2.0f);
    assert(glm::vec3(config.objectTransform * glm::vec4(0, 0, 0, 1)) == glm::vec3(1, 2, 3));
}

// The DNA helix cache committed under assets/samples/fluid. Unlike the reference Blender bake
// below, this one ships with the repo, so it exercises the whole directory import path -- numbered
// file pairing, the UNI configuration, the VDB point grid and the domain re-centring -- on every
// machine rather than only on the one that produced the bake.
void importsShippedDnaHelixSample()
{
    const std::filesystem::path cache = std::filesystem::path(LR_SAMPLE_ASSET_DIR) / "fluid" / "dna_helix";
    // Git LFS stores pointer files until fetched, so a shallow checkout has the paths but not the
    // contents. Skipping keeps that case from reading as an importer failure.
    if (!std::filesystem::exists(cache / "data" / "fluid_data_0001.vdb"))
    {
        return;
    }

    const lr::OfflineFluidSimulation clip = lr::BlenderFluidCacheImporter::importDirectory(cache);
    assert(clip.frames().size() == 24);
    assert(clip.frames().front().sourceFrame == 1);
    assert(clip.frames().back().sourceFrame == 24);
    // Velocities are off by default: nothing renders them yet, so a default import must not pay
    // for the channel.
    assert(!clip.hasVelocities());

    // Every frame of this clip holds the same rigidly rotating structure.
    const uint32_t particles = clip.frames().front().particleCount;
    assert(particles == 1808);
    for (const auto &frame : clip.frames())
    {
        assert(frame.particleCount == particles);
    }

    // Frame times come from the source frame numbers at the clip's frame rate: 24 frames at 24fps
    // is one second, with the last sample one frame short of it.
    assert(std::abs(clip.durationSeconds() - 1.0f) < 1e-5f);
    assert(std::abs(clip.frames().back().seconds - 23.0f / 24.0f) < 1e-5f);

    // centerDomain puts the domain centre on the object origin, so the helix straddles it.
    const auto &bounds = clip.frames().front();
    assert(bounds.boundsMin.x < 0.0f && bounds.boundsMax.x > 0.0f);
    assert(bounds.boundsMin.y < 0.0f && bounds.boundsMax.y > 0.0f);
    assert(bounds.boundsMin.z < 0.0f && bounds.boundsMax.z > 0.0f);

    // Derived from the cache's voxel size (1/64), which is what a Blender cache supplies.
    assert(std::abs(clip.defaultParticleRadius() - 0.5f / 64.0f) < 1e-6f);

    // Up-axis conversion. The cache is authored in Blender's Z-up axes with the helix running
    // along +Z, so a correct import puts its long axis on the engine's up axis, +Y. Getting this
    // wrong renders the clip lying on its side, which no other assertion here would catch.
    const glm::vec3 extent = bounds.boundsMax - bounds.boundsMin;
    assert(extent.y > extent.x * 1.5f && extent.y > extent.z * 1.5f);
    // The helix is round about its own axis, so the two cross-axes stay equal under the rotation.
    assert(std::abs(extent.x - extent.z) < 1e-5f);

    // The conversion must be a rotation, not a mirror or a scale: lengths are preserved and the
    // mapping is exactly (x, y, z)_blender -> (x, z, -y)_engine. Re-importing with the conversion
    // off gives the Blender-space reference to compare against.
    lr::BlenderFluidCacheImporter::Options unconverted;
    unconverted.convertUpAxis                 = false;
    const lr::OfflineFluidSimulation blenderSpace =
        lr::BlenderFluidCacheImporter::importDirectory(cache, unconverted);
    const std::span<const glm::vec3> converted = clip.framePositions(0);
    const std::span<const glm::vec3> original  = blenderSpace.framePositions(0);
    assert(converted.size() == original.size());
    for (std::size_t i = 0; i < converted.size(); ++i)
    {
        assert(converted[i] == glm::vec3(original[i].x, original[i].z, -original[i].y));
        assert(std::abs(glm::length(converted[i]) - glm::length(original[i])) < 1e-5f);
    }
}

void importsReferenceBlenderCacheWhenAvailable()
{
    const std::filesystem::path cache(R"(C:\Users\seani\AppData\Local\Temp\blender_a37396\cache_fluid_ab1c5968)");
    const auto                  singleFrame = cache / "data" / "fluid_data_0001.vdb";
    if (!std::filesystem::exists(singleFrame))
    {
        return;
    }

    lr::VdbFluidImporter::Options vdbOptions;
    // This cache does carry a velocity attribute; asking for it explicitly is what the default no
    // longer does, so the channel stays covered.
    vdbOptions.importVelocities          = true;
    const lr::OfflineFluidSimulation vdb = lr::VdbFluidImporter::importFile(singleFrame, vdbOptions);
    assert(vdb.frames().size() == 1);
    assert(vdb.framePositions(0).size() == 6776);
    assert(vdb.hasVelocities());
    assert(!lr::VdbFluidImporter::importFile(singleFrame).hasVelocities());

    lr::BlenderFluidCacheImporter::Options options;
    options.firstFrame                    = 1;
    options.lastFrame                     = 3;
    const lr::OfflineFluidSimulation clip = lr::BlenderFluidCacheImporter::importDirectory(cache, options);
    assert(clip.frames().size() == 3);
    assert(clip.frames()[0].sourceFrame == 1 && clip.frames()[2].sourceFrame == 3);
    assert(clip.framePositions(0).size() == 6776);
    assert(std::abs(clip.frames()[1].seconds - 1.0f / 24.0f) < 1e-6f);
    assert(clip.frames()[0].boundsMin.x < 0.0f && clip.frames()[0].boundsMax.x > 0.0f);

    // This bake is liquid settling under gravity in a cube domain. After the up-axis conversion
    // that fall has to show on the engine's up axis: the top of the fluid drops frame over frame
    // while the horizontal extents stay pinned to the domain walls. Before the conversion existed
    // this signal appeared on Z, i.e. the fluid drained sideways on screen.
    assert(clip.frames()[1].boundsMax.y < clip.frames()[0].boundsMax.y);
    assert(clip.frames()[2].boundsMax.y < clip.frames()[1].boundsMax.y);
    assert(std::abs(clip.frames()[2].boundsMax.x - clip.frames()[0].boundsMax.x) < 1e-5f);
    assert(std::abs(clip.frames()[2].boundsMax.z - clip.frames()[0].boundsMax.z) < 1e-5f);
}

void importsReferenceBlenderSmokeCacheWhenAvailable()
{
    const std::filesystem::path cache(
        R"(C:\Users\seani\AppData\Local\Temp\blender_a37396\cache_fluid_c306d54c)");
    if (!std::filesystem::exists(cache / "data" / "fluid_data_0001.vdb")) return;

    const auto clip = lr::eulerian::BlenderSmokeCacheImporter::importDirectory(cache);
    assert(clip.frames().size() == 100);
    assert(clip.gridSize() == glm::uvec3(32));
    assert(std::abs(clip.voxelSize().x - 0.23825003f) < 1e-5f);
    assert(std::abs(clip.worldSize().x - 7.624001f) < 1e-4f);
    assert(clip.frames().front().sourceFrame == 1 && clip.frames().back().sourceFrame == 100);
    assert(std::ranges::any_of(clip.frameDensity(49), [](float value) { return value > 0.0f; }));
    assert(std::ranges::any_of(clip.frameTemperature(49), [](float value) { return value > 0.0f; }));
}

void reportsOutOfRangeSmokeFrameNumbers()
{
    const std::filesystem::path cache =
        std::filesystem::temp_directory_path() / "lr_blender_smoke_frame_number_test";
    std::filesystem::remove_all(cache);
    std::filesystem::create_directories(cache / "data");
    const std::filesystem::path malformed = cache / "data" / "fluid_data_99999999999.vdb";
    std::ofstream(malformed).put('\0');

    std::string message;
    try
    {
        (void)lr::eulerian::BlenderSmokeCacheImporter::importDirectory(cache);
    }
    catch (const std::runtime_error &error)
    {
        message = error.what();
    }
    std::filesystem::remove_all(cache);

    assert(message.find("BlenderSmokeCacheImporter:") != std::string::npos);
    assert(message.find("99999999999") != std::string::npos);
}

} // namespace

int main()
{
    importsUniConfiguration();
    importsShippedDnaHelixSample();
    importsReferenceBlenderCacheWhenAvailable();
    importsReferenceBlenderSmokeCacheWhenAvailable();
    reportsOutOfRangeSmokeFrameNumbers();
    return 0;
}
