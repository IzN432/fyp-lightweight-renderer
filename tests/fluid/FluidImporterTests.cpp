#include "features/fluid/import/BlenderFluidCacheImporter.hpp"
#include "features/fluid/import/BlenderUniImporter.hpp"
#include "features/fluid/import/VdbFluidImporter.hpp"

#include <zlib.h>

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
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

void importsReferenceBlenderCacheWhenAvailable()
{
    const std::filesystem::path cache(R"(C:\Users\seani\AppData\Local\Temp\blender_a37396\cache_fluid_ab1c5968)");
    const auto                  singleFrame = cache / "data" / "fluid_data_0001.vdb";
    if (!std::filesystem::exists(singleFrame))
    {
        return;
    }

    const lr::OfflineFluidSimulation vdb = lr::VdbFluidImporter::importFile(singleFrame);
    assert(vdb.frames().size() == 1);
    assert(vdb.framePositions(0).size() == 6776);
    assert(vdb.hasVelocities());

    lr::BlenderFluidCacheImporter::Options options;
    options.firstFrame                    = 1;
    options.lastFrame                     = 3;
    const lr::OfflineFluidSimulation clip = lr::BlenderFluidCacheImporter::importDirectory(cache, options);
    assert(clip.frames().size() == 3);
    assert(clip.frames()[0].sourceFrame == 1 && clip.frames()[2].sourceFrame == 3);
    assert(clip.framePositions(0).size() == 6776);
    assert(std::abs(clip.frames()[1].seconds - 1.0f / 24.0f) < 1e-6f);
    assert(clip.frames()[0].boundsMin.x < 0.0f && clip.frames()[0].boundsMax.x > 0.0f);
}

} // namespace

int main()
{
    importsUniConfiguration();
    importsReferenceBlenderCacheWhenAvailable();
    return 0;
}
