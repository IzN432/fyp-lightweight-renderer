// Writes the assets/samples/fluid/dna_helix Blender-format fluid cache that ships with the repo.
//
// This is a build-time authoring tool, not part of the renderer. It exists so the committed sample
// is reproducible: Blender's own bake of an equivalent helix is not, and a procedural clip built in
// C++ at startup would bypass the importers this sample is meant to exercise. The output is a real
// OpenVDB PointDataGrid cache plus Blender's 204-byte gzip-compressed Mantaflow domain
// configuration per frame, so BlenderFluidCacheImporter reads it by the same path as a Blender bake.
//
// Run: generate_dna_helix_cache <output-cache-directory>
//
// Output is stable but not byte-reproducible: OpenVDB stamps each file with a fresh UUID in its
// header, so re-running this rewrites ~31 header bytes per .vdb while every point sample stays
// identical. Regenerate only when the sample itself needs to change, or the LFS objects churn for
// no reason.

#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>
#include <openvdb/points/AttributeArray.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointDataGrid.h>

#include <zlib.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

namespace
{

// Domain shape. The cache is authored on a 64^3 Mantaflow-style grid: positions below are built in
// VDB world units with the origin at the domain's lower corner, which is where Blender puts them,
// and BlenderFluidCacheImporter re-centres them on the object origin when it reads the cache.
constexpr int   kResolution      = 64;
constexpr float kCellSize        = 1.0f / static_cast<float>(kResolution);
constexpr int   kFrameCount      = 24;
constexpr int   kFirstFrame      = 1;
constexpr float kFramesPerSecond = 24.0f;

// Two strands plus the base pairs bridging them. Particle counts are chosen so the helix reads as a
// continuous surface through the screen-space liquid pass rather than as separate beads.
constexpr int kParticlesPerStrand = 760;
constexpr int kRungCount          = 12;
constexpr int kParticlesPerRung   = 24;

constexpr float kTau = 6.28318530717958647692f;

// Helix shape in domain-local units. Sized so the structure occupies the middle of the domain,
// leaving the margin Blender's own caches normally carry around the fluid.
constexpr float kHelixRadius = 0.17f;
constexpr float kHelixHeight = 0.62f;
constexpr float kTurns       = 2.0f;

struct Particle
{
    openvdb::Vec3f position;
    openvdb::Vec3f velocity;
};

// One frame of the animation. The helix rotates about Y by a full turn over the clip, so the frame
// after the last matches the first and the cache loops seamlessly.
std::vector<Particle> buildFrame(int frameIndex)
{
    const float phase = kTau * static_cast<float>(frameIndex) / static_cast<float>(kFrameCount);
    // Angular speed of the whole helix, in radians per second: one revolution over the clip.
    const float spin   = kTau * kFramesPerSecond / static_cast<float>(kFrameCount);
    const float centre = 0.5f * static_cast<float>(kResolution) * kCellSize;

    std::vector<Particle> particles;
    particles.reserve(static_cast<std::size_t>(kParticlesPerStrand) * 2 +
                      static_cast<std::size_t>(kRungCount) * kParticlesPerRung);

    // Written in Blender's axes, because that is what this file claims to be: Blender is Z-up, so
    // the helix runs along +Z and rotates about Z. BlenderFluidCacheImporter rotates it into the
    // engine's Y-up convention on import, which is exactly the path a real bake takes -- authoring
    // it Y-up here would produce a file no Blender would write and would hide that conversion.
    //
    // A point rigidly rotating about the domain's Z axis has velocity omega x r; both strands and
    // rungs share that, so the exported velocity channel is consistent with the motion.
    const auto emit = [&](float radius, float angle, float height) {
        const float x = radius * std::cos(angle);
        const float y = radius * std::sin(angle);
        Particle    particle;
        particle.position = {centre + x, centre + y, centre + height};
        // d/dt (r cos(a + spin t), r sin(a + spin t), h) = (-spin * y, spin * x, 0).
        particle.velocity = {-spin * y, spin * x, 0.0f};
        particles.push_back(particle);
    };

    for (int strand = 0; strand < 2; ++strand)
    {
        const float strandPhase = static_cast<float>(strand) * 0.5f * kTau;
        for (int i = 0; i < kParticlesPerStrand; ++i)
        {
            const float t = static_cast<float>(i) / static_cast<float>(kParticlesPerStrand - 1);
            emit(kHelixRadius, kTurns * kTau * t + strandPhase + phase, kHelixHeight * (t - 0.5f));
        }
    }

    // Base pairs. Each rung spans the two strands at a fixed height, so it inherits the same
    // rotation and the structure reads as one rigid object.
    for (int rung = 0; rung < kRungCount; ++rung)
    {
        const float t     = static_cast<float>(rung + 1) / static_cast<float>(kRungCount + 1);
        const float angle = kTurns * kTau * t + phase;
        for (int i = 0; i < kParticlesPerRung; ++i)
        {
            // Spans strand A (at +radius) through the axis to strand B (at -radius).
            const float across = -1.0f + 2.0f * static_cast<float>(i) / static_cast<float>(kParticlesPerRung - 1);
            emit(kHelixRadius * across, angle, kHelixHeight * (t - 0.5f));
        }
    }
    return particles;
}

void writeVdbFrame(const fs::path &path, const std::vector<Particle> &particles)
{
    std::vector<openvdb::Vec3f> positions;
    std::vector<openvdb::Vec3f> velocities;
    positions.reserve(particles.size());
    velocities.reserve(particles.size());
    for (const Particle &particle : particles)
    {
        positions.push_back(particle.position);
        velocities.push_back(particle.velocity);
    }

    // Blender's caches carry one voxel-sized linear transform, which is what gives the importer the
    // voxel size it derives the suggested particle radius from.
    openvdb::math::Transform::Ptr transform = openvdb::math::Transform::createLinearTransform(kCellSize);

    const openvdb::points::PointAttributeVector<openvdb::Vec3f> positionWrapper(positions);
    openvdb::tools::PointIndexGrid::Ptr                         pointIndex =
        openvdb::tools::createPointIndexGrid<openvdb::tools::PointIndexGrid>(positionWrapper, *transform);
    openvdb::points::PointDataGrid::Ptr grid =
        openvdb::points::createPointDataGrid<openvdb::points::NullCodec, openvdb::points::PointDataGrid>(
            *pointIndex, positionWrapper, *transform);

    // The importer looks for this attribute by name to populate the clip's optional velocity
    // channel, matching the name Blender's Mantaflow export uses.
    openvdb::points::appendAttribute<openvdb::Vec3f>(grid->tree(), "particles_velocity");
    const openvdb::points::PointAttributeVector<openvdb::Vec3f> velocityWrapper(velocities);
    openvdb::points::populateAttribute(grid->tree(), pointIndex->tree(), "particles_velocity", velocityWrapper);

    // The grid name the importer reads by default (VdbFluidImporter::Options::pointGridName).
    grid->setName("particles");

    openvdb::io::File   file(path.string());
    openvdb::GridPtrVec grids;
    grids.push_back(grid);
    file.write(grids);
    file.close();
}

template <typename T> void append(std::vector<std::byte> &bytes, const T &value)
{
    const auto *first = reinterpret_cast<const std::byte *>(&value);
    bytes.insert(bytes.end(), first, first + sizeof(T));
}

// Blender's per-frame Mantaflow domain record: 204 bytes, gzip-compressed, little-endian. Field
// order matches BlenderUniImporter::importFile exactly.
void writeUniFrame(const fs::path &path, int frameIndex)
{
    std::vector<std::byte> bytes;
    // Active fields bitmask. Blender sets this per enabled physics channel; the importer only
    // reports it, so the value here is descriptive rather than load-bearing.
    append(bytes, int32_t(16));
    for (int axis = 0; axis < 3; ++axis) append(bytes, int32_t(kResolution));
    append(bytes, kCellSize);
    append(bytes, 1.0f / kFramesPerSecond);

    const float extent = static_cast<float>(kResolution) * kCellSize;
    for (int axis = 0; axis < 3; ++axis) append(bytes, 0.0f);          // bounds min
    for (int axis = 0; axis < 3; ++axis) append(bytes, extent);        // bounds max
    for (int axis = 0; axis < 3; ++axis) append(bytes, 0.0f);          // previous bounds min
    for (int axis = 0; axis < 3; ++axis) append(bytes, int32_t(0));    // shift
    for (int axis = 0; axis < 3; ++axis) append(bytes, 0.0f);          // object shift

    // Blender stores a row-vector matrix row-major with translation in row 3; identity here, since
    // the sample's domain sits at the world origin.
    const std::array<float, 16> transform{
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
    };
    for (float value : transform) append(bytes, value);

    for (int axis = 0; axis < 3; ++axis) append(bytes, int32_t(kResolution)); // base resolution
    for (int axis = 0; axis < 3; ++axis) append(bytes, int32_t(0));           // resolution min
    for (int axis = 0; axis < 3; ++axis) append(bytes, int32_t(kResolution)); // resolution max
    for (int axis = 0; axis < 3; ++axis) append(bytes, 0.0f);                 // active color

    const float totalTime = static_cast<float>(frameIndex) / kFramesPerSecond;
    int32_t     timeBits{};
    std::memcpy(&timeBits, &totalTime, sizeof(timeBits));
    append(bytes, timeBits); // Blender writes the float's four bytes through an int-sized write.

    const std::array<char, 4> version{'C', '0', '1', '\0'};
    for (char value : version) append(bytes, value);

    if (bytes.size() != 204) throw std::runtime_error("UNI record is not 204 bytes");

    gzFile output = gzopen(path.string().c_str(), "wb");
    if (!output) throw std::runtime_error("cannot open " + path.string());
    const bool wrote =
        gzwrite(output, bytes.data(), static_cast<unsigned int>(bytes.size())) == static_cast<int>(bytes.size());
    const bool closed = gzclose(output) == Z_OK;
    if (!wrote || !closed) throw std::runtime_error("cannot write " + path.string());
}

std::string numbered(const char *prefix, int frame, const char *extension)
{
    std::ostringstream name;
    name << prefix << std::setfill('0') << std::setw(4) << frame << extension;
    return name.str();
}

} // namespace

int main(int argc, char **argv)
try
{
    if (argc != 2)
    {
        std::cerr << "usage: generate_dna_helix_cache <output-cache-directory>\n";
        return 2;
    }
    openvdb::initialize();

    const fs::path cache = argv[1];
    fs::create_directories(cache / "data");
    fs::create_directories(cache / "config");

    for (int i = 0; i < kFrameCount; ++i)
    {
        const int frame = kFirstFrame + i;
        writeVdbFrame(cache / "data" / numbered("fluid_data_", frame, ".vdb"), buildFrame(i));
        writeUniFrame(cache / "config" / numbered("config_", frame, ".uni"), i);
    }

    const std::size_t particles =
        static_cast<std::size_t>(kParticlesPerStrand) * 2 + static_cast<std::size_t>(kRungCount) * kParticlesPerRung;
    std::cout << "Wrote " << kFrameCount << " frames of " << particles << " particles to " << cache.string() << "\n";
    return 0;
} catch (const std::exception &e)
{
    std::cerr << "generate_dna_helix_cache: " << e.what() << "\n";
    return 1;
}
