#include "features/fluid/import/VdbFluidImporter.hpp"

#include <openvdb/io/File.h>
#include <openvdb/openvdb.h>
#include <openvdb/points/AttributeArray.h>
#include <openvdb/points/PointDataGrid.h>
#include <openvdb/points/PointCount.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace lr
{
namespace
{

void initializeOpenVdb()
{
    static std::once_flag once;
    std::call_once(once, [] {
        openvdb::initialize();
        // Blender writes Mantaflow point-data attributes with these compact codecs. They must be
        // registered before the file instantiates its attribute arrays.
        using namespace openvdb::points;
        TypedAttributeArray<int32_t, TruncateCodec>::registerType();
        TypedAttributeArray<float, TruncateCodec>::registerType();
        TypedAttributeArray<openvdb::Vec3f, TruncateCodec>::registerType();
    });
}

glm::vec3 toGlm(const openvdb::Vec3d &value)
{
    return {static_cast<float>(value.x()), static_cast<float>(value.y()), static_cast<float>(value.z())};
}

} // namespace

VdbFluidFrame VdbFluidImporter::importFrame(const std::filesystem::path &path, const Options &options)
{
    initializeOpenVdb();
    if (!std::filesystem::is_regular_file(path))
    {
        throw std::runtime_error("VdbFluidImporter: file does not exist: '" + path.string() + "'");
    }

    openvdb::io::File file(path.string());
    try
    {
        file.open();
        openvdb::GridBase::Ptr base = file.readGrid(options.pointGridName);
        file.close();
        auto grid = openvdb::gridPtrCast<openvdb::points::PointDataGrid>(base);
        if (!grid)
        {
            throw std::runtime_error("grid '" + options.pointGridName + "' is not a PointDataGrid");
        }

        VdbFluidFrame        result;
        const openvdb::Vec3d voxel = grid->voxelSize();
        result.voxelSize           = toGlm(voxel);
        result.positions.reserve(static_cast<std::size_t>(openvdb::points::pointCount(grid->tree())));

        for (auto leaf = grid->tree().cbeginLeaf(); leaf; ++leaf)
        {
            const openvdb::points::AttributeHandle<openvdb::Vec3f> positions(leaf->constAttributeArray("P"));
            const auto                                            &descriptor = leaf->attributeSet().descriptor();
            const bool hasVelocity = options.importVelocities && descriptor.find(options.velocityAttribute) !=
                                                                     openvdb::points::AttributeSet::INVALID_POS;
            std::unique_ptr<openvdb::points::AttributeHandle<openvdb::Vec3f>> velocities;
            if (hasVelocity)
            {
                velocities = std::make_unique<openvdb::points::AttributeHandle<openvdb::Vec3f>>(
                    leaf->constAttributeArray(options.velocityAttribute));
            }

            for (auto point = leaf->beginIndexOn(); point; ++point)
            {
                const openvdb::Vec3f local = positions.get(*point);
                const openvdb::Coord coord = point.getCoord();
                const openvdb::Vec3d indexPosition(static_cast<double>(coord.x()) + local.x(),
                                                   static_cast<double>(coord.y()) + local.y(),
                                                   static_cast<double>(coord.z()) + local.z());
                result.positions.push_back(toGlm(grid->transform().indexToWorld(indexPosition)));
                if (velocities)
                {
                    const openvdb::Vec3f value = velocities->get(*point);
                    result.velocities.emplace_back(value.x(), value.y(), value.z());
                }
            }
        }
        if (!result.velocities.empty() && result.velocities.size() != result.positions.size())
        {
            throw std::runtime_error("velocity attribute is present on only part of the point grid");
        }
        return result;
    } catch (const std::exception &error)
    {
        try
        {
            file.close();
        } catch (...)
        {}
        throw std::runtime_error("VdbFluidImporter: failed to import '" + path.string() + "': " + error.what());
    }
}

OfflineFluidSimulation VdbFluidImporter::importFile(const std::filesystem::path &path, const Options &options)
{
    VdbFluidFrame          frame = importFrame(path, options);
    OfflineFluidSimulation result(path.stem().string(), options.framesPerSecond);
    result.setDefaultParticleRadius(0.5f * std::min({frame.voxelSize.x, frame.voxelSize.y, frame.voxelSize.z}));
    result.addFrame(0.0f, frame.positions, frame.velocities);
    return result;
}

} // namespace lr
