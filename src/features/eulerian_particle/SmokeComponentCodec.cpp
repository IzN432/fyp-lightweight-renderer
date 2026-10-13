#include "core/scene/serialization/ComponentCodec.hpp"
#include "core/scene/serialization/JsonUtils.hpp"

#include "core/scene/SceneObject.hpp"
#include "features/eulerian_particle/SmokeComponent.hpp"

#include <memory>
#include <utility>

namespace lr
{
namespace
{
using namespace scene_serialization;
using eulerian::SmokeComponent;

class SmokeComponentCodec final : public ComponentCodec
{
public:
    std::string_view key() const override { return "smoke"; }
    std::type_index  componentType() const override { return typeid(SmokeComponent); }

    json encode(const SceneObject &object, const ComponentSaveContext &context) const override
    {
        const SmokeComponent &component = object.getComponent<SmokeComponent>();
        if (component.isBaked())
        {
            const auto &c = component.bakedConfig();
            return {{"mode", "baked"},
                    {"simulation", toString(context.offlineSmokes.idOf(c.simulation))},
                    {"playback_speed", c.playbackSpeed}, {"loop", c.loop},
                    {"start_paused", c.startPaused}, {"show_bounds", c.showBounds},
                    {"enabled", c.enabled}, {"bounds_centre", vec3(c.boundsCentre)},
                    {"cool_color", vec3(c.coolColor)}, {"warm_color", vec3(c.warmColor)},
                    {"cool_temperature", c.coolTemperature}, {"warm_temperature", c.warmTemperature},
                    {"density_scale", c.densityScale}, {"sample_count", c.sampleCount}};
        }
        const auto &c = component.config();
        return {
            {"mode", "live"},
            {"grid_size", {c.gridSize.x, c.gridSize.y, c.gridSize.z}},
            {"simulation_scale", c.simulationScale}, {"bounds_centre", vec3(c.boundsCentre)},
            {"show_bounds", c.showBounds}, {"enabled", c.enabled}, {"start_paused", c.startPaused},
            {"voxelize_obstacles", c.voxelizeObstacles}, {"jacobi_iterations", c.jacobiIterations},
            {"step", {{"buoyancy", c.step.buoyancy}, {"gravity", c.step.gravity},
                      {"ambient_temperature", c.step.ambientTemperature}, {"cooling", c.step.cooling},
                      {"velocity_damp", c.step.velocityDamp}, {"density_damp", c.step.densityDamp},
                      {"vorticity", c.step.vorticity}, {"open_top", c.step.openTop}}},
            {"source", {{"position", vec3(c.source.position)}, {"radius", c.source.radius},
                        {"density", c.source.density}, {"velocity", vec3(c.source.velocity)},
                        {"temperature", c.source.temperature}, {"heating", c.source.heating},
                        {"drift", c.source.drift}}},
            {"cool_color", vec3(c.coolColor)}, {"warm_color", vec3(c.warmColor)},
            {"cool_temperature", c.coolTemperature}, {"warm_temperature", c.warmTemperature},
            {"density_scale", c.densityScale}, {"sample_count", c.sampleCount},
        };
    }

    void decode(const json &value, SceneObject &object, ComponentLoadContext &context,
                const std::string &where) const override
    {
        const std::string mode = value.contains("mode") ? required(value, "mode", where).get<std::string>() : "live";
        if (mode == "baked")
        {
            const auto id = parseUuid(required(value, "simulation", where).get<std::string>());
            const auto handle = context.offlineSmokes.find(id);
            if (!handle)
                throw std::runtime_error("SceneSerializer: missing offline smoke simulation " + toString(id) +
                                         " at " + where + ".simulation");
            SmokeComponent::BakedConfig c;
            c.simulation = *handle;
            c.playbackSpeed = required(value, "playback_speed", where).get<float>();
            c.loop = required(value, "loop", where).get<bool>();
            c.startPaused = required(value, "start_paused", where).get<bool>();
            c.showBounds = required(value, "show_bounds", where).get<bool>();
            c.enabled = required(value, "enabled", where).get<bool>();
            c.boundsCentre = readVector<3, float>(required(value, "bounds_centre", where), where + ".bounds_centre");
            c.coolColor = readVector<3, float>(required(value, "cool_color", where), where + ".cool_color");
            c.warmColor = readVector<3, float>(required(value, "warm_color", where), where + ".warm_color");
            c.coolTemperature = required(value, "cool_temperature", where).get<float>();
            c.warmTemperature = required(value, "warm_temperature", where).get<float>();
            c.densityScale = required(value, "density_scale", where).get<float>();
            c.sampleCount = required(value, "sample_count", where).get<int>();
            object.addComponent<SmokeComponent>(std::move(c));
            return;
        }
        if (mode != "live")
            throw std::runtime_error("SceneSerializer: invalid smoke mode '" + mode + "' at " + where);

        SmokeComponent::Config c;
        c.gridSize = readVector<3, uint32_t>(required(value, "grid_size", where), where + ".grid_size");
        c.simulationScale = required(value, "simulation_scale", where).get<float>();
        c.boundsCentre = readVector<3, float>(required(value, "bounds_centre", where), where + ".bounds_centre");
        c.showBounds = required(value, "show_bounds", where).get<bool>();
        c.enabled = required(value, "enabled", where).get<bool>();
        c.startPaused = required(value, "start_paused", where).get<bool>();
        c.voxelizeObstacles = required(value, "voxelize_obstacles", where).get<bool>();
        c.jacobiIterations = required(value, "jacobi_iterations", where).get<int>();

        const json &step = required(value, "step", where);
        c.step.buoyancy = required(step, "buoyancy", where + ".step").get<float>();
        c.step.gravity = required(step, "gravity", where + ".step").get<float>();
        c.step.ambientTemperature = required(step, "ambient_temperature", where + ".step").get<float>();
        c.step.cooling = required(step, "cooling", where + ".step").get<float>();
        c.step.velocityDamp = required(step, "velocity_damp", where + ".step").get<float>();
        c.step.densityDamp = required(step, "density_damp", where + ".step").get<float>();
        c.step.vorticity = required(step, "vorticity", where + ".step").get<float>();
        c.step.openTop = required(step, "open_top", where + ".step").get<bool>();

        const json &source = required(value, "source", where);
        c.source.position = readVector<3, float>(required(source, "position", where + ".source"),
                                                where + ".source.position");
        c.source.radius = required(source, "radius", where + ".source").get<float>();
        c.source.density = required(source, "density", where + ".source").get<float>();
        c.source.velocity = readVector<3, float>(required(source, "velocity", where + ".source"),
                                                where + ".source.velocity");
        c.source.temperature = required(source, "temperature", where + ".source").get<float>();
        c.source.heating = required(source, "heating", where + ".source").get<float>();
        c.source.drift = required(source, "drift", where + ".source").get<float>();

        c.coolColor = readVector<3, float>(required(value, "cool_color", where), where + ".cool_color");
        c.warmColor = readVector<3, float>(required(value, "warm_color", where), where + ".warm_color");
        c.coolTemperature = required(value, "cool_temperature", where).get<float>();
        c.warmTemperature = required(value, "warm_temperature", where).get<float>();
        c.densityScale = required(value, "density_scale", where).get<float>();
        c.sampleCount = required(value, "sample_count", where).get<int>();
        object.addComponent<SmokeComponent>(std::move(c));
    }
};
} // namespace

void addSmokeComponentCodecs(ComponentCodecRegistry &registry)
{
    registry.add(std::make_unique<SmokeComponentCodec>());
}

} // namespace lr
