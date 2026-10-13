#include "core/scene/serialization/ComponentCodec.hpp"
#include "core/scene/serialization/JsonUtils.hpp"

#include "core/scene/SceneObject.hpp"
#include "features/fluid/FluidComponent.hpp"

#include <memory>
#include <stdexcept>
#include <string>

namespace lr
{
namespace
{
using namespace scene_serialization;

class FluidComponentCodec final : public ComponentCodec
{
public:
    std::string_view key() const override { return "fluid"; }
    std::type_index  componentType() const override { return typeid(FluidComponent); }

    json encode(const SceneObject &object, const ComponentSaveContext &context) const override
    {
        const FluidComponent &component = object.getComponent<FluidComponent>();
        if (component.isBaked())
        {
            const FluidComponent::BakedConfig &config = component.bakedConfig();
            return {{"mode", "baked"},
                    {"simulation", toString(context.offlineFluids.idOf(config.simulation))},
                    {"playback_speed", config.playbackSpeed},
                    {"loop", config.loop},
                    {"start_paused", config.startPaused},
                    {"particle_radius", config.particleRadius},
                    {"smoothing_radius_scale", config.smoothingRadiusScale},
                    {"particle_color", vec3(config.particleColor)},
                    {"render_scale", config.renderScale}};
        }

        const FluidComponent::Config &config  = component.config();
        json                          sources = json::array();
        for (const FluidComponent::CuboidSource &source : config.sources)
        {
            sources.push_back({{"dimensions", {source.dimensions.x, source.dimensions.y, source.dimensions.z}},
                               {"origin", vec3(source.origin)}});
        }
        return {{"mode", "live"},
                {"sources", std::move(sources)},
                {"max_particles", config.maxParticles},
                {"particle_spacing", config.particleSpacing},
                {"smoothing_radius_scale", config.smoothingRadiusScale},
                {"particle_render_scale", config.particleRenderScale},
                {"solver_iterations", config.solverIterations},
                {"artificial_pressure_k", config.artificialPressureK},
                {"vorticity_epsilon", config.vorticityEpsilon},
                {"viscosity", config.viscosity},
                {"bounds_friction", config.boundsFriction},
                {"max_displacement_scale", config.maxDisplacementScale},
                {"bounds_min", vec3(config.boundsMin)},
                {"bounds_max", vec3(config.boundsMax)},
                {"show_bounds", config.showBounds},
                {"fixed_delta_time", config.fixedDeltaTime},
                {"max_steps_per_frame", config.maxStepsPerFrame},
                {"start_paused", config.startPaused},
                {"particle_color", vec3(config.particleColor)},
                {"render_scale", config.renderScale}};
    }

    void decode(const json &value, SceneObject &object, ComponentLoadContext &context,
                const std::string &where) const override
    {
        // Version 4 and earlier had no discriminator and always represented a live solver.
        const std::string mode = value.contains("mode") ? required(value, "mode", where).get<std::string>() : "live";
        if (mode == "baked")
        {
            const OfflineFluidSimulationId id     = parseUuid(required(value, "simulation", where).get<std::string>());
            const auto                     handle = context.offlineFluids.find(id);
            if (!handle)
            {
                throw std::runtime_error("SceneSerializer: missing offline fluid simulation " + toString(id) + " at " +
                                         where + ".simulation");
            }
            FluidComponent::BakedConfig config;
            config.simulation           = *handle;
            config.playbackSpeed        = required(value, "playback_speed", where).get<float>();
            config.loop                 = required(value, "loop", where).get<bool>();
            config.startPaused          = required(value, "start_paused", where).get<bool>();
            config.particleRadius       = required(value, "particle_radius", where).get<float>();
            config.smoothingRadiusScale = required(value, "smoothing_radius_scale", where).get<float>();
            config.particleColor =
                readVector<3, float>(required(value, "particle_color", where), where + ".particle_color");
            config.renderScale = required(value, "render_scale", where).get<float>();
            object.addComponent<FluidComponent>(std::move(config));
            return;
        }
        if (mode != "live")
        {
            throw std::runtime_error("SceneSerializer: invalid fluid mode '" + mode + "' at " + where);
        }

        FluidComponent::Config config;
        config.sources.clear();
        const json &sources = required(value, "sources", where);
        if (!sources.is_array())
        {
            throw std::runtime_error("SceneSerializer: " + where + ".sources must be an array");
        }
        for (size_t i = 0; i < sources.size(); ++i)
        {
            const std::string sourceWhere = where + ".sources[" + std::to_string(i) + "]";
            const json       &source      = sources[i];
            config.sources.push_back({
                readVector<3, uint32_t>(required(source, "dimensions", sourceWhere), sourceWhere + ".dimensions"),
                readVector<3, float>(required(source, "origin", sourceWhere), sourceWhere + ".origin"),
            });
        }
        config.maxParticles         = required(value, "max_particles", where).get<uint32_t>();
        config.particleSpacing      = required(value, "particle_spacing", where).get<float>();
        config.smoothingRadiusScale = required(value, "smoothing_radius_scale", where).get<float>();
        config.particleRenderScale  = required(value, "particle_render_scale", where).get<float>();
        config.solverIterations     = required(value, "solver_iterations", where).get<int>();
        config.artificialPressureK  = required(value, "artificial_pressure_k", where).get<float>();
        config.vorticityEpsilon     = required(value, "vorticity_epsilon", where).get<float>();
        config.viscosity            = required(value, "viscosity", where).get<float>();
        config.boundsFriction       = required(value, "bounds_friction", where).get<float>();
        config.maxDisplacementScale = required(value, "max_displacement_scale", where).get<float>();
        config.boundsMin            = readVector<3, float>(required(value, "bounds_min", where), where + ".bounds_min");
        config.boundsMax            = readVector<3, float>(required(value, "bounds_max", where), where + ".bounds_max");
        config.showBounds           = required(value, "show_bounds", where).get<bool>();
        config.fixedDeltaTime       = required(value, "fixed_delta_time", where).get<float>();
        config.maxStepsPerFrame     = required(value, "max_steps_per_frame", where).get<int>();
        config.startPaused          = required(value, "start_paused", where).get<bool>();
        config.particleColor =
            readVector<3, float>(required(value, "particle_color", where), where + ".particle_color");
        config.renderScale = required(value, "render_scale", where).get<float>();
        object.addComponent<FluidComponent>(std::move(config));
    }
};
} // namespace

void addFluidComponentCodecs(ComponentCodecRegistry &registry)
{
    registry.add(std::make_unique<FluidComponentCodec>());
}

} // namespace lr
