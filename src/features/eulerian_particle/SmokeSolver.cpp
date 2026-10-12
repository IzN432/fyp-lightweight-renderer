#include "SmokeSolver.hpp"

#include "core/framegraph/ResourceRegistry.hpp"
#include "core/vulkan/VulkanContext.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lr::eulerian
{
namespace
{

void checkCuda(cudaError_t result, const char *operation)
{
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("SmokeSolver: ") + operation + ": " + cudaGetErrorString(result));
}

// Every field is sampled by CUDA and may be read by the renderer, so they share one usage mask.
constexpr VkImageUsageFlags kVolumeUsage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;

struct VolumeSpec
{
    const char *suffix;
    VkFormat    format;
};

// The seven volumes the pipeline needs. Velocity and density are double-buffered because
// semi-Lagrangian advection reads the whole field while writing it; pressure is double-buffered
// for the Jacobi sweeps. Divergence is written once per step and only read afterwards.
constexpr VolumeSpec kVolumes[] = {
    {"_velocity", VK_FORMAT_R32G32B32A32_SFLOAT},
    {"_velocity_next", VK_FORMAT_R32G32B32A32_SFLOAT},
    {"_density", VK_FORMAT_R32_SFLOAT},
    {"_density_next", VK_FORMAT_R32_SFLOAT},
    {"_pressure", VK_FORMAT_R32_SFLOAT},
    {"_pressure_next", VK_FORMAT_R32_SFLOAT},
    {"_divergence", VK_FORMAT_R32_SFLOAT},
};

} // namespace

struct SmokeSolver::Impl
{
    // Index order matches kVolumes.
    std::vector<std::unique_ptr<CudaVolume>> volumes;
    cudaStream_t                             stream = nullptr;
    cudaExtent                               extent{};
    int                                      device = 0;

    CudaVolume &at(std::size_t index) { return *volumes.at(index); }

    SmokeFields fields() const
    {
        SmokeFields f{};
        f.velocity     = volumes[0]->surface();
        f.velocityNext = volumes[1]->surface();
        f.density      = volumes[2]->surface();
        f.densityNext  = volumes[3]->surface();
        f.pressure     = volumes[4]->surface();
        f.pressureNext = volumes[5]->surface();
        f.divergence   = volumes[6]->surface();
        return f;
    }

    ~Impl()
    {
        if (stream != nullptr)
        {
            (void)cudaSetDevice(device);
            (void)cudaStreamDestroy(stream);
        }
    }
};

void SmokeSolver::registerVolumes(ResourceRegistry &resources, const Config &config)
{
    for (const VolumeSpec &spec : kVolumes)
        resources.registerPersistentVolume(config.namePrefix + spec.suffix, spec.format, kVolumeUsage, config.extent,
                                           true);
}

SmokeSolver::SmokeSolver(ResourceRegistry &resources, const VulkanContext &context, Config config)
    : m_config(std::move(config)), m_impl(std::make_unique<Impl>())
{
    if (m_config.extent.width < 3 || m_config.extent.height < 3 || m_config.extent.depth < 3)
        throw std::invalid_argument("SmokeSolver: the grid must be at least 3 cells on every axis so the one-cell "
                                    "boundary leaves an interior to simulate");
    if (m_config.jacobiIterations < 0)
        throw std::invalid_argument("SmokeSolver: jacobiIterations cannot be negative");

    // Register only the volumes that are missing, so a caller may pre-declare them via
    // registerVolumes() for the frame graph and still construct the solver afterwards.
    for (const VolumeSpec &spec : kVolumes)
    {
        const std::string name = m_config.namePrefix + spec.suffix;
        if (resources.getImage(name) == nullptr)
            resources.registerPersistentVolume(name, spec.format, kVolumeUsage, m_config.extent, true);
    }

    m_impl->volumes.reserve(std::size(kVolumes));
    for (const VolumeSpec &spec : kVolumes)
        m_impl->volumes.push_back(
            std::make_unique<CudaVolume>(resources, context, m_config.namePrefix + spec.suffix));

    m_impl->extent = m_impl->volumes.front()->extent();
    m_impl->device = m_impl->volumes.front()->device();
    checkCuda(cudaSetDevice(m_impl->device), "cudaSetDevice");
    checkCuda(cudaStreamCreate(&m_impl->stream), "cudaStreamCreate");

    reset();
}

SmokeSolver::~SmokeSolver()                              = default;
SmokeSolver::SmokeSolver(SmokeSolver &&) noexcept        = default;
SmokeSolver &SmokeSolver::operator=(SmokeSolver &&) noexcept = default;

void SmokeSolver::reset()
{
    if (!m_impl)
        throw std::logic_error("SmokeSolver: cannot use a moved-from solver");

    // Imported Vulkan memory starts undefined, so zero every field before accumulating into it.
    checkCuda(cudaSetDevice(m_impl->device), "cudaSetDevice");
    for (std::unique_ptr<CudaVolume> &volume : m_impl->volumes)
    {
        volume->prepareForExternalWrite();
        checkCuda(launchClearVolume(volume->surface(), volume->extent(), volume->channels(), m_impl->stream),
                  "launchClearVolume");
    }
    checkCuda(cudaStreamSynchronize(m_impl->stream), "cudaStreamSynchronize");
}

void SmokeSolver::step(float dt)
{
    const float previous = m_config.step.dt;
    m_config.step.dt     = dt;
    try
    {
        step();
    } catch (...)
    {
        m_config.step.dt = previous;
        throw;
    }
    m_config.step.dt = previous;
}

void SmokeSolver::step()
{
    if (!m_impl)
        throw std::logic_error("SmokeSolver: cannot use a moved-from solver");

    checkCuda(cudaSetDevice(m_impl->device), "cudaSetDevice");
    for (std::unique_ptr<CudaVolume> &volume : m_impl->volumes)
        volume->prepareForExternalWrite();

    SmokeFields       fields = m_impl->fields();
    const cudaExtent  extent = m_impl->extent;
    const cudaStream_t stream = m_impl->stream;

    // 1. Source injection — add density and velocity in place before anything transports them.
    checkCuda(launchInjectSources(fields, extent, m_config.source, m_config.step.dt, stream), "launchInjectSources");

    // 2. Velocity advection — self-advect into the spare buffer, then swap so `velocity` is current.
    checkCuda(launchAdvectVelocity(fields, extent, m_config.step.dt, stream), "launchAdvectVelocity");
    std::swap(fields.velocity, fields.velocityNext);

    // 3. External forces and buoyancy (plus optional vorticity confinement), applied in place.
    checkCuda(launchApplyForces(fields, extent, m_config.step, stream), "launchApplyForces");

    // 4. Divergence of the forced velocity field — the right-hand side of the pressure equation.
    checkCuda(launchComputeDivergence(fields, extent, stream), "launchComputeDivergence");

    // 5. Jacobi pressure solve. Each sweep reads `pressure` and writes `pressureNext`, so swapping
    //    after every sweep leaves the result in `pressure` whenever the iteration count is even.
    //    Pressure is not re-zeroed: starting from the previous step's solution converges faster.
    for (int iteration = 0; iteration < m_config.jacobiIterations; ++iteration)
    {
        checkCuda(launchPressureJacobi(fields, extent, stream), "launchPressureJacobi");
        std::swap(fields.pressure, fields.pressureNext);
    }

    // 6. Pressure projection — subtract the pressure gradient to make the velocity divergence-free.
    checkCuda(launchProjectVelocity(fields, extent, stream), "launchProjectVelocity");

    // 7. Density advection along the now-incompressible velocity field.
    checkCuda(launchAdvectDensity(fields, extent, m_config.step, stream), "launchAdvectDensity");
    std::swap(fields.density, fields.densityNext);

    // 8. Boundary conditions, applied last so the border is consistent with what the renderer
    //    samples and with what the next step's stencils read.
    checkCuda(launchApplyBoundaries(fields, extent, stream), "launchApplyBoundaries");

    // The swaps above are local aliases, so copy whatever ended up current back into the volume
    // the renderer samples. A real pipeline would instead publish the resource name that holds the
    // live field; copying keeps the registry's "smoke_density"/"smoke_velocity" names authoritative.
    if (fields.density != m_impl->volumes[2]->surface())
        checkCuda(launchCopyVolume(m_impl->volumes[2]->surface(), fields.density, extent, 1, stream),
                  "launchCopyVolume(density)");
    if (fields.velocity != m_impl->volumes[0]->surface())
        checkCuda(launchCopyVolume(m_impl->volumes[0]->surface(), fields.velocity, extent, 4, stream),
                  "launchCopyVolume(velocity)");
    if (fields.pressure != m_impl->volumes[4]->surface())
        checkCuda(launchCopyVolume(m_impl->volumes[4]->surface(), fields.pressure, extent, 1, stream),
                  "launchCopyVolume(pressure)");

    // Blocking until the step is done is what lets Vulkan sample the volumes safely. Replacing
    // this with an external semaphore signal is the next step for overlapping sim and render.
    checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
}

const CudaVolume &SmokeSolver::densityVolume() const { return *m_impl->volumes[2]; }
const CudaVolume &SmokeSolver::velocityVolume() const { return *m_impl->volumes[0]; }
const CudaVolume &SmokeSolver::pressureVolume() const { return *m_impl->volumes[4]; }
const CudaVolume &SmokeSolver::divergenceVolume() const { return *m_impl->volumes[6]; }

std::string SmokeSolver::densityResourceName() const { return m_config.namePrefix + "_density"; }
std::string SmokeSolver::velocityResourceName() const { return m_config.namePrefix + "_velocity"; }
std::string SmokeSolver::pressureResourceName() const { return m_config.namePrefix + "_pressure"; }
std::string SmokeSolver::divergenceResourceName() const { return m_config.namePrefix + "_divergence"; }

} // namespace lr::eulerian
