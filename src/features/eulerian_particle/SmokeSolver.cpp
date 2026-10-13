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

// The one-cell boundary the solver enforces has to leave an interior to simulate.
constexpr uint32_t kMinCellsPerAxis = 3;

void validateExtent(VkExtent3D extent, const char *what)
{
    if (extent.width < kMinCellsPerAxis || extent.height < kMinCellsPerAxis || extent.depth < kMinCellsPerAxis)
        throw std::invalid_argument(std::string("SmokeSolver: ") + what +
                                    " must be at least 3 cells on every axis so the one-cell "
                                    "boundary leaves an interior to simulate");
}

} // namespace

struct SmokeSolver::Impl
{
    // Index order matches kVolumes.
    std::vector<std::unique_ptr<CudaVolume>> volumes;
    // Every image in the handshake: the volumes above plus the obstacle mask when there is one.
    // Cached so the handshake can name them all in one registry call per step.
    std::vector<std::string>                 volumeNames;

    // The obstacle mask, imported read-only. Null when no mask was configured; the kernels take a
    // zero surface to mean "no obstacles", so nothing else has to branch on it.
    std::unique_ptr<CudaVolume>              obstacle;
    ResourceRegistry                        *resources = nullptr;
    cudaStream_t                             stream    = nullptr;
    // The sub-region the kernels are told to simulate. Smaller than the allocated texture whenever
    // the grid has been resized below its allocation; reset() clears the allocation in full.
    cudaExtent                               extent{};
    int                                      device = 0;

    // Vulkan signals `toCuda` once the volumes are in GENERAL and prior graphics work is done;
    // CUDA signals `toVulkan` once the step's kernels are done. Both are binary, so exactly one
    // signal may be outstanding on each and the two sides must alternate.
    std::unique_ptr<CudaInteropSemaphore> toCuda;
    std::unique_ptr<CudaInteropSemaphore> toVulkan;

    // True between step() and endStep(): CUDA holds the volumes and `toVulkan` has a signal
    // pending (or will have once the stream drains).
    bool pending = false;

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
        // Zero when no mask was imported, which every kernel reads as fully fluid.
        f.obstacle     = obstacle ? obstacle->surface() : 0;
        return f;
    }

    // Hands the volumes to CUDA: Vulkan transitions them and signals, and the stream waits on that
    // signal on the device. Neither call blocks the host.
    void beginExternalWrite()
    {
        resources->releaseImagesToExternalWrite(volumeNames, toCuda->handle());
        toCuda->waitInCuda(stream);
    }

    // Hands them back: CUDA signals after the work already on the stream, and a Vulkan submission
    // waits on that signal so later Vulkan work sees the writes.
    void endExternalWrite()
    {
        toVulkan->signalInCuda(stream);
        resources->acquireImagesFromExternalWrite(volumeNames, toVulkan->handle());
    }

    ~Impl()
    {
        if (stream == nullptr)
            return;

        (void)cudaSetDevice(device);

        // A step left open has a CUDA-side signal of `toVulkan` that nothing waits on, and the
        // semaphores are about to be destroyed. Close the handshake the way endStep() would, then
        // drain the stream so no queued wait or signal outlives its semaphore. Errors are ignored:
        // a destructor cannot report them, and the device wait below is the real safety net.
        if (pending)
        {
            (void)closeHandshakeNoThrow();
            pending = false;
        }
        (void)cudaStreamSynchronize(stream);
        (void)cudaStreamDestroy(stream);
    }

    // endExternalWrite() without the throwing wrapper, for the destructor. Consuming the signal on
    // the Vulkan side matters: leaving a binary semaphore signalled while a queued wait on the
    // other one is still outstanding is what would wedge the queue.
    bool closeHandshakeNoThrow()
    {
        try
        {
            endExternalWrite();
            return true;
        } catch (const std::exception &)
        {
            return false;
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
    validateExtent(m_config.extent, "the allocated grid");
    if (m_config.activeExtent.width == 0 && m_config.activeExtent.height == 0 && m_config.activeExtent.depth == 0)
        m_config.activeExtent = m_config.extent;
    validateExtent(m_config.activeExtent, "the active grid");
    if (m_config.activeExtent.width > m_config.extent.width ||
        m_config.activeExtent.height > m_config.extent.height ||
        m_config.activeExtent.depth > m_config.extent.depth)
        throw std::invalid_argument("SmokeSolver: the active grid cannot exceed the allocated extent");
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

    m_impl->resources = &resources;
    m_impl->volumes.reserve(std::size(kVolumes));
    m_impl->volumeNames.reserve(std::size(kVolumes));
    for (const VolumeSpec &spec : kVolumes)
    {
        const std::string name = m_config.namePrefix + spec.suffix;
        m_impl->volumes.push_back(std::make_unique<CudaVolume>(resources, context, name));
        m_impl->volumeNames.push_back(name);
    }

    if (!m_config.obstacleMaskName.empty())
    {
        // Imported but never registered or written here: the voxelizer owns the contents. It joins
        // the handshake name list all the same, which is what orders the Vulkan-side rasterization
        // before the kernels that read it — and, on the way back, the kernels before the next
        // frame's rasterization.
        m_impl->obstacle = std::make_unique<CudaVolume>(resources, context, m_config.obstacleMaskName);
        const cudaExtent maskExtent = m_impl->obstacle->extent();
        if (maskExtent.width != m_config.extent.width || maskExtent.height != m_config.extent.height ||
            maskExtent.depth != m_config.extent.depth)
            throw std::invalid_argument("SmokeSolver: the obstacle mask '" + m_config.obstacleMaskName +
                                        "' must be allocated at the same extent as the simulation volumes, because "
                                        "the kernels index both with the same cell coordinates");
        m_impl->volumeNames.push_back(m_config.obstacleMaskName);
    }

    m_impl->extent = cudaExtent{m_config.activeExtent.width, m_config.activeExtent.height,
                                m_config.activeExtent.depth};
    m_impl->device = m_impl->volumes.front()->device();
    checkCuda(cudaSetDevice(m_impl->device), "cudaSetDevice");
    checkCuda(cudaStreamCreate(&m_impl->stream), "cudaStreamCreate");

    m_impl->toCuda   = std::make_unique<CudaInteropSemaphore>(context, m_impl->device);
    m_impl->toVulkan = std::make_unique<CudaInteropSemaphore>(context, m_impl->device);

    reset();
}

SmokeSolver::~SmokeSolver()                              = default;
SmokeSolver::SmokeSolver(SmokeSolver &&) noexcept        = default;
SmokeSolver &SmokeSolver::operator=(SmokeSolver &&) noexcept = default;

void SmokeSolver::reset()
{
    if (!m_impl)
        throw std::logic_error("SmokeSolver: cannot use a moved-from solver");

    // Close any step still in flight before overwriting the fields it is writing.
    waitForStep();

    checkCuda(cudaSetDevice(m_impl->device), "cudaSetDevice");
    m_impl->beginExternalWrite();
    m_impl->pending = true;

    // Imported Vulkan memory starts undefined, so zero every field before accumulating into it.
    // This clears the whole allocation rather than just the active region: cells outside it are
    // never written again, and the renderer must not sample stale density from a previous size.
    //
    // Only the owned fields: the obstacle mask is deliberately absent from this loop, since the
    // voxelizer rewrites it from the live scene every frame and clearing it here would drop one
    // frame of collisions on every reset.
    for (std::unique_ptr<CudaVolume> &volume : m_impl->volumes)
        checkCuda(launchClearVolume(volume->surface(), volume->extent(), volume->channels(), m_impl->stream),
                  "launchClearVolume");

    // reset() is a setup call rather than a per-frame one, so it is allowed to block: callers
    // expect the fields to read as zero the moment it returns.
    waitForStep();
}

void SmokeSolver::step(float dt)
{
    if (!m_impl)
        throw std::logic_error("SmokeSolver: cannot use a moved-from solver");

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
    // The handshake semaphores are binary: a second step before endStep() would try to re-signal
    // an already-signalled semaphore, which the driver is free to treat as undefined behaviour.
    // Failing loudly beats a deadlock or a corrupted field.
    if (m_impl->pending)
        throw std::logic_error("SmokeSolver: a step is already in flight; call endStep() or waitForStep() first");

    checkCuda(cudaSetDevice(m_impl->device), "cudaSetDevice");

    // Transition the volumes and have the stream wait for that transition on the device. From here
    // on the step is in flight: pending is set before any launch so a throw mid-step still leaves
    // endStep() responsible for consuming the signals.
    m_impl->beginExternalWrite();
    m_impl->pending = true;

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
    checkCuda(launchComputeDivergence(fields, extent, m_config.step.openTop, stream), "launchComputeDivergence");

    // 5. Jacobi pressure solve. Each sweep reads `pressure` and writes `pressureNext`, so swapping
    //    after every sweep leaves the result in `pressure` whenever the iteration count is even.
    //    Pressure is not re-zeroed: starting from the previous step's solution converges faster.
    for (int iteration = 0; iteration < m_config.jacobiIterations; ++iteration)
    {
        checkCuda(launchPressureJacobi(fields, extent, m_config.step.openTop, stream), "launchPressureJacobi");
        std::swap(fields.pressure, fields.pressureNext);
    }

    // 6. Pressure projection — subtract the pressure gradient to make the velocity divergence-free.
    checkCuda(launchProjectVelocity(fields, extent, m_config.step.openTop, stream), "launchProjectVelocity");

    // 7. Density advection along the now-incompressible velocity field.
    checkCuda(launchAdvectDensity(fields, extent, m_config.step, stream), "launchAdvectDensity");
    std::swap(fields.density, fields.densityNext);

    // 8. Boundary conditions, applied last so the border is consistent with what the renderer
    //    samples and with what the next step's stencils read.
    checkCuda(launchApplyBoundaries(fields, extent, m_config.step.openTop, stream), "launchApplyBoundaries");

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

    // No host wait: the stream signals `toVulkan` when the kernels above retire, and endStep()
    // submits the Vulkan wait that orders them before anything that samples the volumes.
}

void SmokeSolver::endStep()
{
    if (!m_impl)
        throw std::logic_error("SmokeSolver: cannot use a moved-from solver");
    if (!m_impl->pending)
        return;

    // Clear `pending` first: if the signal or the Vulkan submission throws, the semaphore state is
    // already indeterminate and retrying the same handoff cannot fix it, so do not leave the
    // solver wedged in a state where every later step() refuses to run.
    m_impl->pending = false;
    checkCuda(cudaSetDevice(m_impl->device), "cudaSetDevice");
    m_impl->endExternalWrite();
}

void SmokeSolver::waitForStep()
{
    if (!m_impl)
        throw std::logic_error("SmokeSolver: cannot use a moved-from solver");

    const bool wasPending = m_impl->pending;
    endStep();
    if (!wasPending)
        return;

    checkCuda(cudaSetDevice(m_impl->device), "cudaSetDevice");
    checkCuda(cudaStreamSynchronize(m_impl->stream), "cudaStreamSynchronize");
}

void SmokeSolver::setActiveExtent(VkExtent3D extent)
{
    if (!m_impl)
        throw std::logic_error("SmokeSolver: cannot use a moved-from solver");

    validateExtent(extent, "the active grid");
    if (extent.width > m_config.extent.width || extent.height > m_config.extent.height ||
        extent.depth > m_config.extent.depth)
        throw std::invalid_argument("SmokeSolver: the active grid cannot exceed the allocated extent");

    if (extent.width == m_config.activeExtent.width && extent.height == m_config.activeExtent.height &&
        extent.depth == m_config.activeExtent.depth)
        return;

    // A step in flight is writing the old region; let it finish before the kernels are retargeted.
    waitForStep();

    m_config.activeExtent = extent;
    m_impl->extent        = cudaExtent{extent.width, extent.height, extent.depth};
}

VkExtent3D SmokeSolver::activeExtent() const
{
    return m_config.activeExtent;
}

bool SmokeSolver::stepPending() const
{
    return m_impl && m_impl->pending;
}

const CudaVolume *SmokeSolver::obstacleVolume() const { return m_impl ? m_impl->obstacle.get() : nullptr; }

const CudaVolume &SmokeSolver::densityVolume() const { return *m_impl->volumes[2]; }
const CudaVolume &SmokeSolver::velocityVolume() const { return *m_impl->volumes[0]; }
const CudaVolume &SmokeSolver::pressureVolume() const { return *m_impl->volumes[4]; }
const CudaVolume &SmokeSolver::divergenceVolume() const { return *m_impl->volumes[6]; }

std::string SmokeSolver::densityResourceName() const { return m_config.namePrefix + "_density"; }
std::string SmokeSolver::velocityResourceName() const { return m_config.namePrefix + "_velocity"; }
std::string SmokeSolver::pressureResourceName() const { return m_config.namePrefix + "_pressure"; }
std::string SmokeSolver::divergenceResourceName() const { return m_config.namePrefix + "_divergence"; }

} // namespace lr::eulerian
