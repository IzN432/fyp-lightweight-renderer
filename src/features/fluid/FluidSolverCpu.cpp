#include "features/fluid/FluidSolverCpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace lr
{
namespace
{
constexpr float kPi = 3.14159265358979323846f;
} // namespace

FluidSolverCpu::FluidSolverCpu() = default;

FluidSolverCpu::FluidSolverCpu(Config config) : m_config(config) {}

void FluidSolverCpu::setParticles(std::vector<glm::vec3> positions)
{
    m_positions = std::move(positions);
    m_velocities.assign(m_positions.size(), glm::vec3(0.0f));
    m_predicted.resize(m_positions.size());
    m_corrections.resize(m_positions.size());
    m_densities.assign(m_positions.size(), 0.0f);
    m_lambdas.assign(m_positions.size(), 0.0f);
    m_neighbourOffsets.assign(m_positions.size() + 1, 0u);
    m_neighbours.clear();
    m_particleBuckets.assign(m_positions.size(), 0u);
    m_cellParticles.assign(m_positions.size(), 0u);
    m_cellStarts.clear();
}

void FluidSolverCpu::clear()
{
    setParticles({});
}

float FluidSolverCpu::poly6(float squaredDistance, float smoothingRadius)
{
    const float h2 = smoothingRadius * smoothingRadius;
    if (squaredDistance >= h2)
    {
        return 0.0f;
    }
    const float h9        = std::pow(smoothingRadius, 9.0f);
    const float deficit   = h2 - squaredDistance;
    const float normaliser = 315.0f / (64.0f * kPi * h9);
    return normaliser * deficit * deficit * deficit;
}

glm::vec3 FluidSolverCpu::spikyGradient(const glm::vec3 &offset, float smoothingRadius)
{
    const float distance = glm::length(offset);
    // Coincident particles have no defined direction to separate along. Returning zero leaves them
    // to be pushed apart by their other neighbours.
    if (distance <= 1e-6f || distance >= smoothingRadius)
    {
        return glm::vec3(0.0f);
    }
    const float h6        = std::pow(smoothingRadius, 6.0f);
    const float deficit   = smoothingRadius - distance;
    const float magnitude = -45.0f / (kPi * h6) * deficit * deficit;
    return magnitude * (offset / distance);
}

glm::ivec3 FluidSolverCpu::cellForPosition(const glm::vec3 &position) const
{
    const glm::vec3 scaled = position / m_config.smoothingRadius;
    return glm::ivec3(static_cast<int>(std::floor(scaled.x)), static_cast<int>(std::floor(scaled.y)),
                      static_cast<int>(std::floor(scaled.z)));
}

std::size_t FluidSolverCpu::bucketForCell(const glm::ivec3 &cell) const
{
    // Large primes, one per axis. Three rather than two: with a shared multiplier on any pair of
    // axes, cells that are permutations of each other collapse onto the same hash, and a fluid
    // sitting on a diagonal would pile every such pair into one bucket.
    constexpr std::uint32_t kPrimeX = 73856093u;
    constexpr std::uint32_t kPrimeY = 19349663u;
    constexpr std::uint32_t kPrimeZ = 83492791u;

    // Cast through unsigned so negative cells wrap defined rather than overflowing a signed int.
    const std::uint32_t hash = (static_cast<std::uint32_t>(cell.x) * kPrimeX) ^
                               (static_cast<std::uint32_t>(cell.y) * kPrimeY) ^
                               (static_cast<std::uint32_t>(cell.z) * kPrimeZ);
    return static_cast<std::size_t>(hash) & m_bucketMask;
}

void FluidSolverCpu::buildSpatialHash()
{
    const std::size_t count = m_predicted.size();

    // Two buckets per particle keeps the load factor low enough that a bucket holds about one
    // cell's worth of particles, which is the point of the grid.
    const std::size_t requested = m_config.hashBuckets != 0 ? m_config.hashBuckets : count * 2;

    // Round up to a power of two so wrapping is a mask, and keep at least one bucket so an empty
    // solver still has a valid table.
    std::size_t buckets = 1;
    while (buckets < requested)
    {
        buckets <<= 1;
    }
    m_bucketMask = buckets - 1;

    // Counting sort over buckets. Pass one tallies how many particles each bucket holds.
    m_cellStarts.assign(buckets + 1, 0u);
    m_particleBuckets.resize(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        const std::size_t bucket = bucketForCell(cellForPosition(m_predicted[i]));
        m_particleBuckets[i]     = static_cast<unsigned int>(bucket);
        ++m_cellStarts[bucket + 1];
    }

    // Prefix sum turns the tallies into the start offset of each bucket.
    for (std::size_t b = 0; b < buckets; ++b)
    {
        m_cellStarts[b + 1] += m_cellStarts[b];
    }

    // Pass two scatters each particle into its bucket's span. Walking a cursor copy leaves
    // m_cellStarts itself intact as the offsets the lookup needs.
    std::vector<unsigned int> cursor(m_cellStarts.begin(), m_cellStarts.end() - 1);
    m_cellParticles.resize(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        m_cellParticles[cursor[m_particleBuckets[i]]++] = static_cast<unsigned int>(i);
    }
}

void FluidSolverCpu::findNeighbours()
{
    const std::size_t count = m_predicted.size();
    const float       h2    = m_config.smoothingRadius * m_config.smoothingRadius;

    buildSpatialHash();

    m_neighbours.clear();
    m_neighbourOffsets.assign(count + 1, 0u);

    // Cells are a full smoothing radius across, so every particle within range lies in this
    // particle's own cell or one of the 26 around it. The result is identical to the brute force
    // search this replaced -- same sets, same order within each cell span -- which is what lets the
    // density assertions in the tests stay as they were.
    for (std::size_t i = 0; i < count; ++i)
    {
        m_neighbourOffsets[i]  = static_cast<unsigned int>(m_neighbours.size());
        const glm::ivec3 centre = cellForPosition(m_predicted[i]);

        for (int dz = -1; dz <= 1; ++dz)
        {
            for (int dy = -1; dy <= 1; ++dy)
            {
                for (int dx = -1; dx <= 1; ++dx)
                {
                    const glm::ivec3  cell   = centre + glm::ivec3(dx, dy, dz);
                    const std::size_t bucket = bucketForCell(cell);

                    for (unsigned int s = m_cellStarts[bucket]; s < m_cellStarts[bucket + 1]; ++s)
                    {
                        const unsigned int j = m_cellParticles[s];
                        if (j == i)
                        {
                            continue;
                        }
                        // A bucket can hold particles from unrelated cells, either through a hash
                        // collision or because a neighbouring cell wrapped onto the same bucket, so
                        // the distance test below is what actually decides membership. The same
                        // bucket can also be visited twice by two of the 27 offsets; checking the
                        // cell the particle really sits in keeps it from being listed twice.
                        if (cellForPosition(m_predicted[j]) != cell)
                        {
                            continue;
                        }
                        const glm::vec3 offset = m_predicted[i] - m_predicted[j];
                        if (glm::dot(offset, offset) < h2)
                        {
                            m_neighbours.push_back(j);
                        }
                    }
                }
            }
        }
    }
    m_neighbourOffsets[count] = static_cast<unsigned int>(m_neighbours.size());
}

void FluidSolverCpu::computeDensitiesAndLambdas()
{
    const std::size_t count = m_predicted.size();
    const float       h     = m_config.smoothingRadius;
    const float       rho0  = m_config.restDensity;

    for (std::size_t i = 0; i < count; ++i)
    {
        // A particle contributes to its own density, so start from the r = 0 kernel value.
        float density = m_config.particleMass * poly6(0.0f, h);

        // Gradient of particle i's constraint with respect to i itself, accumulated as the negated
        // sum of its neighbour gradients.
        glm::vec3 gradientSelf(0.0f);
        float     squaredGradientSum = 0.0f;

        for (unsigned int n = m_neighbourOffsets[i]; n < m_neighbourOffsets[i + 1]; ++n)
        {
            const std::size_t j      = m_neighbours[n];
            const glm::vec3   offset = m_predicted[i] - m_predicted[j];
            density += m_config.particleMass * poly6(glm::dot(offset, offset), h);

            const glm::vec3 gradient = spikyGradient(offset, h) / rho0;
            gradientSelf += gradient;
            squaredGradientSum += glm::dot(gradient, gradient);
        }
        squaredGradientSum += glm::dot(gradientSelf, gradientSelf);

        m_densities[i] = density;

        const float constraint = density / rho0 - 1.0f;
        // Only compression is resolved. Letting the solve pull particles together as well makes
        // the free surface clump into blobs.
        if (constraint <= 0.0f)
        {
            m_lambdas[i] = 0.0f;
            continue;
        }
        m_lambdas[i] = -constraint / (squaredGradientSum + m_config.relaxation);
    }
}

void FluidSolverCpu::computePositionCorrections()
{
    const std::size_t count = m_predicted.size();
    const float       h     = m_config.smoothingRadius;
    const float       rho0  = m_config.restDensity;

    // Separate pass from the lambda computation on purpose: every correction has to be built from
    // the same iteration's lambdas, or the result depends on particle ordering and stops matching
    // what a parallel implementation produces.
    for (std::size_t i = 0; i < count; ++i)
    {
        glm::vec3 correction(0.0f);
        for (unsigned int n = m_neighbourOffsets[i]; n < m_neighbourOffsets[i + 1]; ++n)
        {
            const std::size_t j      = m_neighbours[n];
            const glm::vec3   offset = m_predicted[i] - m_predicted[j];
            // The symmetric (lambda_i + lambda_j) weight is what makes each pair push itself apart
            // equally and oppositely, conserving momentum.
            correction += (m_lambdas[i] + m_lambdas[j]) * spikyGradient(offset, h);
        }
        m_corrections[i] = correction / rho0;
    }

    for (std::size_t i = 0; i < count; ++i)
    {
        m_predicted[i] += m_corrections[i];
    }
}

void FluidSolverCpu::applyViscosity(float deltaTime)
{
    if (m_config.viscosity <= 0.0f)
    {
        return;
    }

    const std::size_t count = m_velocities.size();
    const float       h     = m_config.smoothingRadius;

    // XSPH. Written into a scratch buffer first so every particle blends against the same
    // pre-viscosity velocity field, for the same ordering-independence reason as the corrections.
    std::vector<glm::vec3> blended(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        glm::vec3 delta(0.0f);
        float     weightSum = 0.0f;
        for (unsigned int n = m_neighbourOffsets[i]; n < m_neighbourOffsets[i + 1]; ++n)
        {
            const std::size_t j      = m_neighbours[n];
            const glm::vec3   offset = m_positions[i] - m_positions[j];
            const float       weight = poly6(glm::dot(offset, offset), h);
            delta += weight * (m_velocities[j] - m_velocities[i]);
            weightSum += weight;
        }
        blended[i] = m_velocities[i];
        if (weightSum > 0.0f)
        {
            blended[i] += m_config.viscosity * delta / weightSum;
        }
    }
    m_velocities = std::move(blended);

    (void)deltaTime;
}

void FluidSolverCpu::resolveBoundaryCollisions()
{
    // Positions only. The velocity update at the end of step() derives velocity from total
    // displacement, so anything written to m_velocities here would be discarded; contact response
    // belongs with the velocity update in step(), after it.
    for (std::size_t i = 0; i < m_predicted.size(); ++i)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            m_predicted[i][axis] = std::clamp(m_predicted[i][axis], m_config.boundsMin[axis],
                                              m_config.boundsMax[axis]);
        }
    }
}

void FluidSolverCpu::step(float deltaTime)
{
    const std::size_t count = m_positions.size();
    if (count == 0 || deltaTime <= 0.0f)
    {
        return;
    }

    for (std::size_t i = 0; i < count; ++i)
    {
        m_velocities[i] += m_config.gravity * deltaTime;
        m_predicted[i] = m_positions[i] + m_velocities[i] * deltaTime;
    }

    // Found once from the predicted positions rather than per iteration: the corrections below are
    // small relative to the smoothing radius, so the neighbour sets barely change within a step.
    findNeighbours();

    for (int iteration = 0; iteration < m_config.solverIterations; ++iteration)
    {
        computeDensitiesAndLambdas();
        computePositionCorrections();
        resolveBoundaryCollisions();
    }

    // Also enforced outside the loop so the container still holds with the solve disabled, which
    // is how the integration path is tested in isolation.
    resolveBoundaryCollisions();

    const float maxSpeed    = m_config.maxSpeedInRadiiPerSecond * m_config.smoothingRadius;
    const float maxSpeedSqr = maxSpeed * maxSpeed;

    for (std::size_t i = 0; i < count; ++i)
    {
        // Velocity follows from the total displacement the solve settled on, which is what makes
        // this position based.
        glm::vec3 velocity = (m_predicted[i] - m_positions[i]) / deltaTime;

        // Contact response, applied here rather than during the solve because the line above
        // overwrites whatever velocity the solve left behind. A particle resting on the floor has
        // been clamped upwards; letting that clamp distance read as upward velocity is what makes a
        // settled pile jitter forever instead of coming to rest.
        for (int axis = 0; axis < 3; ++axis)
        {
            const bool atMin = m_predicted[i][axis] <= m_config.boundsMin[axis] + 1e-6f;
            const bool atMax = m_predicted[i][axis] >= m_config.boundsMax[axis] - 1e-6f;
            if (!atMin && !atMax)
            {
                continue;
            }
            // Only cancel velocity pushing further into the wall; a particle leaving the surface
            // keeps its motion.
            if ((atMin && velocity[axis] < 0.0f) || (atMax && velocity[axis] > 0.0f))
            {
                velocity[axis] *= -m_config.boundsRestitution;
            }
            for (int tangent = 0; tangent < 3; ++tangent)
            {
                if (tangent != axis)
                {
                    velocity[tangent] *= m_config.boundsFriction;
                }
            }
        }

        if (maxSpeedSqr > 0.0f)
        {
            const float speedSqr = glm::dot(velocity, velocity);
            if (speedSqr > maxSpeedSqr)
            {
                velocity *= maxSpeed / std::sqrt(speedSqr);
            }
        }

        m_velocities[i] = velocity;
        m_positions[i]  = m_predicted[i];
    }

    // After positions are committed, so neighbours blend against the velocities they just got.
    applyViscosity(deltaTime);
}

std::vector<glm::vec3> FluidSolverCpu::makeParticleBlock(const glm::vec3 &min, const glm::vec3 &max,
                                                        float spacing)
{
    std::vector<glm::vec3> positions;
    if (spacing <= 0.0f)
    {
        return positions;
    }
    for (float x = min.x; x <= max.x; x += spacing)
    {
        for (float y = min.y; y <= max.y; y += spacing)
        {
            for (float z = min.z; z <= max.z; z += spacing)
            {
                positions.emplace_back(x, y, z);
            }
        }
    }
    return positions;
}

float FluidSolverCpu::restDensityForSpacing(float spacing, float smoothingRadius, float particleMass)
{
    // Sum the kernel over a lattice of this spacing, which is the density a settled block of fluid
    // will actually report. Deriving it rather than guessing keeps the first step from inflating.
    const int reach   = static_cast<int>(std::ceil(smoothingRadius / spacing));
    float     density = 0.0f;
    for (int x = -reach; x <= reach; ++x)
    {
        for (int y = -reach; y <= reach; ++y)
        {
            for (int z = -reach; z <= reach; ++z)
            {
                const glm::vec3 offset = glm::vec3(x, y, z) * spacing;
                density += particleMass * poly6(glm::dot(offset, offset), smoothingRadius);
            }
        }
    }
    return density;
}

} // namespace lr
