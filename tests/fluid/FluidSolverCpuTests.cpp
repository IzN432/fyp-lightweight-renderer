#include "features/fluid/FluidSolverCpu.hpp"

#include <glm/glm.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>

namespace
{

bool isFinite(const glm::vec3 &value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

void testEmptyPositionList()
{
    lr::FluidSolverCpu solver;
    solver.setParticles({glm::vec3(1.0f)});
    solver.setParticles({});
    solver.step(1.0f / 60.0f);

    assert(solver.particleCount() == 0);
    assert(solver.positions().empty());
    assert(solver.velocities().empty());
    assert(solver.densities().empty());
}

// Kernels: density falls off to exactly zero at the support radius, and the gradient points from
// neighbour to particle so the correction separates them.
void testKernels()
{
    constexpr float h = 0.1f;

    const float atCentre = lr::FluidSolverCpu::restDensityForSpacing(h * 4.0f, h, 1.0f);
    assert(atCentre > 0.0f); // Only the particle itself is in range at this spacing.

    lr::FluidSolverCpu::Config config;
    config.smoothingRadius = h;
    // Keep gravity and boundary collision handling out of this density-kernel test.
    config.boundsMin       = glm::vec3(-10.0f);
    config.boundsMax       = glm::vec3(10.0f);
    config.gravity         = glm::vec3(0.0f);
    lr::FluidSolverCpu solver(config);

    // A particle's own poly6(0) term is most of its density, so comparing totals puts the
    // neighbour's contribution down in float noise. Measure it as a difference from the isolated
    // density instead, which is what "inside the radius interacts" actually means.
    solver.setParticles({glm::vec3(0.0f)});
    solver.step(1.0f / 120.0f);
    const float aloneDensity = solver.densities()[0];

    // Two particles closer than h must interact; beyond h they must not.
    solver.setParticles({glm::vec3(0.0f), glm::vec3(h * 0.5f, 0.0f, 0.0f)});
    solver.step(1.0f / 120.0f);
    const float nearDensity = solver.densities()[0];

    solver.setParticles({glm::vec3(0.0f), glm::vec3(h * 2.0f, 0.0f, 0.0f)});
    solver.step(1.0f / 120.0f);
    const float farDensity = solver.densities()[0];

    assert(nearDensity > aloneDensity); // Neighbour inside the radius contributes.
    assert(farDensity == aloneDensity); // Neighbour at 2h contributes exactly nothing.
}

// Gravity, integration and the boundary. No fluid behaviour involved: this is the harness check.
void testFreeFallAndFloor()
{
    lr::FluidSolverCpu::Config config;
    config.solverIterations = 0; // Isolate integration from the constraint solve.
    config.boundsMin        = glm::vec3(-1.0f, 0.0f, -1.0f);
    config.boundsMax        = glm::vec3(1.0f, 2.0f, 1.0f);

    lr::FluidSolverCpu solver(config);
    solver.setParticles({glm::vec3(0.0f, 1.0f, 0.0f)});

    solver.step(1.0f / 60.0f);
    assert(solver.positions()[0].y < 1.0f); // Fell.

    for (int i = 0; i < 300; ++i)
    {
        solver.step(1.0f / 60.0f);
    }
    const glm::vec3 resting = solver.positions()[0];
    assert(isFinite(resting));
    assert(resting.y >= config.boundsMin.y);       // Did not tunnel through the floor.
    assert(resting.y < config.boundsMin.y + 1e-3f); // Came to rest on it.
}

// The actual fluid assertion: a settled block holds near rest density without blowing up.
void testBlockSettlesNearRestDensity()
{
    constexpr float spacing = 0.025f;
    constexpr float h       = 0.05f;

    lr::FluidSolverCpu::Config config;
    config.smoothingRadius  = h;
    config.particleMass     = 1.0f;
    // Derived from the lattice rather than chosen, so the block starts at equilibrium instead of
    // immediately inflating or collapsing.
    config.restDensity      = lr::FluidSolverCpu::restDensityForSpacing(spacing, h, 1.0f);
    config.solverIterations = 4;
    config.relaxation       = config.restDensity * 0.01f;
    // The container is kept narrow and tall, and filled deep enough that the settled fluid stays
    // more than a smoothing radius deep. A wider box lets the same volume spread into a puddle
    // shallower than h, and then no particle is a full radius clear of the floor, so the interior
    // sample below is empty and there is nothing left to measure incompressibility on.
    config.boundsMin        = glm::vec3(-0.15f, 0.0f, -0.15f);
    config.boundsMax        = glm::vec3(0.15f, 0.6f, 0.15f);

    lr::FluidSolverCpu solver(config);
    auto particles = lr::FluidSolverCpu::makeParticleBlock(glm::vec3(-0.1f, 0.05f, -0.1f),
                                                           glm::vec3(0.1f, 0.45f, 0.1f), spacing);
    assert(particles.size() > 200);
    solver.setParticles(particles);

    for (int i = 0; i < 120; ++i)
    {
        solver.step(1.0f / 60.0f);
    }

    // Nothing diverged, and every particle stayed inside the container.
    float       peakDensity   = 0.0f;
    std::size_t interiorCount = 0;
    for (std::size_t i = 0; i < solver.particleCount(); ++i)
    {
        assert(isFinite(solver.positions()[i]));
        assert(isFinite(solver.velocities()[i]));
        assert(std::isfinite(solver.densities()[i]));

        const glm::vec3 &position = solver.positions()[i];
        for (int axis = 0; axis < 3; ++axis)
        {
            assert(position[axis] >= config.boundsMin[axis] - 1e-4f);
            assert(position[axis] <= config.boundsMax[axis] + 1e-4f);
        }
        // densities() includes what the walls contribute, which for a particle resting on the
        // floor is legitimately half a rest density on its own. Measuring incompressibility has to
        // exclude that or a perfectly settled pile reads as compressed; only interior particles,
        // a full support radius clear of every wall, carry fluid density alone.
        bool interior = true;
        for (int axis = 0; axis < 3; ++axis)
        {
            interior = interior && position[axis] > config.boundsMin[axis] + h &&
                       position[axis] < config.boundsMax[axis] - h;
        }
        if (interior)
        {
            peakDensity = std::max(peakDensity, solver.densities()[i]);
            ++interiorCount;
        }
    }

    // The block is dropped into the middle of a wider container, so most of it settles clear of
    // the walls. If that stops being true this test is measuring nothing and should be known to.
    assert(interiorCount > 50);

    // Incompressibility is the point of the solve: settled fluid must not be significantly
    // overcompressed. PBF is only weakly incompressible, so this is a loose bound by design.
    assert(peakDensity < config.restDensity * 1.6f);

    std::printf("settled %zu particles (%zu interior), rest density %.1f, peak interior density "
                "%.1f\n",
                solver.particleCount(), interiorCount, static_cast<double>(config.restDensity),
                static_cast<double>(peakDensity));
}

// Compression has to be resisted: squeezed particles must push back apart.
void testCompressionIsResolved()
{
    constexpr float h = 0.05f;

    lr::FluidSolverCpu::Config config;
    config.smoothingRadius  = h;
    config.restDensity      = lr::FluidSolverCpu::restDensityForSpacing(h * 0.5f, h, 1.0f);
    config.gravity          = glm::vec3(0.0f); // Isolate the constraint solve.
    config.solverIterations = 10;
    config.relaxation       = config.restDensity * 0.01f;
    config.boundsMin        = glm::vec3(-1.0f);
    config.boundsMax        = glm::vec3(1.0f);

    lr::FluidSolverCpu solver(config);
    // A tight clump, far denser than rest.
    auto particles = lr::FluidSolverCpu::makeParticleBlock(glm::vec3(-h * 0.25f),
                                                           glm::vec3(h * 0.25f), h * 0.125f);
    assert(particles.size() > 20);

    float initialExtent = 0.0f;
    for (const glm::vec3 &position : particles)
    {
        initialExtent = std::max(initialExtent, glm::length(position));
    }

    solver.setParticles(particles);
    for (int i = 0; i < 20; ++i)
    {
        solver.step(1.0f / 60.0f);
    }

    float finalExtent = 0.0f;
    for (std::size_t i = 0; i < solver.particleCount(); ++i)
    {
        assert(isFinite(solver.positions()[i]));
        finalExtent = std::max(finalExtent, glm::length(solver.positions()[i]));
    }

    assert(finalExtent > initialExtent); // The clump expanded rather than staying compressed.
}

// Regression: a dropped column in contact with all four walls must not climb them and must come to
// rest. Boundary clamping must not inject upward energy at face or corner contacts, while viscosity
// must dissipate the energy that positional corrections convert back into velocity.
void testFluidSettlesWithoutClimbingWalls()
{
    constexpr float spacing = 0.025f;
    constexpr float h       = 0.05f;

    lr::FluidSolverCpu::Config config;
    config.smoothingRadius  = h;
    config.restDensity      = lr::FluidSolverCpu::restDensityForSpacing(spacing, h, 1.0f);
    // Relaxation an order of magnitude above the settling test's, and viscosity well above the
    // default. At restDensity * 0.01 the constraint solve is stiff enough to overshoot and keep
    // feeding energy back as velocity faster than the default viscosity of 0.05 can absorb it, and
    // the column then sloshes indefinitely instead of settling -- peak speed sits near the
    // maxSpeedInRadiiPerSecond clamp out to thousands of steps. These values converge by step 400
    // and stay converged.
    config.relaxation       = config.restDensity * 0.1f;
    config.viscosity        = 1.0f;
    config.solverIterations = 20;
    config.boundsMin        = glm::vec3(-0.1f, 0.0f, -0.1f);
    config.boundsMax        = glm::vec3(0.1f, 0.5f, 0.1f);

    // A column filling the container's footprint, so the fluid is in contact with all four walls
    // from the first step. This is the configuration that used to creep up the corners.
    lr::FluidSolverCpu solver(config);
    auto particles = lr::FluidSolverCpu::makeParticleBlock(glm::vec3(-0.09f, 0.01f, -0.09f),
                                                           glm::vec3(0.09f, 0.2f, 0.09f), spacing);
    assert(particles.size() > 200);
    solver.setParticles(particles);

    float initialTop = 0.0f;
    for (const glm::vec3 &position : particles)
    {
        initialTop = std::max(initialTop, position.y);
    }

    for (int i = 0; i < 400; ++i)
    {
        solver.step(1.0f / 60.0f);
    }

    float topHeight = 0.0f;
    float peakSpeed = 0.0f;
    for (std::size_t i = 0; i < solver.particleCount(); ++i)
    {
        assert(isFinite(solver.positions()[i]));
        topHeight = std::max(topHeight, solver.positions()[i].y);
        peakSpeed = std::max(peakSpeed, glm::length(solver.velocities()[i]));
    }

    // Settling under gravity can only lower the surface. Climbing shows up as the top rising above
    // where it started, and in the old code it would march steadily towards the ceiling.
    assert(topHeight <= initialTop + 2.0f * spacing);

    // And it has to actually stop. A sloshing column keeps a few particles moving at a good
    // fraction of free-fall speed indefinitely; a settled one is near zero.
    assert(peakSpeed < 0.1f);

    std::printf("column settled: top %.4f (started %.4f), peak speed %.4f\n",
                static_cast<double>(topHeight), static_cast<double>(initialTop),
                static_cast<double>(peakSpeed));
}

} // namespace

int main()
{
    testEmptyPositionList();
    testKernels();
    testFreeFallAndFloor();
    testBlockSettlesNearRestDensity();
    testCompressionIsResolved();
    testFluidSettlesWithoutClimbingWalls();
    return 0;
}
