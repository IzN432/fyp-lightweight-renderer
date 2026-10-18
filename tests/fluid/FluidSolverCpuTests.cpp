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

float paperPoly6(float squaredDistance, float h)
{
    constexpr float pi = 3.14159265358979323846f;
    const float h2 = h * h;
    if (squaredDistance >= h2) return 0.0f;
    const float deficit = h2 - squaredDistance;
    return 315.0f / (64.0f * pi * std::pow(h, 9.0f)) * deficit * deficit * deficit;
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
    config.solverIterations    = 1;
    config.artificialPressureK = 0.0f;
    config.viscosity           = 0.0f;
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

// Equations (13)-(14): even at zero density error, s_corr is a repulsive correction. This also
// guards against accidentally returning to the old compression-only lambda clamp.
void testArtificialPressureCorrection()
{
    constexpr float h = 1.0f;
    const std::vector<glm::vec3> particles{glm::vec3(0.0f), glm::vec3(0.5f, 0.0f, 0.0f)};

    lr::FluidSolverCpu::Config base;
    base.smoothingRadius          = h;
    base.restDensity              = paperPoly6(0.0f, h) + paperPoly6(0.25f, h);
    base.gravity                  = glm::vec3(0.0f);
    base.relaxation               = 0.0f;
    base.solverIterations         = 1;
    base.viscosity                = 0.0f;
    base.maxDisplacementInRadiiPerStep = 0.0f;
    base.boundsMin                    = glm::vec3(-10.0f);
    base.boundsMax                    = glm::vec3(10.0f);

    auto withoutPressure = base;
    withoutPressure.artificialPressureK = 0.0f;
    lr::FluidSolverCpu baseline(withoutPressure);
    baseline.setParticles(particles);
    baseline.step(1.0f / 60.0f);

    lr::FluidSolverCpu paper(base);
    paper.setParticles(particles);
    paper.step(1.0f / 60.0f);

    const float baselineDistance = glm::distance(baseline.positions()[0], baseline.positions()[1]);
    const float correctedDistance = glm::distance(paper.positions()[0], paper.positions()[1]);
    assert(std::fabs(baselineDistance - 0.5f) < 1e-5f);
    assert(correctedDistance > baselineDistance);
}

// Equation (17): c multiplies the raw kernel-weighted velocity difference. Dividing by the weight
// sum, as the previous implementation did, produces a measurably different answer here.
void testPaperXsphEquation()
{
    constexpr float h  = 1.0f;
    constexpr float c  = 1e-4f;
    constexpr float dt = 1e-4f;

    lr::FluidSolverCpu::Config config;
    config.smoothingRadius          = h;
    config.gravity                  = glm::vec3(0.0f);
    config.solverIterations         = 0;
    config.artificialPressureK      = 0.0f;
    config.vorticityEpsilon         = 0.0f;
    config.viscosity                = c;
    config.maxDisplacementInRadiiPerStep = 0.0f;
    config.boundsMin                    = glm::vec3(-10.0f);
    config.boundsMax                    = glm::vec3(10.0f);

    const std::vector<glm::vec3> particles{glm::vec3(0.0f), glm::vec3(0.5f, 0.0f, 0.0f)};
    const std::vector<glm::vec3> velocities{glm::vec3(0.0f), glm::vec3(1.0f, 0.0f, 0.0f)};

    auto withoutViscosity = config;
    withoutViscosity.viscosity = 0.0f;
    lr::FluidSolverCpu baseline(withoutViscosity);
    baseline.setParticles(particles);
    baseline.setVelocities(velocities);
    baseline.step(dt);

    lr::FluidSolverCpu solver(config);
    solver.setParticles(particles);
    solver.setVelocities(velocities);
    solver.step(dt);

    const float separation = glm::distance(baseline.positions()[0], baseline.positions()[1]);
    const float weight = paperPoly6(separation * separation, h);
    const float delta = baseline.velocities()[1].x - baseline.velocities()[0].x;
    assert(std::fabs(solver.velocities()[0].x - (baseline.velocities()[0].x + c * weight * delta)) < 1e-6f);
    assert(std::fabs(solver.velocities()[1].x - (baseline.velocities()[1].x - c * weight * delta)) < 1e-6f);
}

// Equations (15)-(16): an asymmetric rotating neighbourhood must receive a non-zero confinement
// update, while the otherwise identical disabled solve must retain its input velocity field.
void testVorticityConfinement()
{
    lr::FluidSolverCpu::Config config;
    config.smoothingRadius          = 2.0f;
    config.gravity                  = glm::vec3(0.0f);
    config.solverIterations         = 0;
    config.artificialPressureK      = 0.0f;
    config.viscosity                = 0.0f;
    config.maxDisplacementInRadiiPerStep = 0.0f;
    config.boundsMin                    = glm::vec3(-10.0f);
    config.boundsMax                    = glm::vec3(10.0f);

    const std::vector<glm::vec3> particles{glm::vec3(0.0f), glm::vec3(0.5f, 0.0f, 0.0f),
                                            glm::vec3(0.0f, 0.75f, 0.0f)};
    const std::vector<glm::vec3> velocities{glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f),
                                             glm::vec3(-0.5f, 0.0f, 0.0f)};

    lr::FluidSolverCpu disabled(config);
    disabled.setParticles(particles);
    disabled.setVelocities(velocities);
    disabled.step(1e-4f);

    config.vorticityEpsilon = 1.0f;
    lr::FluidSolverCpu enabled(config);
    enabled.setParticles(particles);
    enabled.setVelocities(velocities);
    enabled.step(1e-4f);

    float totalDifference = 0.0f;
    for (std::size_t i = 0; i < particles.size(); ++i)
    {
        totalDifference += glm::length(enabled.velocities()[i] - disabled.velocities()[i]);
    }
    assert(totalDifference > 1e-6f);
}

// Equations (15)-(16) with a known answer: a counter-clockwise rigid rotation about +z has curl
// 2*Omega*z_hat, so the estimate must point along +z, and confinement must speed up the rotation
// at the rim rather than brake it. Catches the operand order of the curl cross product, which the
// test above cannot, since it only asks for a non-zero change.
void testVorticityMatchesRigidRotation()
{
    constexpr float omegaRate = 2.0f;
    constexpr float spacing   = 0.5f;

    lr::FluidSolverCpu::Config config;
    config.smoothingRadius               = 1.0f;
    config.gravity                       = glm::vec3(0.0f);
    config.solverIterations              = 0;
    config.artificialPressureK           = 0.0f;
    config.viscosity                     = 0.0f;
    config.maxDisplacementInRadiiPerStep = 0.0f;
    config.boundsMin                     = glm::vec3(-10.0f);
    config.boundsMax                     = glm::vec3(10.0f);

    // A 5x5 sheet in the xy plane, so the curl is purely along z.
    std::vector<glm::vec3> particles;
    std::vector<glm::vec3> velocities;
    std::size_t            centre = 0;
    std::size_t            rim    = 0;
    for (int y = -2; y <= 2; ++y)
    {
        for (int x = -2; x <= 2; ++x)
        {
            const glm::vec3 position(static_cast<float>(x) * spacing, static_cast<float>(y) * spacing, 0.0f);
            if (x == 0 && y == 0) centre = particles.size();
            if (x == 2 && y == 0) rim = particles.size();
            particles.push_back(position);
            velocities.push_back(omegaRate * glm::vec3(-position.y, position.x, 0.0f));
        }
    }

    // A tiny step keeps positions, and so the derived velocities, essentially unchanged before the
    // post-process runs.
    constexpr float dt = 1e-4f;

    lr::FluidSolverCpu disabled(config);
    disabled.setParticles(particles);
    disabled.setVelocities(velocities);
    disabled.step(dt);

    config.vorticityEpsilon = 1.0f;
    lr::FluidSolverCpu enabled(config);
    enabled.setParticles(particles);
    enabled.setVelocities(velocities);
    enabled.step(dt);

    const glm::vec3 omega = enabled.vorticities()[centre];
    assert(omega.z > 0.0f);
    assert(std::fabs(omega.x) < 1e-4f * omega.z && std::fabs(omega.y) < 1e-4f * omega.z);

    // The rim particle at (+x, 0) moves along +y. |omega| falls off towards the sheet's edge, so N
    // points inward and N x omega must add to that motion.
    const float boost = enabled.velocities()[rim].y - disabled.velocities()[rim].y;
    assert(boost > 0.0f);
}

// Retaining particle mass in equations (2), (7) and the PBD inverse-mass denominator must not
// change an equal-mass fluid when both particle mass and rest density are scaled together.
void testMassConsistentConstraintGradient()
{
    constexpr float h = 1.0f;
    const float unitRestDensity = paperPoly6(0.0f, h) + paperPoly6(0.25f, h);

    lr::FluidSolverCpu::Config unit;
    unit.smoothingRadius          = h;
    unit.restDensity              = unitRestDensity;
    unit.gravity                  = glm::vec3(0.0f);
    unit.relaxation               = 0.0f;
    unit.solverIterations         = 1;
    unit.artificialPressureK      = 0.0f;
    unit.viscosity                = 0.0f;
    unit.maxDisplacementInRadiiPerStep = 0.0f;
    unit.boundsMin                    = glm::vec3(-10.0f);
    unit.boundsMax                    = glm::vec3(10.0f);

    const std::vector<glm::vec3> compressed{glm::vec3(0.0f), glm::vec3(0.25f, 0.0f, 0.0f)};
    lr::FluidSolverCpu unitMass(unit);
    unitMass.setParticles(compressed);
    unitMass.step(1.0f / 60.0f);

    auto doubled = unit;
    doubled.particleMass = 2.0f;
    doubled.restDensity  = 2.0f * unitRestDensity;
    lr::FluidSolverCpu doubleMass(doubled);
    doubleMass.setParticles(compressed);
    doubleMass.step(1.0f / 60.0f);

    for (std::size_t i = 0; i < compressed.size(); ++i)
    {
        assert(glm::distance(unitMass.positions()[i], doubleMass.positions()[i]) < 1e-5f);
    }
}

// Regression for the visible failure that prompted the scale audit: the current 2 x 5 x 2 demo
// must remain a compact block after its first frame rather than snapping particles to the walls.
void testPaperTermsStayBoundedAtDemoScale()
{
    constexpr float spacing = 0.05f;
    constexpr float h       = 0.1f;

    lr::FluidSolverCpu::Config config;
    config.smoothingRadius          = h;
    config.particleMass             = 1.0f;
    config.restDensity              = lr::FluidSolverCpu::restDensityForSpacing(spacing, h, 1.0f);
    config.relaxation               = config.restDensity * 0.01f;
    config.solverIterations         = 8;
    config.artificialPressureK      = 0.001f * h * h;
    config.viscosity                = 0.01f * config.particleMass / config.restDensity;
    config.boundsMin                = glm::vec3(-0.5f, 0.0f, -0.5f);
    config.boundsMax                = glm::vec3(0.5f, 1.0f, 0.5f);

    const glm::vec3 first = config.boundsMin + glm::vec3(spacing);
    const glm::vec3 last  = first + glm::vec3(1.0f, 4.0f, 1.0f) * spacing;
    const std::vector<glm::vec3> initial = lr::FluidSolverCpu::makeParticleBlock(first, last, spacing);
    assert(initial.size() == 20);

    lr::FluidSolverCpu solver(config);
    solver.setParticles(initial);
    solver.step(1.0f / 60.0f);

    float largestDisplacement = 0.0f;
    for (std::size_t i = 0; i < initial.size(); ++i)
    {
        assert(isFinite(solver.positions()[i]));
        largestDisplacement = std::max(largestDisplacement, glm::distance(initial[i], solver.positions()[i]));
    }
    assert(largestDisplacement < spacing);
}

// The travel limit is expressed per timestep, so changing dt changes the speed ceiling while the
// maximum visible displacement remains fixed. This prevents fast particles from skipping their
// entire neighbour region in one solve.
void testPerStepDisplacementCap()
{
    constexpr float h                 = 0.2f;
    constexpr float displacementScale = 0.25f;
    constexpr float dt                = 0.1f;
    constexpr float expectedTravel    = h * displacementScale;

    lr::FluidSolverCpu::Config config;
    config.smoothingRadius               = h;
    config.gravity                       = glm::vec3(0.0f);
    config.solverIterations              = 0;
    config.viscosity                     = 0.0f;
    config.maxDisplacementInRadiiPerStep = displacementScale;
    config.boundsMin                     = glm::vec3(-10.0f);
    config.boundsMax                     = glm::vec3(10.0f);

    lr::FluidSolverCpu solver(config);
    solver.setParticles({glm::vec3(0.0f)});
    solver.setVelocities({glm::vec3(100.0f, 0.0f, 0.0f)});
    solver.step(dt);

    assert(std::fabs(solver.positions()[0].x - expectedTravel) < 1e-6f);
    assert(std::fabs(solver.velocities()[0].x - expectedTravel / dt) < 1e-6f);
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
    config.artificialPressureK = 0.001f * h * h;
    config.viscosity = 0.01f * config.particleMass / config.restDensity;
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
    config.artificialPressureK = 0.001f * h * h;
    config.viscosity        = 0.0f;
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
    // Relaxation an order of magnitude above the settling test's, with a strong equation (17)
    // coefficient expressed at this kernel scale. These values converge by step 400 and stay
    // converged.
    config.relaxation       = config.restDensity * 0.1f;
    config.artificialPressureK = 0.001f * h * h;
    config.viscosity        = 1.0f / config.restDensity;
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
    testArtificialPressureCorrection();
    testPaperXsphEquation();
    testVorticityConfinement();
    testVorticityMatchesRigidRotation();
    testMassConsistentConstraintGradient();
    testPaperTermsStayBoundedAtDemoScale();
    testPerStepDisplacementCap();
    testFreeFallAndFloor();
    testBlockSettlesNearRestDensity();
    testCompressionIsResolved();
    testFluidSettlesWithoutClimbingWalls();
    return 0;
}
