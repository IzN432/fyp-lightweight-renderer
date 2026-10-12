#pragma once

#include <glm/glm.hpp>

#include <cstddef>
#include <vector>

namespace lr
{

// Reference CPU implementation of Position Based Fluids (Macklin & Muller 2013).
//
// This is deliberately the simplest correct formulation: boundaries are limited to an axis-aligned
// box. It exists to be the ground truth the GPU compute solver is diffed against, so clarity
// outranks speed here. Neighbour search goes through a spatial hash grid, which is the one place
// speed was allowed to win, because brute force is quadratic and made even a few thousand
// particles unusable; the grid is structured the way the compute solver will want it.
class FluidSolverCpu
{
public:
    struct Config
    {
        // Kernel support radius. Particles further apart than this do not interact.
        float smoothingRadius = 0.1f;
        // Target density. Initial particle spacing has to agree with this or the fluid will
        // expand/collapse on the first step regardless of how correct the solve is; see
        // restDensityForSpacing().
        float restDensity = 1000.0f;
        float particleMass = 1.0f;
        glm::vec3 gravity{0.0f, -9.81f, 0.0f};
        // Constraint force mixing / relaxation. Prevents division by a near-zero gradient sum
        // where a particle has too few neighbours to constrain, i.e. at the free surface.
        float relaxation = 1000.0f;
        // Jacobi iterations per step. More iterations means less compressible.
        int solverIterations = 4;
        // Particles are confined to [boundsMin, boundsMax] and lose normal velocity on contact.
        glm::vec3 boundsMin{-0.5f, 0.0f, -0.5f};
        glm::vec3 boundsMax{0.5f, 1.0f, 0.5f};
        float     boundsRestitution = 0.0f;
        // Tangential velocity kept on contact. Below 1 is wall friction, which is what stops a
        // sheet of particles from sliding along a wall indefinitely.
        float boundsFriction = 0.98f;

        // XSPH viscosity: each particle blends a fraction of its neighbourhood's mean velocity.
        // Without this PBF has no way to dissipate energy and a dropped block never comes to rest,
        // because the positional solve keeps converting its own corrections back into velocity.
        float viscosity = 0.05f;
        // Speed cap applied after the velocity update, as a multiple of smoothingRadius per second.
        // Stops a single bad step from throwing a particle clean out of its neighbourhood.
        float maxSpeedInRadiiPerSecond = 20.0f;

        // Density the walls contribute to a particle within smoothingRadius of them, as a fraction
        // of restDensity at contact, falling to zero at a radius out. A particle beside a wall is
        // missing half its neighbourhood, so without this its density reads low, its constraint
        // goes slack, and the over-dense fluid behind it extrudes it up the wall -- the corners
        // worst of all, where two walls are missing at once.
        float boundsDensity = 1.0f;

        // Buckets in the neighbour-search hash table. The grid itself is unbounded -- cells are
        // only ever reached through the hash -- so this trades memory against how often two
        // distant cells share a bucket and get scanned together. Zero sizes the table from the
        // particle count each step, which is what keeps a large sim from degrading back towards
        // brute force; set it explicitly only to pin the table for a GPU buffer of fixed size.
        // Either way it is rounded up to a power of two internally so the modulo becomes a mask.
        std::size_t hashBuckets = 0;
    };

    FluidSolverCpu();
    explicit FluidSolverCpu(Config config);

    // Replaces all particles. Velocities start at zero.
    void setParticles(std::vector<glm::vec3> positions);
    void clear();

    void step(float deltaTime);

    std::size_t                   particleCount() const { return m_positions.size(); }
    const std::vector<glm::vec3> &positions() const { return m_positions; }
    const std::vector<glm::vec3> &velocities() const { return m_velocities; }

    const Config &config() const { return m_config; }
    void          setConfig(const Config &config) { m_config = config; }

    // Density of particle i as of the last solver iteration. Diagnostic: a converged solve leaves
    // interior particles near restDensity, so this is what tests and the GPU diff assert on.
    // Surface particles read legitimately low (they have no neighbours outside the fluid).
    const std::vector<float> &densities() const { return m_densities; }

    // Fills a lattice spanning [min, max] with the spacing the other helpers assume.
    static std::vector<glm::vec3> makeParticleBlock(const glm::vec3 &min, const glm::vec3 &max,
                                                    float spacing);
    // The rest density a settled lattice of this spacing actually produces. Picking restDensity
    // independently of spacing is the usual reason a first run inflates or implodes.
    static float restDensityForSpacing(float spacing, float smoothingRadius, float particleMass);

private:
    // Poly6, used for density only: its gradient vanishes as r approaches 0, so particles sitting
    // on top of each other would feel no separating force.
    static float poly6(float squaredDistance, float smoothingRadius);
    // Spiky gradient, used for every gradient term for the reason above.
    static glm::vec3 spikyGradient(const glm::vec3 &offset, float smoothingRadius);

    // Cell a position falls in. Negative coordinates floor towards -inf, so the lattice stays
    // uniform across the origin rather than doubling up the cells either side of it.
    glm::ivec3 cellForPosition(const glm::vec3 &position) const;
    // Bucket a cell hashes to. Multiplying each axis by its own large prime and XORing decorrelates
    // the low bits, which is what makes neighbouring cells land in unrelated buckets instead of
    // clustering.
    std::size_t bucketForCell(const glm::ivec3 &cell) const;

    void buildSpatialHash();
    void findNeighbours();
    void computeDensitiesAndLambdas();
    void computePositionCorrections();
    void resolveBoundaryCollisions();
    void applyViscosity(float deltaTime);

    // Density the walls stand in for at this position, so a particle beside one is not treated as
    // though it were at a free surface.
    float boundaryDensity(const glm::vec3 &position) const;

    Config m_config;

    std::vector<glm::vec3> m_positions;
    std::vector<glm::vec3> m_velocities;
    // Positions advanced by velocity, which the constraint solve then corrects in place. PBF
    // derives the new velocity from how far these moved overall.
    std::vector<glm::vec3> m_predicted;
    std::vector<glm::vec3> m_corrections;
    std::vector<float>     m_densities;
    std::vector<float>     m_lambdas;
    // Flattened neighbour lists: neighbours of i are m_neighbours[m_neighbourOffsets[i]) up to
    // m_neighbourOffsets[i + 1]).
    std::vector<unsigned int> m_neighbours;
    std::vector<unsigned int> m_neighbourOffsets;

    // Spatial hash, in the same counting-sort layout as the neighbour lists above: the particles
    // in bucket b are m_cellParticles[m_cellStarts[b]) up to m_cellStarts[b + 1]). Held as two
    // flat arrays rather than a vector-per-bucket so rebuilding it each step does not reallocate,
    // and so the structure ports to the compute solver unchanged.
    std::vector<unsigned int> m_cellStarts;
    std::vector<unsigned int> m_cellParticles;
    // Bucket of each particle, kept from the counting pass so the fill pass does not rehash.
    std::vector<unsigned int> m_particleBuckets;
    // m_config.hashBuckets rounded up to a power of two, minus one: the mask that replaces the
    // modulo when wrapping a hash into a bucket.
    std::size_t m_bucketMask = 0;
};

} // namespace lr
