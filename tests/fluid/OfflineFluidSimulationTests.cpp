#include "features/fluid/OfflineFluidSimulation.hpp"
#include "features/fluid/OfflineFluidSimulationStore.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace
{

void buildsAnimatedDnaHelix()
{
    const lr::OfflineFluidSimulation helix = lr::offline_fluid_examples::makeDnaHelix(12, 20, 24.0f);
    assert(helix.name() == "DNA helix");
    assert(helix.frames().size() == 12);
    assert(helix.positions().size() == 12 * 40);
    assert(helix.maximumParticleCount() == 40);
    assert(std::abs(helix.durationSeconds() - 0.5f) < 1e-6f);
    assert(helix.framePositions(0).size() == 40);
    assert(helix.framePositions(1).front() != helix.framePositions(0).front());
    for (const auto &frame : helix.frames())
    {
        assert(frame.boundsMin.y <= -0.79f);
        assert(frame.boundsMax.y >= 0.79f);
    }
}

void enforcesFrameInvariants()
{
    lr::OfflineFluidSimulation   simulation("validation", 30.0f);
    const std::vector<glm::vec3> positions{{0, 0, 0}, {1, 2, 3}};
    simulation.addFrame(0.0f, positions);

    bool rejected = false;
    try
    {
        simulation.addFrame(0.0f, positions);
    } catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    assert(rejected);

    rejected = false;
    try
    {
        simulation.addFrame(1.0f / 30.0f, positions, std::span(positions).first(1));
    } catch (const std::invalid_argument &)
    {
        rejected = true;
    }
    assert(rejected);
}

void storeUsesStableIdsAndDenseHandles()
{
    lr::OfflineFluidSimulationStore store;
    const auto                      first  = store.add(lr::offline_fluid_examples::makeDnaHelix(2, 4));
    const auto                      second = store.add(lr::offline_fluid_examples::makeDnaHelix(3, 5));
    assert(first == 0 && second == 1);
    assert(store.find(store.idOf(first)) == first);
    assert(store.get(second).frames().size() == 3);
}

} // namespace

int main()
{
    buildsAnimatedDnaHelix();
    enforcesFrameInvariants();
    storeUsesStableIdsAndDenseHandles();
    return 0;
}
