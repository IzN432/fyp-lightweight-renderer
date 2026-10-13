#include "features/eulerian_particle/OfflineSmokeSimulationStore.hpp"

#include <stdexcept>
#include <utility>

namespace lr::eulerian
{

OfflineSmokeSimulationHandle OfflineSmokeSimulationStore::add(OfflineSmokeSimulation simulation)
{
    return add(std::move(simulation), generateUuid());
}

OfflineSmokeSimulationHandle OfflineSmokeSimulationStore::add(OfflineSmokeSimulation simulation,
                                                               OfflineSmokeSimulationId id)
{
    if (id.is_nil()) throw std::invalid_argument("An offline smoke simulation cannot use a nil ID");
    if (m_handlesById.contains(id))
        throw std::invalid_argument("An offline smoke simulation with ID " + toString(id) + " already exists");
    const auto handle = static_cast<OfflineSmokeSimulationHandle>(m_simulations.size());
    m_simulations.push_back(std::make_unique<OfflineSmokeSimulation>(std::move(simulation)));
    m_ids.push_back(id);
    m_handlesById.emplace(id, handle);
    return handle;
}

OfflineSmokeSimulation &OfflineSmokeSimulationStore::get(OfflineSmokeSimulationHandle handle)
{
    return const_cast<OfflineSmokeSimulation &>(std::as_const(*this).get(handle));
}

const OfflineSmokeSimulation &OfflineSmokeSimulationStore::get(OfflineSmokeSimulationHandle handle) const
{
    if (!contains(handle)) throw std::out_of_range("Offline smoke simulation handle is out of range");
    return *m_simulations[handle];
}

OfflineSmokeSimulationId OfflineSmokeSimulationStore::idOf(OfflineSmokeSimulationHandle handle) const
{
    if (!contains(handle)) throw std::out_of_range("Offline smoke simulation handle is out of range");
    return m_ids[handle];
}

std::optional<OfflineSmokeSimulationHandle> OfflineSmokeSimulationStore::find(OfflineSmokeSimulationId id) const
{
    const auto found = m_handlesById.find(id);
    return found == m_handlesById.end() ? std::nullopt : std::optional(found->second);
}

void OfflineSmokeSimulationStore::clear()
{
    m_simulations.clear();
    m_ids.clear();
    m_handlesById.clear();
}

} // namespace lr::eulerian
