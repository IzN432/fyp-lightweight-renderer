#include "features/fluid/OfflineFluidSimulationStore.hpp"

#include <stdexcept>
#include <utility>

namespace lr
{

OfflineFluidSimulationHandle OfflineFluidSimulationStore::add(OfflineFluidSimulation simulation)
{
    return add(std::move(simulation), generateUuid());
}

OfflineFluidSimulationHandle OfflineFluidSimulationStore::add(OfflineFluidSimulation   simulation,
                                                              OfflineFluidSimulationId id)
{
    if (id.is_nil())
    {
        throw std::invalid_argument("An offline fluid simulation cannot use a nil ID");
    }
    if (m_handlesById.contains(id))
    {
        throw std::invalid_argument("An offline fluid simulation with ID " + toString(id) + " already exists");
    }
    const auto handle = static_cast<OfflineFluidSimulationHandle>(m_simulations.size());
    m_simulations.push_back(std::make_unique<OfflineFluidSimulation>(std::move(simulation)));
    m_ids.push_back(id);
    m_handlesById.emplace(id, handle);
    return handle;
}

OfflineFluidSimulation &OfflineFluidSimulationStore::get(OfflineFluidSimulationHandle handle)
{
    return const_cast<OfflineFluidSimulation &>(std::as_const(*this).get(handle));
}

const OfflineFluidSimulation &OfflineFluidSimulationStore::get(OfflineFluidSimulationHandle handle) const
{
    if (!contains(handle))
    {
        throw std::out_of_range("Offline fluid simulation handle is out of range");
    }
    return *m_simulations[handle];
}

OfflineFluidSimulationId OfflineFluidSimulationStore::idOf(OfflineFluidSimulationHandle handle) const
{
    if (!contains(handle))
    {
        throw std::out_of_range("Offline fluid simulation handle is out of range");
    }
    return m_ids[handle];
}

std::optional<OfflineFluidSimulationHandle> OfflineFluidSimulationStore::find(OfflineFluidSimulationId id) const
{
    const auto found = m_handlesById.find(id);
    return found == m_handlesById.end() ? std::nullopt : std::optional(found->second);
}

void OfflineFluidSimulationStore::clear()
{
    m_simulations.clear();
    m_ids.clear();
    m_handlesById.clear();
}

} // namespace lr
