#pragma once

#include "core/utility/Uuid.hpp"
#include "features/eulerian_particle/OfflineSmokeSimulation.hpp"

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace lr::eulerian
{

using OfflineSmokeSimulationHandle = uint32_t;
using OfflineSmokeSimulationId     = Uuid;

class OfflineSmokeSimulationStore
{
public:
    OfflineSmokeSimulationHandle add(OfflineSmokeSimulation simulation);
    OfflineSmokeSimulationHandle add(OfflineSmokeSimulation simulation, OfflineSmokeSimulationId id);
    OfflineSmokeSimulation       &get(OfflineSmokeSimulationHandle handle);
    const OfflineSmokeSimulation &get(OfflineSmokeSimulationHandle handle) const;
    OfflineSmokeSimulationId      idOf(OfflineSmokeSimulationHandle handle) const;
    std::optional<OfflineSmokeSimulationHandle> find(OfflineSmokeSimulationId id) const;
    uint32_t size() const { return static_cast<uint32_t>(m_simulations.size()); }
    bool contains(OfflineSmokeSimulationHandle handle) const { return handle < m_simulations.size(); }
    void clear();

private:
    std::vector<std::unique_ptr<OfflineSmokeSimulation>>                       m_simulations;
    std::vector<OfflineSmokeSimulationId>                                      m_ids;
    std::unordered_map<OfflineSmokeSimulationId, OfflineSmokeSimulationHandle> m_handlesById;
};

} // namespace lr::eulerian
