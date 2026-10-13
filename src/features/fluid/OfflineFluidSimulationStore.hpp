#pragma once

#include "core/utility/Uuid.hpp"
#include "features/fluid/OfflineFluidSimulation.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace lr
{

using OfflineFluidSimulationHandle = uint32_t;
using OfflineFluidSimulationId     = Uuid;

class OfflineFluidSimulationStore
{
public:
    OfflineFluidSimulationHandle add(OfflineFluidSimulation simulation);
    OfflineFluidSimulationHandle add(OfflineFluidSimulation simulation, OfflineFluidSimulationId id);

    OfflineFluidSimulation                     &get(OfflineFluidSimulationHandle handle);
    const OfflineFluidSimulation               &get(OfflineFluidSimulationHandle handle) const;
    OfflineFluidSimulationId                    idOf(OfflineFluidSimulationHandle handle) const;
    std::optional<OfflineFluidSimulationHandle> find(OfflineFluidSimulationId id) const;

    uint32_t size() const { return static_cast<uint32_t>(m_simulations.size()); }
    bool     contains(OfflineFluidSimulationHandle handle) const { return handle < m_simulations.size(); }
    void     clear();

private:
    std::vector<std::unique_ptr<OfflineFluidSimulation>>                       m_simulations;
    std::vector<OfflineFluidSimulationId>                                      m_ids;
    std::unordered_map<OfflineFluidSimulationId, OfflineFluidSimulationHandle> m_handlesById;
};

} // namespace lr
