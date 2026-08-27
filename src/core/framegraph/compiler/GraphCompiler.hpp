#pragma once

#include "core/framegraph/model/GraphDefinition.hpp"

#include <vector>

namespace lr::framegraph
{

// Backend-independent compiler output. Barrier and physical-allocation plans
// will be added without introducing Vulkan types into this layer.
struct ExecutionPlan
{
    std::vector<PassId> orderedPasses;
};

// Compatibility policy for the current renderer. Step 2 will replace this
// with full RAW/WAR/WAW inference while retaining GraphDefinition unchanged.
ExecutionPlan buildLegacyExecutionPlan(const GraphDefinition &graph);

} // namespace lr::framegraph
