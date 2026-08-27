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

// Builds a deterministic execution order from logical resource accesses.
// Accesses are interpreted in declaration order:
//   read       depends on the most recent writer (RAW),
//   write      depends on the most recent writer (WAW) and all readers since it (WAR),
//   read-write applies both rules before becoming the new writer.
// Pass declaration order breaks ties between otherwise independent passes.
ExecutionPlan buildExecutionPlan(const GraphDefinition &graph);

} // namespace lr::framegraph
