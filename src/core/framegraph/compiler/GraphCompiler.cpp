#include "GraphCompiler.hpp"

#include <algorithm>
#include <functional>
#include <numeric>
#include <optional>
#include <queue>
#include <stdexcept>
#include <unordered_set>

namespace lr::framegraph
{

ExecutionPlan buildExecutionPlan(const GraphDefinition &graph)
{
    const size_t passCount     = graph.passes().size();
    const size_t resourceCount = graph.resources().size();

    struct AccessHistory
    {
        std::optional<PassId> lastWriter;
        std::vector<PassId>   readersSinceLastWrite;
    };

    std::vector<AccessHistory>                histories(resourceCount);
    std::vector<std::unordered_set<uint32_t>> outEdges(passCount);

    const auto addEdge = [&](PassId source, PassId destination) {
        if (source != destination)
        {
            outEdges[source.value].insert(destination.value);
        }
    };

    // Effective declaration order: runsLast passes behave as if declared after all others, both
    // when deriving hazards from access order and when breaking ties between ready passes.
    std::vector<uint32_t> declarationOrder(passCount);
    std::iota(declarationOrder.begin(), declarationOrder.end(), 0u);
    std::stable_partition(declarationOrder.begin(), declarationOrder.end(), [&](uint32_t index) {
        return !graph.passes()[index].runsLast;
    });
    std::vector<uint32_t> rank(passCount);
    for (uint32_t position = 0; position < passCount; ++position)
    {
        rank[declarationOrder[position]] = position;
    }

    // A frontend may mention the same resource more than once in one pass
    // (for example through two descriptor bindings). Collapse those mentions
    // before updating history so a pass can never create a dependency on itself.
    for (const uint32_t passIndex : declarationOrder)
    {
        const PassNode &pass = graph.passes()[passIndex];
        struct CombinedAccess
        {
            bool reads  = false;
            bool writes = false;
        };

        std::vector<CombinedAccess> combined(resourceCount);
        for (const ResourceAccess &access : pass.accesses)
        {
            CombinedAccess &intent = combined[access.resource.value];
            intent.reads |= access.mode != AccessMode::Write;
            intent.writes |= access.mode != AccessMode::Read;
        }

        for (uint32_t resourceIndex = 0; resourceIndex < resourceCount; ++resourceIndex)
        {
            const CombinedAccess intent = combined[resourceIndex];
            if (!intent.reads && !intent.writes)
            {
                continue;
            }

            AccessHistory &history = histories[resourceIndex];

            // RAW: this pass consumes the value produced by the latest writer.
            if (intent.reads && history.lastWriter)
            {
                addEdge(*history.lastWriter, pass.id);
            }

            if (intent.writes)
            {
                // WAW: preserve the order of successive writes.
                if (history.lastWriter)
                {
                    addEdge(*history.lastWriter, pass.id);
                }

                // WAR: do not overwrite a value until all preceding readers
                // that consumed it have completed.
                for (PassId reader : history.readersSinceLastWrite)
                {
                    addEdge(reader, pass.id);
                }

                history.readersSinceLastWrite.clear();
                history.lastWriter = pass.id;
            } else
            {
                history.readersSinceLastWrite.push_back(pass.id);
            }
        }
    }

    for (const PassNode &pass : graph.passes())
    {
        for (PassId dependency : pass.explicitDependencies)
        {
            addEdge(dependency, pass.id);
        }
    }

    std::vector<size_t> inDegree(passCount, 0);
    for (const auto &edges : outEdges)
    {
        for (uint32_t destination : edges)
        {
            ++inDegree[destination];
        }
    }

    // Always choose the earliest-declared ready pass (by effective rank). This makes compilation
    // reproducible even though edge de-duplication uses unordered sets.
    std::priority_queue<uint32_t, std::vector<uint32_t>, std::greater<>> ready; // holds ranks
    for (uint32_t index = 0; index < passCount; ++index)
    {
        if (inDegree[index] == 0)
        {
            ready.push(rank[index]);
        }
    }

    ExecutionPlan plan;
    plan.orderedPasses.reserve(passCount);
    while (!ready.empty())
    {
        const uint32_t pass = declarationOrder[ready.top()];
        ready.pop();
        plan.orderedPasses.push_back(PassId{pass});

        for (uint32_t destination : outEdges[pass])
        {
            if (--inDegree[destination] == 0)
            {
                ready.push(rank[destination]);
            }
        }
    }

    if (plan.orderedPasses.size() != passCount)
    {
        throw std::runtime_error("FrameGraph: cycle detected in pass dependencies");
    }

    return plan;
}

} // namespace lr::framegraph
