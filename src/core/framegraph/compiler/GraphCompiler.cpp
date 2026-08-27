#include "GraphCompiler.hpp"

#include <queue>
#include <stdexcept>
#include <unordered_set>

namespace
{

bool isDescriptorUsage(lr::framegraph::ResourceUsage usage)
{
    using lr::framegraph::ResourceUsage;
    return usage == ResourceUsage::SampledImage || usage == ResourceUsage::StorageImage ||
           usage == ResourceUsage::UniformBuffer || usage == ResourceUsage::StorageBuffer ||
           usage == ResourceUsage::Unknown;
}

} // namespace

namespace lr::framegraph
{

ExecutionPlan buildLegacyExecutionPlan(const GraphDefinition &graph)
{
    const size_t count = graph.passes().size();
    std::vector<PassId> resourceProducers(graph.resources().size());

    for (const PassNode &pass : graph.passes())
    {
        for (const ResourceAccess &access : pass.accesses)
        {
            const bool legacyWrite = access.usage == ResourceUsage::ColorOrDepthAttachment ||
                                     (isDescriptorUsage(access.usage) &&
                                      access.mode != AccessMode::Read);
            if (legacyWrite)
            {
                resourceProducers[access.resource.value] = pass.id;
            }
        }
    }

    std::vector<std::unordered_set<uint32_t>> outEdges(count);
    for (const PassNode &consumer : graph.passes())
    {
        for (const ResourceAccess &access : consumer.accesses)
        {
            if (!isDescriptorUsage(access.usage) || access.mode != AccessMode::Read)
            {
                continue;
            }

            const PassId producer = resourceProducers[access.resource.value];
            if (producer && producer != consumer.id)
            {
                outEdges[producer.value].insert(consumer.id.value);
            }
        }

        for (PassId dependency : consumer.explicitDependencies)
        {
            outEdges[dependency.value].insert(consumer.id.value);
        }
    }

    std::vector<size_t> inDegree(count, 0);
    for (const auto &edges : outEdges)
    {
        for (uint32_t destination : edges)
        {
            ++inDegree[destination];
        }
    }

    std::queue<uint32_t> ready;
    for (uint32_t index = 0; index < count; ++index)
    {
        if (inDegree[index] == 0)
        {
            ready.push(index);
        }
    }

    ExecutionPlan plan;
    plan.orderedPasses.reserve(count);
    while (!ready.empty())
    {
        const uint32_t pass = ready.front();
        ready.pop();
        plan.orderedPasses.push_back(PassId{pass});

        for (uint32_t destination : outEdges[pass])
        {
            if (--inDegree[destination] == 0)
            {
                ready.push(destination);
            }
        }
    }

    if (plan.orderedPasses.size() != count)
    {
        throw std::runtime_error("FrameGraph: cycle detected in pass dependencies");
    }

    return plan;
}

} // namespace lr::framegraph
