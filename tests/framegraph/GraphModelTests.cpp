#include "core/framegraph/compiler/GraphCompiler.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

void require(bool condition, const std::string &message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

void directFrontendConstruction()
{
    using namespace lr::framegraph;

    GraphDefinition graph;
    const ResourceId output = graph.addResource("lighting", ResourceKind::Image);
    const PassId producer = graph.addPass("producer", PassKind::Graphics);
    const PassId consumer = graph.addPass("consumer", PassKind::Graphics);

    graph.addAccess(producer, {
        .resource = output,
        .mode     = AccessMode::Write,
        .usage    = ResourceUsage::ColorOrDepthAttachment,
    });
    graph.addAccess(consumer, {
        .resource = output,
        .mode     = AccessMode::Read,
        .usage    = ResourceUsage::SampledImage,
    });

    const ExecutionPlan plan = buildLegacyExecutionPlan(graph);
    require(plan.orderedPasses == std::vector<PassId>({producer, consumer}),
            "a direct frontend should compile without Vulkan-facing descriptions");
}

void stableTypedIdentity()
{
    using namespace lr::framegraph;

    GraphDefinition graph;
    const ResourceId first = graph.addResource("color", ResourceKind::Image);
    const ResourceId repeated = graph.addResource("color", ResourceKind::Image);
    const PassId pass = graph.addPass("final", PassKind::Graphics);

    require(first == repeated, "resource names should intern to stable ids");
    require(graph.findResource("color") == first, "resource lookup should return its typed id");
    require(graph.findPass("final") == pass, "pass lookup should return its typed id");
    require(!graph.findResource("missing"), "missing resources should return an invalid id");
}

} // namespace

int main()
{
    try
    {
        directFrontendConstruction();
        stableTypedIdentity();
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }

    std::cout << "2 test(s) passed\n";
    return 0;
}
