#include "core/framegraph/compiler/GraphCompiler.hpp"

#include <functional>
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

template <typename Fn>
void requireThrowsContaining(Fn &&fn, const std::string &needle)
{
    try
    {
        fn();
    }
    catch (const std::exception &error)
    {
        require(std::string(error.what()).find(needle) != std::string::npos,
                "exception did not contain '" + needle + "': " + error.what());
        return;
    }
    throw std::runtime_error("expected an exception containing '" + needle + "'");
}

lr::framegraph::ResourceAccess access(lr::framegraph::ResourceId resource,
                                      lr::framegraph::AccessMode mode)
{
    return {
        .resource = resource,
        .mode     = mode,
        .usage    = lr::framegraph::ResourceUsage::StorageBuffer,
    };
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

    const ExecutionPlan plan = buildExecutionPlan(graph);
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

void rawDependency()
{
    using namespace lr::framegraph;

    GraphDefinition graph;
    const ResourceId resource = graph.addResource("value", ResourceKind::Buffer);
    const PassId writer = graph.addPass("writer", PassKind::Compute);
    const PassId reader = graph.addPass("reader", PassKind::Compute);
    graph.addAccess(writer, access(resource, AccessMode::Write));
    graph.addAccess(reader, access(resource, AccessMode::Read));

    // This reverse explicit edge forms a cycle only if RAW inferred writer -> reader.
    graph.addDependency(writer, reader);
    requireThrowsContaining([&] { (void)buildExecutionPlan(graph); }, "cycle detected");
}

void warDependency()
{
    using namespace lr::framegraph;

    GraphDefinition graph;
    const ResourceId resource = graph.addResource("history", ResourceKind::Buffer);
    const PassId reader = graph.addPass("reader", PassKind::Compute);
    const PassId writer = graph.addPass("writer", PassKind::Compute);
    graph.addAccess(reader, access(resource, AccessMode::Read));
    graph.addAccess(writer, access(resource, AccessMode::Write));

    graph.addDependency(reader, writer);
    requireThrowsContaining([&] { (void)buildExecutionPlan(graph); }, "cycle detected");
}

void wawDependency()
{
    using namespace lr::framegraph;

    GraphDefinition graph;
    const ResourceId resource = graph.addResource("output", ResourceKind::Image);
    const PassId first = graph.addPass("first writer", PassKind::Compute);
    const PassId second = graph.addPass("second writer", PassKind::Compute);
    graph.addAccess(first, access(resource, AccessMode::Write));
    graph.addAccess(second, access(resource, AccessMode::Write));

    graph.addDependency(first, second);
    requireThrowsContaining([&] { (void)buildExecutionPlan(graph); }, "cycle detected");
}

void readWriteDependency()
{
    using namespace lr::framegraph;

    GraphDefinition graph;
    const ResourceId resource = graph.addResource("accumulator", ResourceKind::Buffer);
    const PassId writer = graph.addPass("initialize", PassKind::Compute);
    const PassId readWriter = graph.addPass("accumulate", PassKind::Compute);
    graph.addAccess(writer, access(resource, AccessMode::Write));
    graph.addAccess(readWriter, access(resource, AccessMode::ReadWrite));

    graph.addDependency(writer, readWriter);
    requireThrowsContaining([&] { (void)buildExecutionPlan(graph); }, "cycle detected");
}

void repeatedAccessDoesNotSelfDepend()
{
    using namespace lr::framegraph;

    GraphDefinition graph;
    const ResourceId resource = graph.addResource("shared", ResourceKind::Buffer);
    const PassId pass = graph.addPass("combined", PassKind::Compute);
    graph.addAccess(pass, access(resource, AccessMode::Read));
    graph.addAccess(pass, access(resource, AccessMode::Write));

    const ExecutionPlan plan = buildExecutionPlan(graph);
    require(plan.orderedPasses == std::vector<PassId>({pass}),
            "multiple mentions in one pass must collapse without a self-cycle");
}

void declarationOrderBreaksTies()
{
    using namespace lr::framegraph;

    GraphDefinition graph;
    const PassId first = graph.addPass("first", PassKind::Compute);
    const PassId second = graph.addPass("second", PassKind::Graphics);
    const PassId third = graph.addPass("third", PassKind::External);

    const ExecutionPlan plan = buildExecutionPlan(graph);
    require(plan.orderedPasses == std::vector<PassId>({first, second, third}),
            "independent passes should retain declaration order");
}

} // namespace

int main()
{
    try
    {
        directFrontendConstruction();
        stableTypedIdentity();
        rawDependency();
        warDependency();
        wawDependency();
        readWriteDependency();
        repeatedAccessDoesNotSelfDepend();
        declarationOrderBreaksTies();
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }

    std::cout << "8 test(s) passed\n";
    return 0;
}
