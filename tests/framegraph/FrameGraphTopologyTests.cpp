#include "core/framegraph/PassDescAdapter.hpp"
#include "core/framegraph/FrameGraphTopology.hpp"
#include "core/framegraph/compiler/GraphCompiler.hpp"
#include "core/framegraph/compiler/VulkanBarrierPlanner.hpp"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using Test = std::pair<const char *, std::function<void()>>;

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

lr::BindingDesc read(std::string resource)
{
    return {
        .resourceName = std::move(resource),
        .binding      = 0,
        .type         = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .access       = lr::BindingAccess::Read,
    };
}

void declarationOrderDefinesReadBeforeWrite()
{
    std::vector<lr::PassDesc> passes;
    auto &reader = lr::framegraph::appendPass(passes, "reader");
    reader.bindings.push_back(read("history"));
    auto &writer = lr::framegraph::appendPass(passes, "writer");
    writer.writes.push_back({.name = "history", .format = VK_FORMAT_R16G16B16A16_SFLOAT});

    const auto order = lr::framegraph::sortPasses(passes);
    require(order == std::vector<size_t>({0, 1}),
            "a read declared before a write is an initial read followed by a WAR dependency");
}

void unknownDependency()
{
    std::vector<lr::PassDesc> passes;
    auto &pass = lr::framegraph::appendPass(passes, "final");
    pass.explicitDeps.push_back("missing");

    requireThrowsContaining([&] { (void)lr::framegraph::sortPasses(passes); }, "no such pass exists");
}

void cycleDetection()
{
    std::vector<lr::PassDesc> passes;
    auto &a = lr::framegraph::appendPass(passes, "a");
    a.explicitDeps.push_back("b");
    auto &b = lr::framegraph::appendPass(passes, "b");
    b.explicitDeps.push_back("a");

    requireThrowsContaining([&] { (void)lr::framegraph::sortPasses(passes); }, "cycle detected");
}

void duplicatePassNames()
{
    std::vector<lr::PassDesc> passes;
    (void)lr::framegraph::appendPass(passes, "geometry");
    requireThrowsContaining([&] { (void)lr::framegraph::appendPass(passes, "geometry"); },
                            "duplicate pass name");
}

void attachmentResourceCreation()
{
    std::vector<lr::PassDesc> passes(2);
    passes[0].name = "geometry";
    passes[0].writes = {
        {.name = "color", .format = VK_FORMAT_R16G16B16A16_SFLOAT},
        {.name = "depth", .format = VK_FORMAT_D32_SFLOAT},
    };
    passes[1].name = "overlay";
    passes[1].writes = {{.name = "color", .format = VK_FORMAT_R16G16B16A16_SFLOAT}};

    const auto images = lr::framegraph::planAttachmentImages(passes, {1600, 900});
    require(images.size() == 2, "duplicate attachment writes should plan one image");
    require(images[0].extent.width == 1600 && images[0].extent.height == 900,
            "implicit color extent should use the default extent");
    require((images[0].usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) != 0,
            "color image needs color-attachment usage");
    require(images[0].aspect == VK_IMAGE_ASPECT_COLOR_BIT, "color image needs color aspect");
    require((images[1].usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0,
            "depth image needs depth-attachment usage");
    require(images[1].aspect == VK_IMAGE_ASPECT_DEPTH_BIT, "depth image needs depth aspect");

    const auto excludingExisting =
        lr::framegraph::planAttachmentImages(passes, {1600, 900}, {"color"});
    require(excludingExisting.size() == 1 && excludingExisting[0].name == "depth",
            "pre-registered images must not be planned again");
}

void relativeAndAbsoluteExtents()
{
    std::vector<lr::PassDesc> passes(1);
    passes[0].name = "outputs";
    passes[0].writes = {
        {.name = "relative", .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {0, 0}},
        {.name = "partially-zero", .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {320, 0}},
        {.name = "absolute", .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {512, 256}},
    };

    const auto images = lr::framegraph::planAttachmentImages(passes, {1920, 1080});
    require(images[0].extent.width == 1920 && images[0].extent.height == 1080,
            "zero extent should resolve to the default");
    require(images[1].extent.width == 1920 && images[1].extent.height == 1080,
            "today's partially-zero extent behavior should be captured");
    require(images[2].extent.width == 512 && images[2].extent.height == 256,
            "absolute extent should be preserved");
}

void deterministicDebugDump()
{
    std::vector<lr::PassDesc> passes(1);
    passes[0].name = "final";
    passes[0].type = lr::PassType::Fullscreen;
    passes[0].bindings.push_back(read("pbr"));
    passes[0].writes.push_back({.name = "swapchain", .format = VK_FORMAT_B8G8R8A8_SRGB});

    std::vector<size_t> order{0};
    std::vector<std::vector<lr::framegraph::BarrierDebugInfo>> barriers(1);
    barriers[0].push_back({
        .resourceName = "swapchain",
        .srcStage     = VK_PIPELINE_STAGE_2_NONE,
        .srcAccess    = VK_ACCESS_2_NONE,
        .dstStage     = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstAccess    = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        .oldLayout    = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    });

    const std::string first  = lr::framegraph::dumpTopology(passes, order, barriers);
    const std::string second = lr::framegraph::dumpTopology(passes, order, barriers);
    require(first == second, "debug dump must be deterministic");
    require(first.find("execution-order: final") != std::string::npos,
            "debug dump must contain execution order");
    require(first.find("binding 0 resource=pbr access=read") != std::string::npos,
            "debug dump must contain resource accesses");
    require(first.find("before=final resource=swapchain") != std::string::npos,
            "debug dump must contain barriers");
}

void graphDefinitionTranslation()
{
    std::vector<lr::PassDesc> passes(2);
    passes[0].name = "geometry";
    passes[0].type = lr::PassType::Geometry;
    passes[0].bindings.push_back({
        .resourceName = "camera",
        .binding      = 0,
        .type         = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .access       = lr::BindingAccess::Read,
    });
    passes[0].vertexBufferRefs.push_back({0, "vertices"});
    passes[0].indexBufferName = "indices";
    passes[0].writes.push_back({
        .name   = "gbuffer",
        .format = VK_FORMAT_R16G16B16A16_SFLOAT,
    });

    passes[1].name = "lighting";
    passes[1].bindings.push_back(read("gbuffer"));
    passes[1].explicitDeps.push_back("geometry");

    const lr::framegraph::GraphDefinition graph =
        lr::framegraph::translatePassDescriptions(passes);

    require(graph.passes().size() == 2, "IR should contain every pass");
    require(graph.resources().size() == 4, "IR should intern each logical resource once");
    require(graph.pass(graph.findPass("geometry")).kind == lr::framegraph::PassKind::Graphics,
            "adapter should normalize frontend pass type");
    require(graph.resource(graph.findResource("gbuffer")).kind ==
                lr::framegraph::ResourceKind::Image,
            "attachment resources should be images");
    require(graph.resource(graph.findResource("vertices")).kind ==
                lr::framegraph::ResourceKind::Buffer,
            "vertex resources should be buffers");

    const auto &geometry = graph.pass(graph.findPass("geometry"));
    require(geometry.accesses.size() == 4,
            "IR should capture descriptor, attachment, vertex, and index accesses");
    require(geometry.accesses[0].usage == lr::framegraph::ResourceUsage::UniformBuffer &&
                geometry.accesses[0].mode == lr::framegraph::AccessMode::Read,
            "descriptor access should be normalized");
    require(geometry.accesses[1].usage == lr::framegraph::ResourceUsage::ColorOrDepthAttachment &&
                geometry.accesses[1].mode == lr::framegraph::AccessMode::Write,
            "attachment access should be normalized");
    require(geometry.accesses[2].usage == lr::framegraph::ResourceUsage::VertexBuffer,
            "vertex-buffer access should be represented");
    require(geometry.accesses[3].usage == lr::framegraph::ResourceUsage::IndexBuffer,
            "index-buffer access should be represented");

    const auto &lighting = graph.pass(graph.findPass("lighting"));
    require(lighting.explicitDependencies.size() == 1 &&
                graph.pass(lighting.explicitDependencies[0]).name == "geometry",
            "explicit dependency names should resolve to stable pass ids");
}

void executionPlanMatchesPassDescFrontend()
{
    std::vector<lr::PassDesc> passes;
    auto &consumer = lr::framegraph::appendPass(passes, "consumer");
    consumer.bindings.push_back(read("lighting"));
    auto &producer = lr::framegraph::appendPass(passes, "producer");
    producer.writes.push_back({.name = "lighting", .format = VK_FORMAT_R16G16B16A16_SFLOAT});

    const auto graph = lr::framegraph::translatePassDescriptions(passes);
    const auto plan = lr::framegraph::buildExecutionPlan(graph);
    const auto compatibilityOrder = lr::framegraph::sortPasses(passes);

    require(plan.orderedPasses.size() == compatibilityOrder.size(),
            "IR execution plan should contain every sorted pass");
    for (size_t index = 0; index < compatibilityOrder.size(); ++index)
    {
        require(plan.orderedPasses[index].value == compatibilityOrder[index],
                "IR execution plan should preserve characterized ordering");
    }
}

void sameLayoutAttachmentHazard()
{
    std::vector<lr::PassDesc> passes(2);
    passes[0].name = "final";
    passes[0].writes.push_back({
        .name = "swapchain",
        .format = VK_FORMAT_B8G8R8A8_SRGB,
    });
    passes[1].name = "imgui";
    passes[1].writes.push_back({
        .name = "swapchain",
        .format = VK_FORMAT_B8G8R8A8_SRGB,
        .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
    });

    const auto plan = lr::framegraph::planVulkanBarriers(passes, std::vector<size_t>{0, 1});
    require(plan.beforePass[1].size() == 1,
            "attachment WAW must synchronize even when its layout is unchanged");
    const auto &barrier = plan.beforePass[1][0];
    require(barrier.source.layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL &&
                barrier.destination.layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            "same-layout attachment barrier must preserve the attachment layout");
    require((barrier.source.access & VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT) != 0,
            "ImGui barrier must wait for the final pass color write");
    require((barrier.destination.access & VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT) != 0 &&
                (barrier.destination.access & VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT) != 0,
            "LOAD attachment must declare both color read and write access");
}

void readOnlyImageAccessesCoalesce()
{
    std::vector<lr::PassDesc> passes(2);
    passes[0].name = "first reader";
    passes[0].bindings.push_back(read("texture"));
    passes[1].name = "second reader";
    passes[1].bindings.push_back(read("texture"));

    const auto plan = lr::framegraph::planVulkanBarriers(
        passes, std::vector<size_t>{0, 1},
        {{"texture", VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}});
    require(plan.beforePass[0].empty() && plan.beforePass[1].empty(),
            "same-layout read-only accesses should not emit barriers");
}

void storageImageSameLayoutHazard()
{
    auto storageWrite = [](std::string resource) {
        return lr::BindingDesc{
            .resourceName = std::move(resource),
            .binding = 0,
            .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .stages = VK_SHADER_STAGE_COMPUTE_BIT,
            .access = lr::BindingAccess::Write,
        };
    };

    std::vector<lr::PassDesc> passes(2);
    passes[0].name = "first write";
    passes[0].bindings.push_back(storageWrite("storage"));
    passes[1].name = "second write";
    passes[1].bindings.push_back(storageWrite("storage"));

    const auto plan = lr::framegraph::planVulkanBarriers(passes, std::vector<size_t>{0, 1});
    require(plan.beforePass[1].size() == 1,
            "storage WAW must synchronize while remaining in GENERAL");
    const auto &barrier = plan.beforePass[1][0];
    require(barrier.source.layout == VK_IMAGE_LAYOUT_GENERAL &&
                barrier.destination.layout == VK_IMAGE_LAYOUT_GENERAL,
            "storage WAW should use a same-layout GENERAL barrier");
    require(barrier.source.access == VK_ACCESS_2_SHADER_WRITE_BIT &&
                barrier.destination.access == VK_ACCESS_2_SHADER_WRITE_BIT,
            "storage WAW barrier must connect shader writes");
}

void bufferRawHazard()
{
    std::vector<lr::PassDesc> passes(2);
    passes[0].name = "compute writer";
    passes[0].bindings.push_back({
        .resourceName = "values",
        .binding = 0,
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .stages = VK_SHADER_STAGE_COMPUTE_BIT,
        .access = lr::BindingAccess::Write,
    });
    passes[1].name = "fragment reader";
    passes[1].bindings.push_back({
        .resourceName = "values",
        .binding = 0,
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .stages = VK_SHADER_STAGE_FRAGMENT_BIT,
        .access = lr::BindingAccess::Read,
    });

    const auto plan = lr::framegraph::planVulkanBarriers(passes, std::vector<size_t>{0, 1});
    require(plan.beforePass[1].size() == 1,
            "storage-buffer RAW must produce a buffer barrier");
    const auto &barrier = plan.beforePass[1][0];
    require(barrier.kind == lr::framegraph::BarrierResourceKind::Buffer,
            "storage-buffer hazard must retain buffer identity");
    require(barrier.source.stages == VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT &&
                barrier.destination.stages == VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            "buffer RAW must synchronize compute writes to fragment reads");
    require(barrier.source.access == VK_ACCESS_2_SHADER_WRITE_BIT &&
                barrier.destination.access == VK_ACCESS_2_SHADER_READ_BIT,
            "buffer RAW must use shader write/read access masks");
}

void vertexShaderStageMapping()
{
    std::vector<lr::PassDesc> passes(1);
    passes[0].name = "vertex sampler";
    passes[0].bindings.push_back({
        .resourceName = "displacement",
        .binding = 0,
        .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .stages = VK_SHADER_STAGE_VERTEX_BIT,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    });

    const auto plan = lr::framegraph::planVulkanBarriers(passes, std::vector<size_t>{0});
    require(plan.beforePass[0].size() == 1 &&
                plan.beforePass[0][0].destination.stages == VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
            "sampled images used by vertex shaders must target the vertex stage");
}

} // namespace

int main()
{
    const std::vector<Test> tests = {
        {"read before write declaration order", declarationOrderDefinesReadBeforeWrite},
        {"unknown dependencies", unknownDependency},
        {"cycle detection", cycleDetection},
        {"duplicate pass names", duplicatePassNames},
        {"attachment resource creation", attachmentResourceCreation},
        {"relative and absolute extents", relativeAndAbsoluteExtents},
        {"deterministic debug dump", deterministicDebugDump},
        {"graph definition translation", graphDefinitionTranslation},
        {"execution plan PassDesc frontend", executionPlanMatchesPassDescFrontend},
        {"same-layout attachment hazard", sameLayoutAttachmentHazard},
        {"read-only image accesses coalesce", readOnlyImageAccessesCoalesce},
        {"same-layout storage image hazard", storageImageSameLayoutHazard},
        {"buffer RAW hazard", bufferRawHazard},
        {"vertex shader stage mapping", vertexShaderStageMapping},
    };

    size_t failures = 0;
    for (const auto &[name, test] : tests)
    {
        try
        {
            test();
            std::cout << "[PASS] " << name << '\n';
        }
        catch (const std::exception &error)
        {
            ++failures;
            std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
        }
    }

    if (failures != 0)
    {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }

    std::cout << tests.size() << " test(s) passed\n";
    return 0;
}
