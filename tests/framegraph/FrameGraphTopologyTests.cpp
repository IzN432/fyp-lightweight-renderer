#include "core/framegraph/FrameGraphDefinition.hpp"
#include "core/framegraph/FrameGraphTopology.hpp"
#include "core/framegraph/PassBuilder.hpp"
#include "core/framegraph/PassDescAdapter.hpp"
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

template <typename Fn> void requireThrowsContaining(Fn &&fn, const std::string &needle)
{
    try
    {
        fn();
    } catch (const std::exception &error)
    {
        require(std::string(error.what()).find(needle) != std::string::npos,
                "exception did not contain '" + needle + "': " + error.what());
        return;
    }
    throw std::runtime_error("expected an exception containing '" + needle + "'");
}

lr::PassBuilder builder(lr::FrameGraphDefinition &definition, lr::PassHandle pass) { return {definition, pass}; }

std::vector<size_t> sort(const lr::FrameGraphDefinition &definition)
{
    return lr::framegraph::sortPasses(definition.passes(), definition.resources(), definition.owner());
}

void declarationOrderDefinesReadBeforeWrite()
{
    lr::FrameGraphDefinition definition;
    const auto               history = definition.buffer("history");
    builder(definition, definition.addPass("reader")).storageBufferRead(0, history, VK_SHADER_STAGE_COMPUTE_BIT);
    builder(definition, definition.addPass("writer")).storageBufferWrite(0, history, VK_SHADER_STAGE_COMPUTE_BIT);

    require(sort(definition) == std::vector<size_t>({0, 1}),
            "a read declared before a write should create a WAR dependency");
}

void explicitDependenciesAreTyped()
{
    lr::FrameGraphDefinition definition;
    const auto               first  = definition.addPass("first");
    const auto               second = definition.addPass("second");
    builder(definition, first).dependsOn(second);

    require(sort(definition) == std::vector<size_t>({1, 0}), "typed dependencies should determine execution order");

    definition.pass(first).explicitDependencies = {{99, definition.owner()}};
    requireThrowsContaining(
        [&] {
            (void)sort(definition);
        },
        "invalid pass dependency handle");
}

void cycleDetection()
{
    lr::FrameGraphDefinition definition;
    const auto               a = definition.addPass("a");
    const auto               b = definition.addPass("b");
    builder(definition, a).dependsOn(b);
    builder(definition, b).dependsOn(a);
    requireThrowsContaining(
        [&] {
            (void)sort(definition);
        },
        "cycle detected");
}

void duplicatePassNames()
{
    lr::FrameGraphDefinition definition;
    (void)definition.addPass("geometry");
    requireThrowsContaining(
        [&] {
            (void)definition.addPass("geometry");
        },
        "duplicate pass name");
}

void attachmentPlanningUsesSemanticDeclarations()
{
    lr::FrameGraphDefinition definition;
    const auto               color = definition.image("color");
    const auto               depth = definition.image("depth");
    builder(definition, definition.addPass("geometry"))
        .colorAttachment(color, VK_FORMAT_R16G16B16A16_SFLOAT)
        .depthAttachment(depth, VK_FORMAT_D32_SFLOAT, VK_ATTACHMENT_LOAD_OP_CLEAR, {}, {800, 600});
    builder(definition, definition.addPass("overlay"))
        .colorAttachment(color, VK_FORMAT_R16G16B16A16_SFLOAT, VK_ATTACHMENT_LOAD_OP_LOAD);
    builder(definition, definition.addPass("post-process")).storageImageWrite(0, color, VK_SHADER_STAGE_COMPUTE_BIT);

    const auto images = lr::framegraph::planAttachmentImages(definition.passes(), definition.resources(), {1600, 900});
    require(images.size() == 2, "repeated attachment declarations should plan one image");
    require(images[0].name == "color" && images[0].extent.width == 1600, "default extent should be used for color");
    require((images[0].usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) != 0 &&
                (images[0].usage & VK_IMAGE_USAGE_STORAGE_BIT) != 0,
            "image creation flags should aggregate every declared use");
    require(images[1].name == "depth" && images[1].extent.width == 800,
            "explicit attachment extent should be retained");
}

void contradictoryAttachmentDeclarationsAreRejected()
{
    lr::FrameGraphDefinition definition;
    const auto               output = definition.image("output");
    builder(definition, definition.addPass("first")).colorAttachment(output, VK_FORMAT_R16G16B16A16_SFLOAT);
    builder(definition, definition.addPass("second")).colorAttachment(output, VK_FORMAT_R8G8B8A8_UNORM);

    requireThrowsContaining(
        [&] {
            (void)lr::framegraph::planAttachmentImages(definition.passes(), definition.resources(), {1600, 900});
        },
        "conflicting formats");
}

void translationPreservesSemanticIntent()
{
    lr::FrameGraphDefinition definition;
    const auto               image  = definition.image("lighting");
    const auto               buffer = definition.buffer("constants");
    builder(definition, definition.addPass("shade"))
        .sampledImage(0, image, VK_SHADER_STAGE_FRAGMENT_BIT)
        .uniformBuffer(1, buffer, VK_SHADER_STAGE_FRAGMENT_BIT);

    const auto graph =
        lr::framegraph::translatePassDescriptions(definition.passes(), definition.resources(), definition.owner());
    require(graph.passes().size() == 1 && graph.passes()[0].accesses.size() == 2,
            "semantic uses should translate to graph accesses");
    require(graph.resources().size() == 2, "typed image and buffer handles should become graph resources");
}

void handlesCannotCrossDefinitions()
{
    lr::FrameGraphDefinition first;
    lr::FrameGraphDefinition second;
    const auto               foreign = first.image("foreign");
    builder(second, second.addPass("bad")).sampledImage(0, foreign, VK_SHADER_STAGE_FRAGMENT_BIT);

    requireThrowsContaining(
        [&] {
            (void)lr::framegraph::translatePassDescriptions(second.passes(), second.resources(), second.owner());
        },
        "another graph");
}

void barriersRemainWholeResource()
{
    lr::FrameGraphDefinition definition;
    const auto               image  = definition.image("mipped");
    const auto               buffer = definition.buffer("data");
    builder(definition, definition.addPass("write"))
        .storageImageWrite(0, lr::ImageView::mip(image, 3), VK_SHADER_STAGE_COMPUTE_BIT)
        .storageBufferWrite(1, buffer, VK_SHADER_STAGE_COMPUTE_BIT);
    builder(definition, definition.addPass("read"))
        .sampledImage(0, image, VK_SHADER_STAGE_FRAGMENT_BIT)
        .storageBufferRead(1, buffer, VK_SHADER_STAGE_COMPUTE_BIT);

    require(definition.passes()[0].imageUses[0].boundMip == 3,
            "the selected mip should be retained for descriptor creation");
    const auto order = sort(definition);
    const auto plan  = lr::framegraph::planVulkanBarriers(definition.passes(), definition.resources(), order);
    require(plan.beforePass[1].size() == 2, "the reader should synchronize both parent resources");
    require(plan.finalImageLayouts.at("mipped") == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            "mip selection must not split whole-image layout state");
}

void sameLayoutAttachmentHazard()
{
    lr::FrameGraphDefinition definition;
    const auto               image = definition.image("swapchain");
    builder(definition, definition.addPass("final")).colorAttachment(image, VK_FORMAT_B8G8R8A8_UNORM);
    builder(definition, definition.addPass("imgui"))
        .colorAttachment(image, VK_FORMAT_B8G8R8A8_UNORM, VK_ATTACHMENT_LOAD_OP_LOAD);

    const auto order = sort(definition);
    const auto plan  = lr::framegraph::planVulkanBarriers(definition.passes(), definition.resources(), order);
    require(plan.beforePass[1].size() == 1, "attachment WAW should synchronize in the same layout");
    const auto &barrier = plan.beforePass[1][0];
    require(barrier.source.layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL &&
                barrier.destination.layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            "attachment WAW should preserve its layout");
    require((barrier.source.access & VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT) != 0 &&
                (barrier.destination.access & VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT) != 0 &&
                (barrier.destination.access & VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT) != 0,
            "LOAD should synchronize the prior write to attachment read/write access");
}

void readOnlyImageAccessesCoalesce()
{
    lr::FrameGraphDefinition definition;
    const auto               image = definition.image("texture");
    builder(definition, definition.addPass("first")).sampledImage(0, image, VK_SHADER_STAGE_FRAGMENT_BIT);
    builder(definition, definition.addPass("second")).sampledImage(0, image, VK_SHADER_STAGE_FRAGMENT_BIT);

    const auto order = sort(definition);
    const auto plan  = lr::framegraph::planVulkanBarriers(definition.passes(), definition.resources(), order,
                                                          {{"texture", VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}});
    require(plan.beforePass[0].empty() && plan.beforePass[1].empty(),
            "compatible read-only accesses should not emit barriers");
}

void sameLayoutStorageHazard()
{
    lr::FrameGraphDefinition definition;
    const auto               image = definition.image("storage");
    builder(definition, definition.addPass("first")).storageImageWrite(0, image, VK_SHADER_STAGE_COMPUTE_BIT);
    builder(definition, definition.addPass("second")).storageImageWrite(0, image, VK_SHADER_STAGE_COMPUTE_BIT);

    const auto order = sort(definition);
    const auto plan  = lr::framegraph::planVulkanBarriers(definition.passes(), definition.resources(), order);
    require(plan.beforePass[1].size() == 1, "storage WAW should synchronize in GENERAL");
    const auto &barrier = plan.beforePass[1][0];
    require(barrier.source.layout == VK_IMAGE_LAYOUT_GENERAL && barrier.destination.layout == VK_IMAGE_LAYOUT_GENERAL &&
                barrier.source.access == VK_ACCESS_2_SHADER_WRITE_BIT &&
                barrier.destination.access == VK_ACCESS_2_SHADER_WRITE_BIT,
            "storage WAW should connect shader writes without changing layout");
}

void bufferRawHazard()
{
    lr::FrameGraphDefinition definition;
    const auto               buffer = definition.buffer("values");
    builder(definition, definition.addPass("writer")).storageBufferWrite(0, buffer, VK_SHADER_STAGE_COMPUTE_BIT);
    builder(definition, definition.addPass("reader")).storageBufferRead(0, buffer, VK_SHADER_STAGE_FRAGMENT_BIT);

    const auto order = sort(definition);
    const auto plan  = lr::framegraph::planVulkanBarriers(definition.passes(), definition.resources(), order);
    require(plan.beforePass[1].size() == 1, "buffer RAW should emit a barrier");
    const auto &barrier = plan.beforePass[1][0];
    require(barrier.kind == lr::framegraph::BarrierResourceKind::Buffer &&
                barrier.source.stages == VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT &&
                barrier.destination.stages == VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT &&
                barrier.source.access == VK_ACCESS_2_SHADER_WRITE_BIT &&
                barrier.destination.access == VK_ACCESS_2_SHADER_READ_BIT,
            "buffer RAW should preserve compute-write to fragment-read intent");
}

void vertexShaderStageMapping()
{
    lr::FrameGraphDefinition definition;
    const auto               image = definition.image("displacement");
    builder(definition, definition.addPass("vertex sampler")).sampledImage(0, image, VK_SHADER_STAGE_VERTEX_BIT);

    const auto order = sort(definition);
    const auto plan  = lr::framegraph::planVulkanBarriers(definition.passes(), definition.resources(), order);
    require(plan.beforePass[0].size() == 1 &&
                plan.beforePass[0][0].destination.stages == VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
            "vertex sampling should synchronize to the vertex shader stage");
}

void renderingExtentPlanning()
{
    lr::FrameGraphDefinition definition;
    const auto               color = definition.image("color");
    const auto               depth = definition.image("depth");
    builder(definition, definition.addPass("half-resolution"))
        .colorAttachment(color, VK_FORMAT_R16G16B16A16_SFLOAT, VK_ATTACHMENT_LOAD_OP_CLEAR, {}, {800, 450})
        .depthAttachment(depth, VK_FORMAT_D32_SFLOAT, VK_ATTACHMENT_LOAD_OP_CLEAR, {}, {800, 450});

    const auto extents = lr::framegraph::planRenderingExtents(definition.passes(), {1600, 900});
    require(extents[0].width == 800 && extents[0].height == 450,
            "rendering extent should come from the pass attachments");

    definition.pass({0, definition.owner()}).imageUses[1].extent = {400, 225};
    requireThrowsContaining(
        [&] {
            (void)lr::framegraph::planRenderingExtents(definition.passes(), {1600, 900});
        },
        "different extents");
}

void zeroShaderStagesAreRejected()
{
    lr::FrameGraphDefinition definition;
    const auto               image = definition.image("texture");
    const auto               pass  = definition.addPass("invalid");
    requireThrowsContaining(
        [&] {
            builder(definition, pass).sampledImage(0, image, 0);
        },
        "requires a shader stage");
}

void topologyDumpIsStableAndSemantic()
{
    lr::FrameGraphDefinition definition;
    const auto               output = definition.image("output");
    builder(definition, definition.addPass("present")).colorAttachment(output, VK_FORMAT_B8G8R8A8_UNORM);
    const auto order = sort(definition);
    const auto dump  = lr::framegraph::dumpTopology(definition.passes(), definition.resources(), order);
    require(dump.find("present") != std::string::npos && dump.find("output") != std::string::npos,
            "dump should use stable pass and resource names");
}

} // namespace

int main()
{
    const std::vector<Test> tests = {
        {"declaration-order WAR", declarationOrderDefinesReadBeforeWrite},
        {"typed explicit dependencies", explicitDependenciesAreTyped},
        {"cycle detection", cycleDetection},
        {"duplicate pass names", duplicatePassNames},
        {"semantic attachment planning", attachmentPlanningUsesSemanticDeclarations},
        {"contradictory attachments", contradictoryAttachmentDeclarationsAreRejected},
        {"semantic translation", translationPreservesSemanticIntent},
        {"cross-definition handles", handlesCannotCrossDefinitions},
        {"whole-resource barriers", barriersRemainWholeResource},
        {"same-layout attachment hazard", sameLayoutAttachmentHazard},
        {"read-only image coalescing", readOnlyImageAccessesCoalesce},
        {"same-layout storage hazard", sameLayoutStorageHazard},
        {"buffer RAW hazard", bufferRawHazard},
        {"vertex shader stage", vertexShaderStageMapping},
        {"rendering extent planning", renderingExtentPlanning},
        {"zero shader stages", zeroShaderStagesAreRejected},
        {"stable topology dump", topologyDumpIsStableAndSemantic},
    };

    size_t failures = 0;
    for (const auto &[name, test] : tests)
    {
        try
        {
            test();
            std::cout << "[PASS] " << name << '\n';
        } catch (const std::exception &error)
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
