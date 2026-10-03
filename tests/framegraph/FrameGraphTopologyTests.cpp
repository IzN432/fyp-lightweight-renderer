#include "core/framegraph/FrameGraphDefinition.hpp"
#include "core/framegraph/ExternalImageBindings.hpp"
#include "core/framegraph/FrameGraphTopology.hpp"
#include "core/framegraph/PassBuilder.hpp"
#include "core/framegraph/PassDescAdapter.hpp"
#include "core/framegraph/compiler/VulkanBarrierPlanner.hpp"
#include "core/vulkan/VkFormatUtils.hpp"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
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

void symbolicExtentResolution()
{
    const VkExtent2D base{1601, 901};
    const VkExtent2D full  = lr::ExtentSpec::swapchain().resolve(base);
    const VkExtent2D half  = lr::ExtentSpec::relative(1, 2).resolve(base);
    const VkExtent2D fixed = lr::ExtentSpec::absolute(2048, 1024).resolve(base);

    require(full.width == 1601 && full.height == 901, "swapchain extent should resolve unchanged");
    require(half.width == 801 && half.height == 451, "relative extents should round odd dimensions up");
    require(fixed.width == 2048 && fixed.height == 1024, "absolute extents should ignore the swapchain");
    require(lr::ExtentSpec::relative(2, 4) == lr::ExtentSpec::relative(1, 2),
            "equivalent relative ratios should have one canonical representation");
    requireThrowsContaining(
        [] {
            (void)lr::ExtentSpec::relative(1, 0);
        },
        "non-zero");
}

void depthFormatClassificationIsSharedAndComplete()
{
    require(lr::isDepthFormat(VK_FORMAT_D16_UNORM), "D16 should be classified as depth");
    require(lr::isDepthFormat(VK_FORMAT_X8_D24_UNORM_PACK32), "packed D24 should be classified as depth");
    require(lr::isDepthFormat(VK_FORMAT_D32_SFLOAT), "D32 should be classified as depth");
    require(lr::isDepthFormat(VK_FORMAT_D16_UNORM_S8_UINT), "D16S8 should be classified as depth-stencil");
    require(lr::isDepthFormat(VK_FORMAT_D24_UNORM_S8_UINT), "D24S8 should be classified as depth-stencil");
    require(lr::isDepthFormat(VK_FORMAT_D32_SFLOAT_S8_UINT), "D32S8 should be classified as depth-stencil");
    require(!lr::isDepthFormat(VK_FORMAT_R16G16B16A16_SFLOAT), "a color format must not be classified as depth");
}

void attachmentPlanningUsesSemanticDeclarations()
{
    lr::FrameGraphDefinition definition;
    const auto               color = definition.image("color");
    const auto               depth = definition.image("depth");
    builder(definition, definition.addPass("geometry"))
        .colorAttachment(color, VK_FORMAT_R16G16B16A16_SFLOAT)
        .depthAttachment(depth, VK_FORMAT_D32_SFLOAT, VK_ATTACHMENT_LOAD_OP_CLEAR, {},
                         lr::ExtentSpec::absolute(800, 600));
    builder(definition, definition.addPass("overlay"))
        .colorAttachment(color, VK_FORMAT_R16G16B16A16_SFLOAT, VK_ATTACHMENT_LOAD_OP_LOAD);
    builder(definition, definition.addPass("post-process")).storageImageWrite(0, color, VK_SHADER_STAGE_COMPUTE_BIT);

    const auto images = lr::framegraph::planAttachmentImages(definition.passes(), definition.resources());
    require(images.size() == 2, "repeated attachment declarations should plan one image");
    require(images[0].name == "color" && images[0].extent == lr::ExtentSpec::swapchain(),
            "default extent should remain swapchain-relative");
    require((images[0].usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) != 0 &&
                (images[0].usage & VK_IMAGE_USAGE_STORAGE_BIT) != 0,
            "image creation flags should aggregate every declared use");
    require(images[1].name == "depth" && images[1].extent == lr::ExtentSpec::absolute(800, 600),
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
            (void)lr::framegraph::planAttachmentImages(definition.passes(), definition.resources());
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

void definitionSnapshotsPreserveHandleIdentityAndCallbacks()
{
    lr::FrameGraphDefinition definition;
    const auto               image  = definition.image("snapshot-output");
    bool                     called = false;
    builder(definition, definition.addPass("snapshot-pass"))
        .colorAttachment(image, VK_FORMAT_R8G8B8A8_UNORM)
        .execute([&](lr::PassContext &) {
            called = true;
        });

    const lr::FrameGraphDefinition snapshot = definition;
    (void)definition.addPass("later-pass");
    require(snapshot.owner() == definition.owner(), "a definition snapshot must preserve handle ownership");
    require(snapshot.name(image) == "snapshot-output", "existing handles must resolve in a definition snapshot");
    require(snapshot.passes().size() == 1 && static_cast<bool>(snapshot.passes()[0].executeCallback),
            "a definition snapshot must retain callbacks without observing later frontend edits");

    (void)called;
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

void blendingReadsColorAttachments()
{
    // blend() may come before or after colorAttachment(); either way the attachment is read, even with CLEAR.
    for (const bool blendFirst : {true, false})
    {
        lr::FrameGraphDefinition definition;
        const auto               image = definition.image("color");
        builder(definition, definition.addPass("opaque")).colorAttachment(image, VK_FORMAT_R16G16B16A16_SFLOAT);
        auto blended = builder(definition, definition.addPass("blended"));
        if (blendFirst)
        {
            blended.blend(lr::BlendMode::Alpha).colorAttachment(image, VK_FORMAT_R16G16B16A16_SFLOAT);
        } else
        {
            blended.colorAttachment(image, VK_FORMAT_R16G16B16A16_SFLOAT).blend(lr::BlendMode::Alpha);
        }

        const auto order = sort(definition);
        require(order == std::vector<size_t>({0, 1}), "a blended pass should run after the attachment's writer");
        const auto plan = lr::framegraph::planVulkanBarriers(definition.passes(), definition.resources(), order);
        require(plan.beforePass[1].size() == 1 &&
                    (plan.beforePass[1][0].destination.access & VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT) != 0,
                std::string("blending should synchronize to attachment read access (blend ") +
                    (blendFirst ? "before" : "after") + " colorAttachment)");
    }
}

void runsLastOrdersAfterLaterDeclaredPasses()
{
    // An overlay declared early (like the Viewer's ImGui pass) must still follow a pass added later that
    // writes the same image — without the declaration-order WAW edge turning into a cycle.
    lr::FrameGraphDefinition definition;
    const auto               image = definition.image("swapchain");
    builder(definition, definition.addPass("scene")).colorAttachment(image, VK_FORMAT_B8G8R8A8_UNORM);
    builder(definition, definition.addPass("overlay"))
        .runsLast()
        .colorAttachment(image, VK_FORMAT_B8G8R8A8_UNORM, VK_ATTACHMENT_LOAD_OP_LOAD);
    builder(definition, definition.addPass("added_later"))
        .colorAttachment(image, VK_FORMAT_B8G8R8A8_UNORM, VK_ATTACHMENT_LOAD_OP_LOAD);
    builder(definition, definition.addPass("unrelated"))
        .storageBufferWrite(0, definition.buffer("b"), VK_SHADER_STAGE_COMPUTE_BIT);

    require(sort(definition) == std::vector<size_t>({0, 2, 3, 1}),
            "a runsLast pass should execute after every other pass, including later-declared ones");
}

void definitionRevisionTracksChanges()
{
    lr::FrameGraphDefinition definition;
    const uint64_t           initial = definition.revision();
    const auto               pass    = definition.addPass("p");
    require(definition.revision() > initial, "adding a pass should change the revision");

    const uint64_t afterAdd = definition.revision();
    (void)std::as_const(definition).pass(pass);
    (void)std::as_const(definition).passes();
    require(definition.revision() == afterAdd, "read-only access should not change the revision");

    builder(definition, pass).type(lr::PassType::Compute);
    require(definition.revision() > afterAdd, "modifying a pass through its builder should change the revision");
}

void backbufferContractPlansPresentationTransitions()
{
    lr::FrameGraphDefinition definition;
    const auto               backbuffer = definition.importBackbuffer("backbuffer", VK_FORMAT_B8G8R8A8_UNORM);
    builder(definition, definition.addPass("display")).colorAttachment(backbuffer, VK_FORMAT_B8G8R8A8_UNORM);

    require(definition.externalImages().size() == 1 && definition.externalImages()[0].image == backbuffer,
            "the backbuffer should be retained as an opaque external image slot");

    const auto order = sort(definition);
    const auto plan  = lr::framegraph::planVulkanBarriers(definition.passes(), definition.resources(), order,
                                                          {{"backbuffer", VK_IMAGE_LAYOUT_UNDEFINED}},
                                                          {{"backbuffer", VK_IMAGE_LAYOUT_PRESENT_SRC_KHR}});

    require(plan.beforePass[0].size() == 1,
            "backbuffer rendering should transition from its discard state before the first use");
    require(plan.beforePass[0][0].source.layout == VK_IMAGE_LAYOUT_UNDEFINED &&
                plan.beforePass[0][0].destination.layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            "the initial backbuffer transition should discard into COLOR_ATTACHMENT");
    require(plan.afterGraph.size() == 1 &&
                plan.afterGraph[0].source.layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL &&
                plan.afterGraph[0].destination.layout == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            "the graph should return the backbuffer to PRESENT after its final use");
    require(plan.finalImageLayouts.at("backbuffer") == VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            "the exported backbuffer state should be PRESENT");
}

void externalBindingsEnforceHandleOwnership()
{
    lr::FrameGraphDefinition first;
    lr::FrameGraphDefinition second;
    const auto               firstBackbuffer  = first.importBackbuffer("backbuffer", VK_FORMAT_B8G8R8A8_UNORM);
    const auto               secondBackbuffer = second.importBackbuffer("backbuffer", VK_FORMAT_B8G8R8A8_UNORM);

    lr::ExternalImageBindings bindings;
    bindings.bind(firstBackbuffer, reinterpret_cast<VkImage>(1), reinterpret_cast<VkImageView>(2));
    require(bindings.find(firstBackbuffer) != nullptr, "a bound external image should resolve by typed handle");
    requireThrowsContaining(
        [&] {
            bindings.bind(secondBackbuffer, reinterpret_cast<VkImage>(3), reinterpret_cast<VkImageView>(4));
        },
        "different graphs");
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
        .colorAttachment(color, VK_FORMAT_R16G16B16A16_SFLOAT, VK_ATTACHMENT_LOAD_OP_CLEAR, {},
                         lr::ExtentSpec::relative(1, 2))
        .depthAttachment(depth, VK_FORMAT_D32_SFLOAT, VK_ATTACHMENT_LOAD_OP_CLEAR, {}, lr::ExtentSpec::relative(1, 2));

    const auto extents = lr::framegraph::planRenderingExtents(definition.passes(), {1601, 901});
    require(extents[0].width == 801 && extents[0].height == 451,
            "rendering extent should come from the pass attachments");

    definition.pass({0, definition.owner()}).imageUses[1].extent = lr::ExtentSpec::absolute(801, 451);
    requireThrowsContaining(
        [&] {
            (void)lr::framegraph::planRenderingExtents(definition.passes(), {1601, 901});
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
        {"symbolic extent resolution", symbolicExtentResolution},
        {"shared depth format classification", depthFormatClassificationIsSharedAndComplete},
        {"semantic attachment planning", attachmentPlanningUsesSemanticDeclarations},
        {"contradictory attachments", contradictoryAttachmentDeclarationsAreRejected},
        {"semantic translation", translationPreservesSemanticIntent},
        {"cross-definition handles", handlesCannotCrossDefinitions},
        {"definition snapshots", definitionSnapshotsPreserveHandleIdentityAndCallbacks},
        {"whole-resource barriers", barriersRemainWholeResource},
        {"same-layout attachment hazard", sameLayoutAttachmentHazard},
        {"blending reads color attachments", blendingReadsColorAttachments},
        {"runsLast orders after later-declared passes", runsLastOrdersAfterLaterDeclaredPasses},
        {"definition revision tracks changes", definitionRevisionTracksChanges},
        {"backbuffer presentation contract", backbufferContractPlansPresentationTransitions},
        {"external binding ownership", externalBindingsEnforceHandleOwnership},
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
