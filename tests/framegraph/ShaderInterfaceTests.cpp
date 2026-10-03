// validateShaderInterface against passes built from runtime-compiled GLSL — no GPU needed.

#include "core/framegraph/FrameGraphDefinition.hpp"
#include "core/framegraph/PassBuilder.hpp"
#include "core/framegraph/compiler/ShaderInterface.hpp"
#include "core/vulkan/ShaderCompiler.hpp"

#include <functional>
#include <iostream>
#include <memory>
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

std::vector<uint32_t> vert(const std::string &body)
{
    return lr::compileGlslSource(body, lr::ShaderStage::Vertex, "t.vert");
}
std::vector<uint32_t> frag(const std::string &body)
{
    return lr::compileGlslSource(body, lr::ShaderStage::Fragment, "t.frag");
}

const std::string kFullscreenVert = "#version 450\nvoid main() { gl_Position = vec4(0.0); }\n";
const std::string kMeshVert       = "#version 450\n"
                                    "layout(location = 0) in vec3 inPosition;\n"
                                    "layout(location = 1) in vec3 inNormal;\n"
                                    "layout(set = 0, binding = 0) uniform Camera { mat4 viewProj; } camera;\n"
                                    "layout(push_constant) uniform PC { mat4 model; } pc;\n"
                                    "layout(location = 0) out vec3 outNormal;\n"
                                    "void main() { outNormal = inNormal;\n"
                                    "  gl_Position = camera.viewProj * pc.model * vec4(inPosition, 1.0); }\n";
const std::string kTexturedFrag   = "#version 450\n"
                                    "layout(location = 0) in vec3 inNormal;\n"
                                    "layout(location = 0) out vec4 outColor;\n"
                                    "layout(set = 0, binding = 1) uniform sampler2D albedo;\n"
                                    "void main() { outColor = texture(albedo, inNormal.xy); }\n";

struct Fixture
{
    lr::FrameGraphDefinition definition;
    lr::PassHandle           handle = definition.addPass("p");
    lr::PassBuilder          builder{definition, handle};

    const lr::PassDesc &pass() { return definition.pass(handle); }

    // A geometry pass whose declarations match kMeshVert + kTexturedFrag exactly.
    lr::PassBuilder &meshPass()
    {
        return builder.type(lr::PassType::Geometry)
            .vertShader(vert(kMeshVert))
            .fragShader(frag(kTexturedFrag))
            .vertexLayout({{0, 24, VK_VERTEX_INPUT_RATE_VERTEX}},
                          {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}, {1, 0, VK_FORMAT_R32G32B32_SFLOAT, 12}})
            .pushConstantSize(64, VK_SHADER_STAGE_VERTEX_BIT);
    }
};

std::string errorOf(const lr::PassDesc &pass)
{
    try
    {
        lr::validateShaderInterface(pass);
    } catch (const lr::ShaderInterfaceError &e)
    {
        return e.what();
    }
    return {};
}

void requireError(const lr::PassDesc &pass, const std::vector<std::string> &needles)
{
    const std::string message = errorOf(pass);
    require(!message.empty(), "expected a ShaderInterfaceError");
    for (const std::string &needle : needles)
    {
        require(message.find(needle) != std::string::npos, "error should contain '" + needle + "':\n" + message);
    }
}

void matchingPassIsAccepted()
{
    Fixture f;
    f.meshPass()
        .uniformBuffer(0, f.definition.buffer("camera"), VK_SHADER_STAGE_VERTEX_BIT)
        .sampledImage(1, f.definition.image("albedo"), VK_SHADER_STAGE_FRAGMENT_BIT);
    const std::string message = errorOf(f.pass());
    require(message.empty(), "a matching pass should validate, got:\n" + message);
}

void missingBindingNamesTheUnusedDeclaration()
{
    Fixture f;
    f.meshPass()
        .uniformBuffer(2, f.definition.buffer("camera"), VK_SHADER_STAGE_VERTEX_BIT)
        .sampledImage(1, f.definition.image("albedo"), VK_SHADER_STAGE_FRAGMENT_BIT);
    requireError(f.pass(), {"pass 'p'", "binding 0 ('camera', uniform buffer, vertex)", "not declared",
                            "no shader uses: 2 (uniform buffer)"});
}

void descriptorTypeMismatch()
{
    Fixture f;
    f.meshPass()
        .storageBufferRead(0, f.definition.buffer("camera"), VK_SHADER_STAGE_VERTEX_BIT)
        .sampledImage(1, f.definition.image("albedo"), VK_SHADER_STAGE_FRAGMENT_BIT);
    requireError(f.pass(), {"binding 0", "declared as a storage buffer"});
}

void stageMismatch()
{
    Fixture f;
    f.meshPass()
        .uniformBuffer(0, f.definition.buffer("camera"), VK_SHADER_STAGE_VERTEX_BIT)
        .sampledImage(1, f.definition.image("albedo"), VK_SHADER_STAGE_VERTEX_BIT);
    requireError(f.pass(), {"binding 1", "fragment", "declared only for vertex"});
}

void missingVertexAttribute()
{
    Fixture f;
    f.meshPass()
        .vertexLayout({{0, 12, VK_VERTEX_INPUT_RATE_VERTEX}}, {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}})
        .uniformBuffer(0, f.definition.buffer("camera"), VK_SHADER_STAGE_VERTEX_BIT)
        .sampledImage(1, f.definition.image("albedo"), VK_SHADER_STAGE_FRAGMENT_BIT);
    requireError(f.pass(), {"vertex input location 1 'inNormal'", "no attribute"});
}

void unusedDeclarationsAloneAreFine()
{
    Fixture f;
    f.builder.type(lr::PassType::Fullscreen)
        .vertShader(vert(kFullscreenVert))
        .fragShader(frag("#version 450\nlayout(location = 0) out vec4 c;\nvoid main() { c = vec4(1.0); }\n"))
        .uniformBuffer(5, f.definition.buffer("unused"), VK_SHADER_STAGE_FRAGMENT_BIT)
        .pushConstantSize(16, VK_SHADER_STAGE_FRAGMENT_BIT);
    require(errorOf(f.pass()).empty(), "declarations no shader uses shouldn't fail on their own");
}

// The vertex stage owns bytes [0, 64) and the fragment stage [64, 84) of one shared range, as in
// OverlayGeometryPass. The fragment block's reflected offset is 64; it must not be counted twice.
void pushConstantRangeSharedBetweenStages()
{
    const std::string sharedFrag = "#version 450\n"
                                   "layout(location = 0) out vec4 c;\n"
                                   "layout(push_constant) uniform PC {\n"
                                   "  layout(offset = 64) vec3 color; layout(offset = 76) float a;\n"
                                   "  layout(offset = 80) uint index; } pc;\n"
                                   "void main() { c = vec4(pc.color, pc.a + float(pc.index)); }\n";
    const auto        build      = [&](uint32_t size, VkShaderStageFlags stages) {
        auto f = std::make_unique<Fixture>();
        f->builder.type(lr::PassType::Geometry)
            .vertShader(vert("#version 450\nlayout(location = 0) in vec3 p;\n"
                             "layout(push_constant) uniform PC { mat4 model; } pc;\n"
                             "void main() { gl_Position = pc.model * vec4(p, 1.0); }\n"))
            .fragShader(frag(sharedFrag))
            .vertexLayout({{0, 12, VK_VERTEX_INPUT_RATE_VERTEX}}, {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}})
            .pushConstantSize(size, stages);
        return f;
    };

    const auto ok = build(84, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
    require(errorOf(ok->pass()).empty(),
            "an 84-byte range shared by both stages should validate:\n" + errorOf(ok->pass()));
    requireError(build(80, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)->pass(),
                 {"the shaders use 84 bytes, but push_constant_size is 80"});
    requireError(build(84, VK_SHADER_STAGE_VERTEX_BIT)->pass(),
                 {"push constants are used by vertex+fragment but declared only for vertex"});
}

void fullscreenPassesTakeNoVertexInput()
{
    Fixture f;
    f.builder.type(lr::PassType::Fullscreen)
        .vertShader(vert("#version 450\nlayout(location = 0) in vec2 uv;\n"
                         "void main() { gl_Position = vec4(uv, 0.0, 1.0); }\n"))
        .fragShader(frag("#version 450\nlayout(location = 0) out vec4 c;\nvoid main() { c = vec4(1.0); }\n"));
    requireError(f.pass(), {"location 0 'uv'", "fullscreen passes have no vertex input"});
}

void onlySetZeroExists()
{
    Fixture f;
    f.builder.type(lr::PassType::Compute)
        .computeShader(lr::compileGlslSource("#version 450\nlayout(local_size_x = 1) in;\n"
                                             "layout(set = 1, binding = 0) buffer B { float v[]; } b;\n"
                                             "void main() { b.v[0] = 1.0; }\n",
                                             lr::ShaderStage::Compute),
                       "t.comp");
    requireError(f.pass(), {"t.comp uses set 1", "use set = 0"});
}

void everyProblemIsReportedAtOnce()
{
    Fixture f;
    f.meshPass().pushConstantSize(0, VK_SHADER_STAGE_VERTEX_BIT); // and no descriptors declared at all
    requireError(f.pass(), {"binding 0", "binding 1", "push constants"});
}

} // namespace

int main()
{
    const std::vector<Test> tests = {
        {"matching pass is accepted", matchingPassIsAccepted},
        {"missing binding names the unused declaration", missingBindingNamesTheUnusedDeclaration},
        {"descriptor type mismatch", descriptorTypeMismatch},
        {"stage mismatch", stageMismatch},
        {"missing vertex attribute", missingVertexAttribute},
        {"unused declarations alone are fine", unusedDeclarationsAloneAreFine},
        {"push-constant range shared between stages", pushConstantRangeSharedBetweenStages},
        {"fullscreen passes take no vertex input", fullscreenPassesTakeNoVertexInput},
        {"only set 0 exists", onlySetZeroExists},
        {"every problem is reported at once", everyProblemIsReportedAtOnce},
    };

    size_t failures = 0;
    for (const auto &[name, test] : tests)
    {
        try
        {
            test();
            std::cout << "[PASS] " << name << '\n';
        } catch (const std::exception &e)
        {
            ++failures;
            std::cout << "[FAIL] " << name << ": " << e.what() << '\n';
        }
    }
    std::cout << (tests.size() - failures) << " test(s) passed\n";
    return failures == 0 ? 0 : 1;
}
