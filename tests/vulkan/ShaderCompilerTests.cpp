#include "core/vulkan/ShaderCompiler.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

namespace
{

constexpr uint32_t kSpirvMagic = 0x07230203;

int g_failures = 0;

void check(bool condition, const std::string &message)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        ++g_failures;
    }
}

} // namespace

// Usage: shader_compiler_tests [outDir] — when outDir is given, each compiled shader is also written
// there as <name>.spv (e.g. to run spirv-val over the results).
int main(int argc, char **argv)
{
    const fs::path shaderRoot = LR_SHADER_SOURCE_DIR;
    const fs::path outDir     = argc > 1 ? fs::path(argv[1]) : fs::path();

    // Every engine shader compiles from source at runtime, including ones using #include.
    int compiled = 0;
    for (const auto &entry : fs::recursive_directory_iterator(shaderRoot))
    {
        const std::string ext = entry.path().extension().string();
        if (ext != ".vert" && ext != ".frag" && ext != ".comp")
        {
            continue;
        }
        try
        {
            const std::vector<uint32_t> spirv = lr::compileGlslFile(entry.path());
            check(spirv.size() > 5 && spirv[0] == kSpirvMagic, "invalid SPIR-V header for " + entry.path().string());
            if (!outDir.empty())
            {
                std::ofstream out(outDir / (entry.path().filename().string() + ".spv"), std::ios::binary);
                out.write(reinterpret_cast<const char *>(spirv.data()),
                          static_cast<std::streamsize>(spirv.size() * sizeof(uint32_t)));
            }
            ++compiled;
        } catch (const std::exception &e)
        {
            check(false, e.what());
        }
    }
    check(compiled >= 20, "expected to compile every engine shader, compiled " + std::to_string(compiled));

    // Source without #version uses the 450 default; stage is explicit.
    const std::string compute = "layout(local_size_x = 1) in;\n"
                                "layout(set = 0, binding = 0) buffer B { float v[]; };\n"
                                "void main() { v[gl_GlobalInvocationID.x] *= 2.0; }\n";
    check(lr::compileGlslSource(compute, lr::ShaderStage::Compute)[0] == kSpirvMagic, "inline compute source");

    // Errors carry the source name and line.
    try
    {
        lr::compileGlslSource("#version 450\nvoid main() { undefinedThing = 1; }\n", lr::ShaderStage::Fragment,
                              "broken.frag");
        check(false, "broken shader should throw");
    } catch (const lr::ShaderCompileError &e)
    {
        const std::string message = e.what();
        check(message.find("broken.frag") != std::string::npos, "error should name the source: " + message);
        check(message.find("undefinedThing") != std::string::npos, "error should name the symbol: " + message);
    }

    // Angle-bracket includes resolve against includeDirs only.
    const std::string withInclude = "#version 450\n#include <utility/geometry.glslh>\nvoid main() {}\n";
    check(lr::compileGlslSource(withInclude, lr::ShaderStage::Fragment, "inc.frag", {shaderRoot})[0] == kSpirvMagic,
          "include via includeDirs");
    try
    {
        lr::compileGlslSource(withInclude, lr::ShaderStage::Fragment, "inc.frag");
        check(false, "missing include should throw");
    } catch (const lr::ShaderCompileError &)
    {}

    // Unknown extensions need an explicit stage.
    try
    {
        lr::compileGlslFile(shaderRoot / "utility" / "geometry.glslh");
        check(false, "unknown extension should throw");
    } catch (const std::invalid_argument &)
    {}

    std::cout << "compiled " << compiled << " engine shaders, " << g_failures << " failure(s)\n";
    return g_failures == 0 ? 0 : 1;
}
