#include "ShaderCompiler.hpp"

#include <glslang/Public/ResourceLimits.h>
#include <glslang/Public/ShaderLang.h>
#include <glslang/SPIRV/GlslangToSpv.h>

#include <fstream>
#include <iterator>

namespace fs = std::filesystem;

namespace lr
{

namespace
{

// glslang needs one process-wide InitializeProcess() before any TShader is built; the matching
// FinalizeProcess() runs at static destruction.
struct GlslangProcess
{
    GlslangProcess() { glslang::InitializeProcess(); }
    ~GlslangProcess() { glslang::FinalizeProcess(); }
};

void ensureGlslangInitialized() { static GlslangProcess process; }

EShLanguage toEShLanguage(ShaderStage stage)
{
    switch (stage)
    {
        case ShaderStage::Vertex:
            return EShLangVertex;
        case ShaderStage::Fragment:
            return EShLangFragment;
        case ShaderStage::Compute:
            return EShLangCompute;
    }
    throw std::invalid_argument("compileGlsl: unknown shader stage");
}

ShaderStage stageFromExtension(const fs::path &path)
{
    const std::string ext = path.extension().string();
    if (ext == ".vert")
    {
        return ShaderStage::Vertex;
    }
    if (ext == ".frag")
    {
        return ShaderStage::Fragment;
    }
    if (ext == ".comp")
    {
        return ShaderStage::Compute;
    }
    throw std::invalid_argument("compileGlslFile: can't infer shader stage from extension of " + path.string() +
                                " (expected .vert, .frag or .comp) — pass the stage explicitly");
}

std::string readTextFile(const fs::path &path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("compileGlslFile: could not open " + path.string());
    }
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

// Same lookup order as glslc: quoted includes try the including file's directory first, then the
// include dirs; angle-bracket includes only the include dirs.
class FileIncluder : public glslang::TShader::Includer
{
public:
    explicit FileIncluder(const std::vector<fs::path> &includeDirs) : m_includeDirs(includeDirs) {}

    IncludeResult *includeLocal(const char *headerName, const char *includerName, size_t depth) override
    {
        const fs::path includer(includerName ? includerName : "");
        if (!includer.empty())
        {
            if (IncludeResult *result = tryOpen(includer.parent_path() / headerName))
            {
                return result;
            }
        }
        return includeSystem(headerName, includerName, depth);
    }

    IncludeResult *includeSystem(const char *headerName, const char *, size_t) override
    {
        for (const fs::path &dir : m_includeDirs)
        {
            if (IncludeResult *result = tryOpen(dir / headerName))
            {
                return result;
            }
        }
        return nullptr;
    }

    void releaseInclude(IncludeResult *result) override
    {
        if (result)
        {
            delete static_cast<std::string *>(result->userData);
            delete result;
        }
    }

private:
    static IncludeResult *tryOpen(const fs::path &path)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open())
        {
            return nullptr;
        }
        auto *content = new std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        return new IncludeResult(path.lexically_normal().string(), content->data(), content->size(), content);
    }

    const std::vector<fs::path> &m_includeDirs;
};

} // namespace

std::vector<uint32_t> compileGlslFile(const fs::path &path, std::optional<ShaderStage> stage,
                                      const std::vector<fs::path> &includeDirs)
{
    return compileGlslSource(readTextFile(path), stage.value_or(stageFromExtension(path)), path.string(), includeDirs);
}

std::vector<uint32_t> compileGlslSource(std::string_view source, ShaderStage stage, const std::string &name,
                                        const std::vector<fs::path> &includeDirs)
{
    ensureGlslangInitialized();

    const EShLanguage language = toEShLanguage(stage);
    // Used only when the source has no #version line.
    constexpr int     defaultVersion = 450;
    const EShMessages messages       = static_cast<EShMessages>(EShMsgSpvRules | EShMsgVulkanRules);

    const char *strings[] = {source.data()};
    const int   lengths[] = {static_cast<int>(source.size())};
    const char *names[]   = {name.c_str()};

    glslang::TShader shader(language);
    shader.setStringsWithLengthsAndNames(strings, lengths, names, 1);
    // glslc enables #include implicitly; the engine's shaders rely on that.
    shader.setPreamble("#extension GL_GOOGLE_include_directive : enable\n");
    shader.setEnvInput(glslang::EShSourceGlsl, language, glslang::EShClientVulkan, 100);
    shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_0);
    shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_0);

    FileIncluder includer(includeDirs);
    if (!shader.parse(GetDefaultResources(), defaultVersion, false, messages, includer))
    {
        throw ShaderCompileError("GLSL compile failed for " + name + ":\n" + shader.getInfoLog());
    }

    glslang::TProgram program;
    program.addShader(&shader);
    if (!program.link(messages))
    {
        throw ShaderCompileError("GLSL link failed for " + name + ":\n" + program.getInfoLog());
    }

    std::vector<uint32_t> spirv;
    spv::SpvBuildLogger   logger;
    glslang::SpvOptions   options;
    glslang::GlslangToSpv(*program.getIntermediate(language), spirv, &logger, &options);
    return spirv;
}

} // namespace lr
