#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace lr
{

enum class ShaderStage
{
    Vertex,
    Fragment,
    Compute,
};

// Thrown for GLSL that fails to parse or link; what() carries glslang's info log (file:line: message).
class ShaderCompileError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

// Runtime GLSL -> SPIR-V, for frontends (e.g. Python) that can't rely on the build-time glslc step.
// Targets Vulkan 1.0 / SPIR-V 1.0 — the same defaults glslc uses for the engine's own shaders — and
// enables GL_GOOGLE_include_directive implicitly as glslc does: `#include "x"` resolves relative to
// the including file first, then against includeDirs; `#include <x>` only against includeDirs.

// Stage is inferred from the extension (.vert / .frag / .comp) unless given explicitly.
std::vector<uint32_t> compileGlslFile(const std::filesystem::path              &path,
                                      std::optional<ShaderStage>                stage       = std::nullopt,
                                      const std::vector<std::filesystem::path> &includeDirs = {});

// `name` labels the source in error messages and anchors relative #includes (pass a file path to
// resolve includes next to it).
std::vector<uint32_t> compileGlslSource(std::string_view source, ShaderStage stage,
                                        const std::string                        &name        = "<source>",
                                        const std::vector<std::filesystem::path> &includeDirs = {});

} // namespace lr
