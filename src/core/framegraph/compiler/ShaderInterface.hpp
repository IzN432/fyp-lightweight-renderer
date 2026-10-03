#pragma once

#include "core/framegraph/PassDefinition.hpp"

#include <stdexcept>

namespace lr
{

// Thrown when a pass's declarations don't match what its shaders expect. what() names the pass and
// lists every mismatch, so one compile reports them all.
class ShaderInterfaceError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

// Reflects the pass's SPIR-V and checks it against its declarations, before any Vulkan object is
// created from them:
//  - every descriptor a shader uses is declared at the same binding, with the matching descriptor
//    type, for (at least) every stage that uses it; the frame graph has a single set (set 0);
//  - every used push-constant block fits pushConstantSize, declared for the stages that use it;
//  - every vertex-shader input location has a vertex attribute (Fullscreen passes take none).
// Only resources a shader statically uses count — the same rule Vulkan applies. Custom passes,
// which have no shaders, are skipped.
void validateShaderInterface(const PassDesc &pass);

} // namespace lr
