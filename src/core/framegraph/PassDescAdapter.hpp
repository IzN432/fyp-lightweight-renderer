#pragma once

#include "PassBuilder.hpp"
#include "model/GraphDefinition.hpp"

#include <span>

namespace lr::framegraph
{

// Current C++/Vulkan frontend adapter. GraphDefinition itself has no dependency
// on PassDesc or Vulkan and can be constructed by other frontends directly.
GraphDefinition translatePassDescriptions(std::span<const PassDesc> passes);

} // namespace lr::framegraph
