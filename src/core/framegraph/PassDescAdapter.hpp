#pragma once

#include "PassBuilder.hpp"
#include "model/GraphDefinition.hpp"

#include <span>

namespace lr::framegraph
{

// Current C++/Vulkan frontend adapter. GraphDefinition itself has no dependency
// on PassDesc or Vulkan and can be constructed by other frontends directly.
GraphDefinition translatePassDescriptions(std::span<const PassDesc> passes);

// Resolves typed frontend handles into the legacy string fields still consumed
// by the Vulkan backend. Remove this bridge once semantic resource declarations
// replace PassDesc in step 6.
void resolveTypedHandles(std::span<PassDesc> passes, const ResourceHandleRegistry &resources, uint64_t passOwner);

} // namespace lr::framegraph
