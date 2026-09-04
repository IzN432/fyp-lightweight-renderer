#pragma once

#include "PassDefinition.hpp"
#include "model/GraphDefinition.hpp"

#include <span>

namespace lr::framegraph
{

GraphDefinition translatePassDescriptions(std::span<const PassDesc> passes, const ResourceHandleRegistry &resources,
                                          uint64_t passOwner);

} // namespace lr::framegraph
