#pragma once

#include <cstdint>

namespace lr
{

// Opaque per-vertex classification id. SelectionManager stores a color against each id it hands
// out (see SelectionManager::registerRole) but never learns what the id means — that's up to
// whoever registered it (e.g. a feature defining "anchor"/"handle" roles for ARAP).
using VertexRoleId             = uint32_t;
constexpr VertexRoleId kNoRole = 0;

} // namespace lr
