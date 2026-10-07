#pragma once

#include <uuid.h>

#include <cstddef>
#include <string>
#include <string_view>

namespace lr
{

// A random 128-bit identity, used wherever something has to stay recognizable across a save and a
// load. Default-constructed is nil, which names nothing: fields of this type start out holding "no
// identity" rather than accidentally naming whatever happens to live at index zero.
//
// Identities are generated, never derived from position, so two scenes saved from different
// sessions can be loaded side by side without their objects colliding.
using Uuid = uuids::uuid;

// A version 4 (random) identity. Never nil.
Uuid generateUuid();

// Parses the canonical 8-4-4-4-12 hex form, in either case. Throws std::runtime_error on anything
// else, since a malformed identity in a file is not something a caller can sensibly recover from
// mid-load — use uuids::uuid::from_string directly where a bad identity is an expected input.
Uuid parseUuid(std::string_view text);

std::string toString(const Uuid &id);

} // namespace lr
