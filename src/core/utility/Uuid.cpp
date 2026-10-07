#include "core/utility/Uuid.hpp"

#include <random>
#include <stdexcept>

namespace lr
{
namespace
{
// stduuid lays out the identity but leaves the entropy to the caller, so it takes a generator.
// std::random_device is the operating system's own source — on MSVC a CSPRNG — which means there is
// no pseudo-random engine to seed here and no seed quality to reason about. Identities are minted
// when an object, mesh or material is created, never per frame, so drawing from the OS each time
// costs nothing that matters.
uuids::basic_uuid_random_generator<std::random_device> &generator()
{
    static thread_local std::random_device                                     device;
    static thread_local uuids::basic_uuid_random_generator<std::random_device> uuidGenerator{device};
    return uuidGenerator;
}
} // namespace

Uuid generateUuid() { return generator()(); }

Uuid parseUuid(std::string_view text)
{
    const std::optional<Uuid> parsed = Uuid::from_string(text);
    if (!parsed)
    {
        throw std::runtime_error("Uuid: '" + std::string(text) + "' is not a UUID");
    }
    return *parsed;
}

std::string toString(const Uuid &id) { return uuids::to_string(id); }

} // namespace lr
