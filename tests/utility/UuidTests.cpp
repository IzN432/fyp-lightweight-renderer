#include "core/utility/Uuid.hpp"

#include <cassert>
#include <cctype>
#include <stdexcept>
#include <string>
#include <unordered_set>

int main()
{
    // Default is nil, and nil is the one value generateUuid() must never produce — fields holding
    // it mean "names nothing".
    const lr::Uuid nil;
    assert(nil.is_nil());
    assert(lr::toString(nil) == "00000000-0000-0000-0000-000000000000");

    const lr::Uuid generated = lr::generateUuid();
    assert(!generated.is_nil());
    assert(lr::toString(generated).size() == 36);
    assert(generated.version() == uuids::uuid_version::random_number_based);
    assert(generated.variant() == uuids::uuid_variant::rfc);

    // Round trips through the canonical text form unchanged, which is what persistence relies on.
    assert(lr::parseUuid(lr::toString(generated)) == generated);
    assert(lr::parseUuid(lr::toString(nil)) == nil);

    // Uppercase input is accepted; the output is always lowercase.
    std::string upper = lr::toString(generated);
    for (char &character : upper)
    {
        character = static_cast<char>(std::toupper(static_cast<unsigned char>(character)));
    }
    assert(lr::parseUuid(upper) == generated);

    // parseUuid throws rather than returning an empty optional, because a malformed identity in a
    // scene file is not something the loader can carry on past.
    const std::string malformed[] = {
        "",
        "not-a-uuid",
        "00000000-0000-0000-0000-00000000000",   // too short
        "00000000-0000-0000-0000-0000000000000", // too long
        "00000000+0000-0000-0000-000000000000",  // wrong separator
        "0000000g-0000-0000-0000-000000000000",  // not hex
    };
    for (const std::string &text : malformed)
    {
        bool rejected = false;
        try
        {
            (void)lr::parseUuid(text);
        } catch (const std::runtime_error &)
        {
            rejected = true;
        }
        assert(rejected);
    }

    // Distinct, and usable as a hash key — Scene and the asset stores index by these.
    std::unordered_set<lr::Uuid> seen;
    for (int i = 0; i < 1000; ++i)
    {
        seen.insert(lr::generateUuid());
    }
    assert(seen.size() == 1000);
    assert(std::hash<lr::Uuid>{}(generated) == std::hash<lr::Uuid>{}(lr::parseUuid(lr::toString(generated))));

    return 0;
}
