#include "ArapBackend.hpp"

#if defined(LR_ARAP_BACKEND_LIBIGL)
#include "LibiglArapBackend.hpp"
#else
#error "No ARAP backend selected"
#endif

#include <memory>

namespace lr
{

std::unique_ptr<ArapBackend> createArapBackend()
{
#if defined(LR_ARAP_BACKEND_LIBIGL)
    return std::make_unique<LibiglArapBackend>();
#endif
}

} // namespace lr
