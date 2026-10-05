#include "ArapBackend.hpp"

#if defined(LR_ARAP_BACKEND_LIBIGL)
#include "LibiglArapBackend.hpp"
#elif defined(LR_ARAP_BACKEND_PARDISO)
#include "PardisoArapBackend.hpp"
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
#elif defined(LR_ARAP_BACKEND_PARDISO)
    return std::make_unique<PardisoArapBackend>();
#endif
}

} // namespace lr
