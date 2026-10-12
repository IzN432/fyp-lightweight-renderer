#include "features/fluid/FluidEngine.hpp"

#include <spdlog/spdlog.h>

#include <exception>

int main()
try
{
    lr::FluidEngine engine;
    engine.run();
    return 0;
} catch (const std::exception &e)
{
    spdlog::error("Fatal: {}", e.what());
    return 1;
}
