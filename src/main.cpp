#include "app/Engine.hpp"

#include <spdlog/spdlog.h>

#include <exception>

int main()
try
{
    lr::Engine engine;
    engine.run();
    return 0;
} catch (const std::exception &e)
{
    spdlog::error("Fatal: {}", e.what());
    return 1;
}
