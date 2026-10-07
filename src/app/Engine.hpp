#pragma once

namespace lr
{

// Owns the native renderer application's composition and lifetime. The reusable rendering pieces
// remain in lr_core/lr.engine; this class assembles the opinionated desktop editor around them.
class Engine
{
public:
    void run();
};

} // namespace lr
