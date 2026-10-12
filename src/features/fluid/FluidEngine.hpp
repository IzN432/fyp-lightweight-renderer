#pragma once

namespace lr
{

// A standalone application for the fluid simulation: a camera, the CPU solver, particles drawn as
// spheres and the domain drawn as a wireframe box. Nothing else.
//
// This exists alongside lr::Engine rather than inside it. The editor engine assembles the full
// deferred pipeline — IBL, shadows, PBR, ambient occlusion, picking, outlines, transparency, the
// scene document and the inspector — none of which a fluid viewer needs. Building the sim on top
// of it meant inheriting the editor's overlay semantics, where particles were washed toward white
// by hover highlighting and tone mapped on the way to the screen. Here the configured colour is
// the colour that lands on the swapchain.
class FluidEngine
{
public:
    void run();
};

} // namespace lr
