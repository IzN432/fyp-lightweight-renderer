# Python frame-graph bindings — work log

Branch `python-bindings` (worktree `.claude/worktrees/python-bindings`, based on `main` @ 3491a37).

**Goal:** let a user build their own renderer in Python on top of the `lr` frame graph — create
buffers/images, fill them from numpy, write GLSL, declare passes, and run them in the `Viewer`,
with the frame graph handling barriers, layouts and transient-image allocation.

Each step ends with a checkpoint for review. Status: ⬜ not started · 🟨 in progress · ✅ done (awaiting review) · ☑️ reviewed

---

## Step 1a — C++ prerequisites ✅

**Goal:** remove the C++-only assumptions that block a Python frontend, without changing any
existing pass's behaviour.

- Runtime GLSL → SPIR-V compilation (glslang, `#include` support): `compileGlslFile` / `compileGlslSource`.
- `ShaderCode` (a `.spv` path *or* in-memory SPIR-V) flows from `PassBuilder` → pipelines → `ShaderModule`;
  `PassBuilder::vertShader/fragShader/computeShader` accept SPIR-V words.
- `PassBuilder::vertexLayout(bindings, attributes)` — raw vertex layout, no `Mesh`/`GpuMeshLayout` needed.
- `CommandBuffer::pushConstants(layout, stages, const void*, size, offset)` — untyped push constants.

**Demo:**
- `ctest -C Debug -R shader_compiler` — compiles every shader in `src/core/passes` from GLSL source at runtime
  and checks the output is valid SPIR-V.
- `build/Debug/renderer.exe` runs and looks exactly as before (existing passes still use build-time `.spv`).

**Result (2026-10-04):**
- `ctest -C Debug` — 8/8 pass (7 existing + new `vulkan.shader_compiler`).
- All 20 engine shaders compile at runtime; `spirv-val --target-env vulkan1.0` accepts all 20 outputs
  (`shader_compiler_tests.exe <outDir>` writes them out for this).
- `renderer.exe` ran for 12 s without errors. Its single validation warning (location 2 / `inTangent` not consumed
  by `geometry.vert`) was already there before this change.
- The C++ API is ready, but it isn't wired to anything new yet. The first visible use is the Python demo in 1b.

**Files:** new `core/vulkan/ShaderCompiler.{hpp,cpp}`, `tests/vulkan/ShaderCompilerTests.cpp`; `ShaderCode` in
`core/vulkan/ShaderLoader.hpp`; `PassDesc` / `PassBuilder` / `GraphicsPipeline` / `ComputePipeline` /
`FrameGraphCompiler` carry `ShaderCode`; `CommandBuffer::pushConstants(void*)`; `glslang` in `vcpkg.json`.

## Step 1b — minimal Python module ⬜

**Goal:** a nanobind module `lr` exposing `Viewer`, `ResourceRegistry`, `FrameGraph`, `PassBuilder`,
`PassContext`, `CommandBuffer`, curated Vulkan enums, and numpy-backed uploads.

**Demo:** `python examples/python/triangle.py` opens a window and draws a mesh supplied as numpy arrays,
using GLSL written next to the script, with a camera matrix updated from Python each frame.
A port of `HeatmapPass`/`FinalPass` to Python produces the same image as the C++ versions.

## Step 2 — pipeline state on `PassBuilder` ⬜

**Goal:** blend, cull, depth test/write/compare and polygon mode configurable per pass (from C++ and Python).

**Demo:** a Python script draws overlapping alpha-blended quads and a wireframe mesh over a solid one.

## Step 3 — shader reflection and Python-friendly errors ⬜

**Goal:** validate declared bindings and push-constant sizes against the shader's SPIR-V (SPIRV-Reflect) at
`compile()`, and surface validation-layer errors as Python exceptions in debug builds.

**Demo:** a script that declares `uniform_buffer(1, ...)` for a shader expecting binding 0 raises a Python
exception naming the pass and binding, and the interpreter keeps running.

## Step 4 — runtime robustness ⬜

**Goal:** deferred destruction for replaced buffers/images (no manual `wait_idle`), automatic recompile when
the graph changes, indirect draws, buffer readback.

**Demo:** a compute pass whose output is read back into numpy and checked with `np.allclose`; a script that
resizes a vertex buffer every frame and adds a pass while running; 10k instances drawn via one indirect call.

---

## Notes / decisions

- Worktree was created from local `main` HEAD (9 commits ahead of `origin/main`). The uncommitted ImGuizmo
  gizmo work in the main checkout is **not** in this branch; expect a small merge in `CMakeLists.txt`.
- Runtime shader compilation targets Vulkan 1.0 / SPIR-V 1.0 — the same defaults `glslc` uses at build time.
  Like glslc, it enables `GL_GOOGLE_include_directive` implicitly (engine shaders `#include` without declaring it),
  and sources without `#version` default to 450.
- glslang is linked `PRIVATE` to `lr_core`; only `ShaderCompiler.hpp` (no glslang types) is public.
- Changes are left uncommitted on the `python-bindings` branch for review.
