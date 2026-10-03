# Python frame-graph bindings — work log

Branch `python-bindings` (worktree `.claude/worktrees/python-bindings`, based on `main` @ 3491a37).

**Goal:** let a user build their own renderer in Python on top of the `lr` frame graph — create
buffers/images, fill them from numpy, write GLSL, declare passes, and run them in the `Viewer`,
with the frame graph handling barriers, layouts and transient-image allocation.

Each step ends with a checkpoint for review. Status: ⬜ not started · 🟨 in progress · ✅ done (awaiting review) · ☑️ reviewed

---

## Step 1a — C++ prerequisites ☑️ (committed 1142749)

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

## Step 1b — minimal Python module ✅

**Goal:** a nanobind module `lr` exposing `Viewer`, `ResourceRegistry`, `FrameGraph`, `PassBuilder`,
`PassContext`, `CommandBuffer`, curated Vulkan enums, and numpy-backed uploads.

**Demo:**
```
pip install nanobind            # once; CMake skips the module (with a warning) if it's missing
cmake --build build --config Debug
PYTHONPATH=build/python python examples/python/spinning_torus.py
```
Opens a window with a tumbling torus. It's a two-pass renderer written entirely in Python
([examples/python/spinning_torus.py](examples/python/spinning_torus.py)):
- **"torus"** draws a mesh generated with numpy into an HDR target, using GLSL from
  `examples/python/shaders/`. It reads a camera UBO updated from Python each frame and takes the model
  matrix as push constants.
- **"post"** samples that target, tone-maps it and writes to the swapchain. It reuses the engine's
  prebuilt `fullscreen.vert.spv`.
The attachments are never created by hand: the frame graph allocates and resizes them, orders the two
passes, and inserts the barrier between them.

(The originally planned port of `HeatmapPass`/`FinalPass` was replaced by this demo: those passes read
SceneManager buffers, which aren't bound yet, so a port wouldn't show anything this demo doesn't.)

**Result (2026-10-04):**
- `ctest -C Debug` — 10/10 pass, including the new `python.bindings` (5 behavioural tests) and
  `python.spinning_torus` (120 frames). Both fail on any `[error]` log line or nanobind leak report.
- Demo: no validation errors or warnings, no leaks. An orientation check (static torus from above-front)
  confirmed an upright image, correct back-face culling and correct normals.
- `renderer.exe` still runs unchanged (same single pre-existing `inTangent` warning).

**Python API (`import lr`):**
- `Viewer(title, width, height, validation)`: `.frame_graph`, `.resources`, `.swapchain_format`,
  `.on_update(cb(dt, (w, h)))`, `.on_late_update(...)`, `.run()`, `.close()`
- `ResourceRegistry`: `upload_buffer`, `reupload_buffer`, `register_static_buffer`,
  `register_dynamic_buffer`, `update_buffer`, `register_image`, `upload_image`, `has_buffer`,
  `has_image`, `.extent`. Data arguments take any C-contiguous array (numpy, bytes, …).
- `FrameGraph`: `add_pass`, `image`, `buffer`, `compile`, `debug_dump`
- `PassBuilder` mirrors the C++ builder in snake_case:
  - shader arguments accept SPIR-V `bytes`, a `.spv` path, or a GLSL path (compiled immediately, so
    errors point at the declaring line);
  - `vertex_layout([VertexBinding], [VertexAttribute])`;
  - `color_attachment(image, format, load_op, clear_color, extent)` and `depth_attachment(...)` take
    sensible defaults.
- `PassContext`: `.cmd`, `.rendering_extent`, `.extent(image)`, `.push_constants(stages, data)`
- `CommandBuffer`: `draw`, `draw_indexed`, `dispatch`, `set_viewport`, `set_scissor`
- Enums: `Format`, `BufferUsage`, `ImageUsage` and `Stage` (the three usage/stage enums are flags, so
  `|` works), plus `Topology`, `LoadOp`, `PassType`, `ShaderStage`, `Extent`
- `compile_glsl`, `compile_glsl_source`, `ShaderCompileError`, `SHADER_DIR`, `ASSET_DIR`
- `lr.transforms`: `perspective`, `look_at`, `rotation`, `to_gpu`. These use the engine's conventions:
  Vulkan depth range, Y flip, counter-clockwise front faces.

**Files:** `src/python/LrModule.cpp`, `python/lr/{__init__,transforms}.py`, `examples/python/`,
`tests/python/test_bindings.py`; `Viewer::{swapchainFormat, hasImguiPass, requestClose}` and
`Window::requestClose`; Python block in `CMakeLists.txt` (`LR_BUILD_PYTHON`, default ON).

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
- Each step is left uncommitted on the `python-bindings` branch for review; the user commits it.
- Python package layout: CMake builds `_lr.pyd` into `build/python/lr/`, copies `python/lr/*.py` there on
  every build, and copies the vcpkg runtime DLLs (excluding `python3xx.dll`) next to the `.pyd`. This is
  needed because Python only searches an extension's own directory for its DLLs. There's no wheel or
  install step yet.
- Debug builds load fine into the release CPython (nanobind avoids `python3xx_d.lib`).
- **Callback errors:** Python callbacks run inside the C++ frame loop. The first exception is stashed,
  the window is asked to close, and `run()` re-raises it with its traceback once the frame finishes.
  This avoids unwinding through a half-recorded command buffer.
- **Callback lifetimes:** callbacks usually capture the `Viewer`, which creates a reference cycle
  through C++ that Python's GC can't see. The callables are therefore held in slots that are released
  when `run()` returns and at interpreter exit (`atexit`). As a result, a `Viewer` can only `run()` once.
- `PassContext` is only valid inside its execute callback; keeping it afterwards is undefined (documented
  in its docstring, not enforced).
- Push-constant sizes and descriptor bindings aren't checked against the shader yet; that's step 3.
