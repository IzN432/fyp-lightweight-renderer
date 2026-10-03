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

## Step 1b — minimal Python module ☑️ (committed a9900d7)

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

## Step 2 — pipeline state on `PassBuilder` ☑️ (committed 8d9c0e8)

**Goal:** blend, cull, depth test/write/compare and polygon mode configurable per pass (from C++ and Python).

**Demo:**
```
PYTHONPATH=build/python python examples/python/pipeline_state.py
```
A slowly turning torus drawn by three passes into the window
([examples/python/pipeline_state.py](examples/python/pipeline_state.py)):
- **"solid"** uses the default state (back-face culling, depth test + write).
- **"wire"** draws the same mesh as a clean wireframe on top: `polygon_mode(LINE)`,
  `depth(test=True, write=False, compare=LESS_OR_EQUAL)`, and `depth_bias(-1, -1)` so the lines
  don't z-fight with their own surface.
- **"glass"** draws three overlapping alpha-blended quads: `blend(ALPHA)`, `cull(NONE)`, and
  `depth(test=True, write=False)`. The torus hides them where it's in front, and they tint it where
  they're in front.

**Result (2026-10-04):**
- `ctest -C Debug` — 11/11 pass. New: `python.pipeline_state` (120 frames, fails on any `[error]` or leak),
  `framegraph.topology` "blending reads color attachments", and two cases in `python.bindings`
  (`test_pipeline_state_runs`, and `test_invalid_pipeline_state_raises_at_compile`, which checks the error
  names the pass).
- Screenshot confirmed all three effects. There were no validation errors or warnings, which matters
  most for blending: the new barrier read access passes validation.
- `renderer.exe` is unchanged (same single pre-existing `inTangent` warning); with no state set, every
  default resolves to the old hard-coded values.

**API:**
- **C++:** `PassBuilder::blend(BlendMode)`, `polygonMode(VkPolygonMode)`,
  `cull(VkCullModeFlags, VkFrontFace = CCW)`, `depth(test, write, VkCompareOp = LESS)` and
  `depthBias(constant, slope = 0)`. They're stored in `PassDesc::graphics` (`GraphicsState`).
- **Python:** the same in snake_case, plus enums `BlendMode` (`OPAQUE`/`ALPHA`/`PREMULTIPLIED_ALPHA`/
  `ADDITIVE`), `PolygonMode`, `CullMode` (flag), `FrontFace` and `CompareOp`.

**Behaviour:**
- **Defaults:** anything left unset keeps the old behaviour: `BACK` culling for geometry passes,
  `NONE` for fullscreen ones, and depth test + write exactly when there's a depth attachment.
- **Blending and ordering:** any mode other than `OPAQUE` marks the pass's colour attachments
  read-write, regardless of call order and even with `load_op=CLEAR`. This makes the barrier include
  `COLOR_ATTACHMENT_READ` and orders the pass after the attachment's earlier writer.
- **Compile-time checks:** `compile()` / `run()` raise, naming the pass, if depth test or write is
  enabled without a depth attachment, or if `LINE`/`POINT` mode is requested on a device without
  `fillModeNonSolid`.
- **Device feature:** `fillModeNonSolid` is now enabled whenever the device supports it, and exposed as
  `VulkanContext::supportsWireframe()`.

**Not done (deliberately):**
- Blend mode is per pass, not per attachment.
- No custom blend factors.
- No line width (that would need the `wideLines` feature).

**Files:** `GraphicsState`/`BlendMode` in `PassDefinition.hpp`; `PassBuilder`, `GraphicsPipeline`,
`FrameGraphCompiler` and `VulkanContext` changes; `examples/python/{pipeline_state.py, meshes.py}` and
`shaders/{solid.frag, wire.frag, glass.vert, glass.frag}`. `spinning_torus.py` now imports its mesh from
`meshes.py`.

## Step 3 — shader reflection and Python-friendly errors ☑️ (committed 66b2fc7, with 3b)

**Goal:** validate declared bindings and push-constant sizes against the shader's SPIR-V (SPIRV-Reflect) at
`compile()`, and surface validation-layer errors as Python exceptions in debug builds.

**Demo:**
```
PYTHONPATH=build/python python examples/python/error_handling.py
```
[examples/python/error_handling.py](examples/python/error_handling.py) runs six scenarios in one interpreter.
The first five each contain one deliberate mistake; the script catches and prints the exception, and
asserts on its message:

| # | Mistake | Exception (abridged) |
|---|---|---|
| 1 | GLSL typo | `lr.ShaderCompileError: … tinted.frag:6: 'colr' : no such field …` |
| 2 | `uniform_buffer(1, …)`, shader reads binding 0 | `lr.ShaderInterfaceError: pass 'tinted' doesn't match its shaders: binding 0 ('tint', uniform buffer, fragment) is used by the shaders but not declared by the pass` — note: `declares bindings no shader uses: 1 (uniform buffer)` |
| 3 | `push_constant_size(8)` for a 16-byte block | `lr.ShaderInterfaceError: … the shaders use 16 bytes, but push_constant_size is 8` |
| 4 | pushing 32 bytes at runtime | `ValueError: push_constants: writes bytes [0, 32) but the pass declares push_constant_size(16)` (raised from `run()`) |
| 5 | `set_viewport(0, 0, 0, 0)` | `lr.VulkanValidationError: … vkCmdSetViewport(): pViewports[0].width (0.000000) is not greater than zero …` |
| 6 | (none) | renders 30 frames normally |

**Result (2026-10-04):**
- `ctest -C Debug` — 13/13 pass. New: `framegraph.shader_interface` (10 C++ cases, no GPU needed) and
  `python.error_handling` (the demo above, run as a test).
- All 20 of the engine's own passes (IBL graph and main graph) pass the new check.
- The check's first run against the engine exposed a bug in the check itself, not the engine: it
  double-counted push-constant offsets for blocks that start past 0, as `overlay_geometry.frag` does
  (`layout(offset = 64)`). That's fixed and covered by a test.
- `renderer.exe` is unchanged (same single pre-existing `inTangent` warning).

**What `validateShaderInterface` checks** (runs on every pass at `compile()`, before any Vulkan objects
are created; `core/framegraph/compiler/ShaderInterface.{hpp,cpp}`):
- **Descriptors:** each one a shader statically uses must be declared at the same binding, with the
  same descriptor type, for every stage that uses it. Only `set = 0` exists.
- **Push constants:** each used block must fit `push_constant_size`, and be declared for the stages
  that use it.
- **Vertex inputs:** each vertex-shader input location needs a vertex attribute. Fullscreen passes
  take no vertex input.
- **Errors:** every problem is reported at once (`ShaderInterfaceError`, naming the pass and the
  shader file). Unused declarations are listed as a note, since they're usually the other half of an
  off-by-one.
- **Unused resources are allowed:** only what a shader *uses* counts, the same rule Vulkan applies.
  This is why the engine's passes, which declare a few things their shaders don't read, still pass.

**Python error surface:**
- **`lr.ShaderInterfaceError`** and **`lr.VulkanValidationError`** are new; both subclass
  `RuntimeError`. All `lr` exceptions now report as `lr.X` rather than `lr._lr.X`.
- **Validation errors:** with validation on (the default), the first validation-layer error closes the
  window after the current frame, and `run()` raises it, saying how many more followed.
  - Errors are attributed to the `Viewer` that caused them, so a stale error from a destroyed `Viewer`
    never surfaces in a later one.
  - `Viewer(raise_validation_errors=False)` keeps the old log-only behaviour.
  - A Python exception from a callback takes priority over a validation error, since it's usually the
    cause.
- **`PassContext.push_constants`:** raises `ValueError` when you write past `push_constant_size`, use
  undeclared stages, or pass a size or offset that isn't a multiple of 4. This works even with
  validation off.
- **Shader names:** GLSL compiled from a path keeps that path in error messages (`ShaderCode::name`),
  and SPIR-V given as bytes is labelled e.g. "fragment shader (SPIR-V bytes)".

**Not checked (yet):**
- vertex attribute format vs the shader input type;
- descriptor array counts;
- `sampled_image` vs `sampled_depth` (both are combined samplers);
- storage-image format qualifiers.

**Files:** `core/framegraph/compiler/ShaderInterface.{hpp,cpp}`, `tests/framegraph/ShaderInterfaceTests.cpp`,
`examples/python/error_handling.py`; `VulkanContext::{setValidationErrorHandler,notifyValidationError}`,
`Viewer::onValidationError`, `PassContext::{pushConstantSize,pushConstantStages}`, `ShaderCode::name`;
`spirv-reflect` in `vcpkg.json`.

## Step 3b — failures inside lr itself ☑️ (committed 66b2fc7)

**Goal:** a bug *inside* the library must surface as a Python exception or a readable crash report. It
must never hang the script, show a modal dialog, exit silently, or destroy objects the GPU is still using.

**Demo:**
```
PYTHONPATH=build/python python tests/python/test_failures.py
```
It uses private failure-injection hooks in `lr._lr._testing` and checks four cases:
- **C++ exception mid-frame:** `throw_in_frame_loop` throws a C++ exception inside the frame loop after
  5 frames, while about 100 ms of GPU work per frame is still in flight. `run()` raises `RuntimeError`,
  teardown produces no validation errors, and a new `Viewer` then renders normally.
- **Access violation, `std::terminate`, failed debug-runtime assertion:** each is run in a child
  process, which must exit within 20 s with a message and the Python line that called into `lr`.
  Output now looks like this:
  ```
  Windows fatal exception: access violation
  Current thread 0x0000fe80 (most recent call first):
    File "<string>", line 3 in <module>
  ---
  lr: std::terminate called
  Fatal Python error: Aborted
    File "<string>", line 3 in <module>
  ---
  lr: ...\include\vector(1931) : Assertion failed: vector subscript out of range
  Fatal Python error: Aborted
    File "<string>", line 3 in <module>
  ```

**Result (2026-10-04):**
- `ctest -C Debug` — 14/14 pass, including the new `python.failures`.
- **Before the fixes**, the three crash cases exited with *empty* stderr. In Debug builds the
  `terminate` and assertion cases also showed modal dialogs ("abort() has been called" and "vector
  subscript out of range"), which the user had to click away during the first run.
- **After the fixes**, there are no dialogs, and each case prints its message and Python traceback.
- A normal 200-frame demo run with `faulthandler` on produced no spurious reports.
- `renderer.exe` is unchanged.

**What was found and fixed:**
1. **Teardown with frames in flight.** `Viewer::run()` only waited for the GPU on a normal exit, and
   `~Viewer` didn't wait at all. Teardown was safe only by accident: `~ImguiPass` (the last member, so
   destroyed first) called `waitIdle()`. That call *throws* on `VK_ERROR_DEVICE_LOST`, and a throw from a
   destructor means `std::terminate`. So a GPU hang would have turned a catchable `RuntimeError` into a
   hard abort.
   - New `VulkanContext::waitIdleNoThrow()`, which logs failures instead of throwing.
   - `~Viewer` calls it first, before any member is destroyed.
   - `run()` calls it when unwinding from an exception.
   - `~ImguiPass` uses it too.
2. **Silent native crashes.** `lr/__init__.py` now enables `faulthandler` (unless the application
   already configured it), so a crash prints the Python traceback.
3. **Debug runtime dialogs** (Debug builds only). The module runs on the debug C runtime, separate from
   Python's. That runtime shows modal dialogs, and its `abort()` raises `SIGABRT` in its own signal table,
   which `faulthandler` never sees. At import, the module now:
   - sends debug-runtime reports to stderr;
   - turns off `abort()`'s message box;
   - installs a terminate handler, an invalid-parameter handler and a report hook, all of which print
     and then call the *interpreter's* `abort()` (found in `ucrtbase.dll`), so `faulthandler` reports it.

   The report hook is necessary because the current MSVC STL fast-fails right after an assertion report,
   bypassing the invalid-parameter handler. Release builds share Python's runtime, so none of this is
   compiled into them.
4. **Bug or user mistake?** These can't be told apart automatically. `VulkanValidationError` messages
   now end with a line saying it may be a bug in `lr` and asking for a report with the log. Other library
   exceptions already name their component (`CompiledFrameGraph: …`).

**Not covered:**
- A real `VK_ERROR_DEVICE_LOST` can't be produced on demand, so that path is covered by the reasoning in
  fix 1 rather than a test.
- The C++ `renderer.exe` still shows an "abort() has been called" dialog on a fatal error: `main.cpp`
  re-throws from its catch block. That's outside the Python module.

**Files:** `VulkanContext::waitIdleNoThrow`, `Viewer::~Viewer`/`run`/`runFrames`, `ImguiPass::~ImguiPass`,
the debug-runtime setup and `_testing` submodule in `src/python/LrModule.cpp`, `faulthandler` in
`python/lr/__init__.py`, `tests/python/test_failures.py`.

## Step 4 — runtime robustness ✅

**Goal:** deferred destruction for replaced buffers/images (no manual `wait_idle`), automatic recompile when
the graph changes, indirect draws, buffer readback.

**Demo 1 — GPU-driven instancing:**
```
PYTHONPATH=build/python python examples/python/gpu_instancing.py
```
[gpu_instancing.py](examples/python/gpu_instancing.py) runs three passes:
- **"reset"** (compute) writes an indexed-indirect command.
- **"cull"** (compute) animates a 100×100 grid and atomically compacts the visible instances into a
  buffer, counting them into the command.
- **"draw"** issues one `draw_indexed_indirect`, reading the compute output as a per-instance vertex
  buffer.

The CPU never knows the visible count. Every 120 frames the animation holds for 4 frames (more than
the frames in flight); `read_buffer()` then pulls both buffers into numpy, and they're checked cell by
cell against a numpy re-implementation of `instances_cull.comp` (`np.allclose`, atol 2e-3). Output:
`frame 118: GPU kept 4864 of 10000 instances; readback matches numpy`.

**Demo 2 — live editing:**
```
PYTHONPATH=build/python python examples/python/live_edit.py
```
[live_edit.py](examples/python/live_edit.py) makes three kinds of change while running, and checks
`fg.compile_count` after each:
- **Every frame:** the trail's vertex buffer is replaced with one more point → **0 recompiles**.
- **Frame 120:** a new pass ("dots", instanced quads reading the same buffer per instance) is added
  mid-run → **1 recompile**.
- **Frame 240:** the uniform buffer both passes read is reallocated (`replace_dynamic_buffer`) →
  **1 recompile**.

Output: `300 frames, 302 trail points, 3 compiles: 1 at start, 1 for the added pass, 1 for the replaced
uniform buffer`.

**Result (2026-10-04):**
- `ctest -C Debug` — 16/16 pass. New: `python.gpu_instancing` and `python.live_edit` (both fail on any
  `[error]`); `framegraph.topology` cases "runsLast orders after later-declared passes" and "definition
  revision tracks changes"; `python.bindings` cases for readback, replacement and undeclared indirect
  buffers.
- **Proved the live-edit test can fail.** With deferred destruction temporarily disabled, `live_edit.py`
  immediately raised `lr.VulkanValidationError: vkDestroyBuffer(): can't be called on VkBuffer [trail]
  that is currently in use`. With it enabled: no validation messages. The change was reverted (no
  `TEMPORARY` markers left).
- **Screenshots:** the culled disc of cubes; the spiral with the added dots and restyled colours.
- `renderer.exe` is unchanged: the same 2 compiles (IBL + main) over 15 s, so nothing recompiles per
  frame, and the same single pre-existing warning.

**What changed (C++):**
- **Deferred destruction (`ResourceRegistry`):**
  - `replaceUploadedBuffer`, `replaceDynamicBuffer` and `replaceUploadedImage` now *retire* the old
    resource instead of destroying it.
  - `Viewer` numbers submitted frames and calls `resources().beginFrame(frame, lastCompleted)` after
    each frame-slot fence wait (frames ≤ `frame − framesInFlight` are complete). Retired resources are
    destroyed once the frame they were retired in has completed.
  - `replaceUploadedBuffer` now allocates first and retires after, so a failure leaves the old buffer
    usable. It also drops superseded pending uploads.
- **Upload synchronisation:** `flushUploads` now opens and closes its command buffer with whole-queue
  memory barriers. In-place `reuploadBuffer` writes therefore wait for in-flight frames still reading the
  buffer, and the data is visible to every later stage (previously only fragment shaders, via the image
  transitions). It also no longer copies `dest->size` bytes out of a smaller staging buffer, an existing
  out-of-bounds read when re-uploading less than the whole buffer.
- **Automatic recompile (`FrameGraph`):**
  - `FrameGraphDefinition::revision()` increases on `addPass`/backbuffer import and on any mutable pass
    access (i.e. `PassBuilder` calls); const access doesn't count.
  - `ResourceRegistry::generation(name)` increases when a resource is replaced.
  - `execute()` recompiles when the revision changed or a *descriptor-bound* resource's generation did.
    Vertex, index and indirect buffers are looked up by name every frame, so replacing them never
    recompiles.
  - `compile()` waits for the device before discarding an old compiled graph, so an explicit
    `fg.compile()` mid-run is also safe now.
  - New `compileCount()`.
- **ImGui stays last (`runsLast`):** a new pass flag, honoured by `GraphCompiler` both when deriving
  hazards from declaration order and when breaking ties. `Viewer`'s ImGui pass uses it instead of
  `dependsOn(passes declared so far)`, so passes added later still draw underneath the UI with no cycle.
- **Indirect draws:**
  - `BufferUsage::Indirect` and `PassBuilder::indirectBuffer()`; barriers use `DRAW_INDIRECT` /
    `INDIRECT_COMMAND_READ`.
  - `CommandBuffer::drawIndirect/drawIndexedIndirect/dispatchIndirect`.
  - `PassContext::drawIndirect/drawIndexedIndirect/dispatchIndirect` check that the buffer was declared
    with `indirectBuffer()` and bounds-check the read range.
- **Readback:**
  - `ResourceRegistry::readBuffer(name)` flushes pending uploads, waits idle, then copies through a
    host-visible staging buffer with explicit barriers and `vmaInvalidateAllocation`.
  - Every registry buffer now gets `TRANSFER_SRC` so any of them can be read back.
  - The one-shot submit code is now shared (`runOneShotCommands`).

**Python API:**
- **`ResourceRegistry`:** `replace_buffer`, `replace_dynamic_buffer`, `replace_image`, and
  `read_buffer(name)` (returns a `uint8` numpy array; use `.view(np.float32)` etc.).
- **`PassBuilder`:** `indirect_buffer`, `runs_last`.
- **`PassContext`:** `draw_indirect`, `draw_indexed_indirect`, `dispatch_indirect`.
- **`FrameGraph`:** `needs_recompile`, `compile_count`.

**Known limitation (not fixed here):** `update_buffer` writes a single persistently-mapped allocation.
With 2 frames in flight, the CPU can write next frame's data while the previous frame is still reading
it (the engine's own camera UBO has the same issue). It's usually invisible, but it is a race. The fix
is per-frame-in-flight copies of dynamic buffers, which means per-frame descriptor sets. That's a natural
next step.

**Files:**
- **Registry and graph:** `ResourceRegistry.{hpp,cpp}`, `FrameGraph.{hpp,cpp}`,
  `FrameGraphDefinition.{hpp,cpp}`, `PassContext.{hpp,cpp}`, `PassBuilder.{hpp,cpp}`,
  `PassDefinition.hpp`, `PassDescAdapter.cpp`, `VulkanBarrierPlanner.cpp`, `GraphCompiler.cpp`,
  `model/GraphDefinition.{hpp,cpp}`.
- **Vulkan and app:** `CommandBuffer.{hpp,cpp}`, `Renderer.hpp`, `Viewer.{hpp,cpp}`.
- **Bindings:** `LrModule.cpp`.
- **Examples:** `examples/python/{gpu_instancing.py, live_edit.py, meshes.py}` and 8 new shaders.

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
- Push-constant sizes and descriptor bindings are checked against the shader since step 3.
- `python/lr/__init__.pyi` (type stubs) is maintained by the user. It declares the `lr` exceptions as
  `Exception` subclasses, but at runtime `ShaderInterfaceError` and `VulkanValidationError` subclass
  `RuntimeError`.
- **Crash tests:** never run native-crash tests (`_testing.crash`) without the debug-runtime setup from
  step 3b in place. Without it they pop modal dialogs that someone has to click away.
