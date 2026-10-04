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

## Step 4 — runtime robustness ☑️ (committed 1d55ba2)

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

**Known limitation (fixed in step 5):** `update_buffer` writes a single persistently-mapped allocation.
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

## Step 5 — per-frame copies of CPU-written buffers ☑️ (committed 2cb3d52)

**Goal:** remove the known race from step 4. `update_buffer` writes a single persistently-mapped
allocation, so with 2 frames in flight the CPU overwrites data a still-executing frame hasn't read yet.
Fix: each dynamic buffer gets one copy per frame in flight, and passes that bind one get one descriptor
set per frame.

**Demo:**
```
PYTHONPATH=build/python python examples/python/per_frame_data.py
```
[per_frame_data.py](examples/python/per_frame_data.py) runs a slow pass so frames overlap, then has a
compute pass record which frame's uniform data each frame actually read. The true frame number comes
from a push constant, which can't race. The script reads the record back and counts frames that saw
another frame's data.

**Before (measured on 1d55ba2):** `149 of 150 frames read another frame's uniform data (e.g. frame 1 saw
frame 2; offsets seen: [1])`. Every frame with a successor read the successor's value.

**After:** `0 of 150 frames read another frame's uniform data`.

**Result (2026-10-04):**
- `ctest -C Debug` — 17/17 pass. New: `python.per_frame_data` (the demo; asserts 0 stale frames), and two
  `python.bindings` cases:
  - `test_dynamic_data_written_once_reaches_every_frame`: data written once before `run()` reaches the
    copy of every frame in flight;
  - `test_gpu_writes_to_dynamic_buffers_are_rejected`.
- **Proved the write-once test can fail.** With the per-frame refresh in `beginFrame` disabled, the
  copy that was never filled showed up as zeros on every other frame (`[0, 0, 1234, 0, 1234, 0, 1234, 0]`)
  and the test failed. Reverted (no `TEMPORARY` markers).
- **`renderer.exe` (the engine's own camera/light buffers now go through per-frame copies):**
  - logs are unchanged: 2 compiles, 2 descriptor pools, the same single pre-existing warning;
  - a screenshot shows the Cornell box and lion lit and in perspective, so the camera and light data
    reach the GPU.

**What changed:**
- **`ResourceRegistry` dynamic buffers:**
  - Each one holds `framesInFlight` persistently mapped copies, a CPU shadow of the latest contents, and
    a version per copy.
  - `updateBuffer` writes the shadow and the current frame's copy, and flushes the allocation
    (CPU_TO_GPU memory may not be coherent).
  - `beginFrame` refreshes the new frame's copy from the shadow if it's behind. That copy was last used
    by frame `F − framesInFlight`, which has completed. So data written once (e.g. before `run()`)
    reaches every copy, and partial updates keep the rest of the buffer.
  - `getBuffer(name)` returns the current frame's copy, so vertex, index and indirect bindings (looked
    up per frame) pick up the right copy with no further changes.
  - `readBuffer` returns the CPU contents for dynamic buffers.
  - `replaceDynamicBuffer` retires all copies.
  - The slot is `frame % framesInFlight`. `Viewer` calls `setFramesInFlight` (2) before anything can
    register a buffer, and the default is 1, so standalone `FrameGraph` use is unchanged.
- **Descriptor sets:** a pass that binds a per-frame buffer gets one descriptor set per frame in flight,
  each pointing at that slot's copy. `CompiledFrameGraph::execute` binds the set for
  `registry.frameSlot()`. Passes without per-frame buffers still get a single set.
- **`DescriptorAllocator` pools:** these now grow. When a pool is exhausted, another with the same sizes
  is created and the allocation retried. This replaces the fixed 64-set pool, which per-frame sets could
  have exhausted. Each pool still fits the largest single set (`GeometryPass`'s 4 × 256 samplers).
- **New compile error:** a pass that writes a dynamic buffer (storage write/read-write). GPU writes would
  land in one frame's copy only; static buffers are the right tool. None of the engine's passes do this.
- **Python:** `cmd.dispatch(x, y=1, z=1)`. `y` now defaults to 1; `dispatch(n)` used to be a
  `TypeError`.

**Files:** `ResourceRegistry.{hpp,cpp}`, `CompiledFrameGraph.{hpp,cpp}`, `FrameGraphCompiler.cpp`,
`DescriptorAllocator.{hpp,cpp}`, `Viewer.cpp`, `LrModule.cpp`; `examples/python/per_frame_data.py` and
`shaders/{busy.frag, record_frame.comp, show.frag}`; `tests/python/test_bindings.py`.

## Type stubs — kept in sync automatically ✅

`python/lr/__init__.pyi` (hand-written, with docs) is now checked by two ctests, so it can't drift:
- **`python.stubs`** ([tests/python/test_stubs.py](tests/python/test_stubs.py)) generates a reference stub
  from the compiled module with nanobind's stubgen. Every public class, method, property, enum member and
  function must be in the hand-written stub with the same parameter names, and the stub may only describe
  names that exist at runtime. stubgen omits some real names, such as `nb::exception` classes and inherited
  enum members, so those are checked against the live module instead.
- **`python.typecheck`** ([tests/python/typecheck.py](tests/python/typecheck.py)) runs mypy over the
  examples and Python tests against the stub. mypy is optional: without it the test is reported as
  skipped (exit code 77).

**Stub fixes these found:**
- Enums are now `enum.Enum` / `enum.IntFlag` subclasses, as at runtime, with `X = ...` members. `.name` and
  `.value` are therefore typed for all of them, and the invalid narrowing `__or__`/`__and__` overrides on
  the flag enums are gone.
- `vert_shader`, `frag_shader`, `compute_shader` and `depends_on` are `@overload`s with the real parameter
  names (`path=` / `spirv=`, `dependency=` / `dependencies=`).
- Array arguments are typed `collections.abc.Buffer` (numpy arrays, bytes, …) instead of `object`.
- `read_buffer` returns `NDArray[numpy.uint8]`.

**Other:** the example and test scripts got small annotations so mypy runs cleanly on them (e.g. `state:
dict[str, Any]`).

## Merge with main (1046d25) ✅

Merged `main` (10 commits: ImGuizmo translate/rotate/scale gizmos, keyframe editing, Cornell box removal)
into this branch. Only `SphericalCameraController` touched code on both sides, and it auto-merged
correctly: main's `update(dt, gizmoCapturesPrimaryMouse)` overload is kept and still ends in this
branch's `applyPose()`.

Results:
- full rebuild clean, `ctest` 20/20;
- `renderer.exe` runs with the same 2 compiles and the single pre-existing warning;
- every merged pass, including the restored overlay geometry stage and the gizmos, passes the shader
  interface check.

## Step 7 (proposal) — scene, scene manager and loaders from Python

The question: can Python reuse the engine's `Scene` / `SceneManager` / loaders, or are they too
editor-specific? Reading the code, the answer is "split at the right seam".

**What's general and CPU-only (safe to expose as-is):**
- **`Scene`:** a `SceneObject` graph with stable ids, parenting and components (`TransformComponent`,
  `MeshComponent`, `Light`, `Camera`, `AnimatorComponent`, `SkinComponent`, colliders…). It carries some
  editor UI (`onHierarchyGUI`, selection), but Python can simply not call that.
- **`MeshStore`, `MaterialStore`:** plain containers (no `ResourceRegistry`).
- **`SceneLoader::load(path, scene, meshStore, materialStore)`:** OBJ/glTF/GLB into those, including
  hierarchy, materials, skins and animations. Also CPU-only.
- **`Mesh`:** positions, topology, per-vertex/per-corner attributes and face groups → numpy.

**What isn't general: `SceneManager`.** It mixes three jobs:
1. **Owning the asset stores and `load()`.** General.
2. **GPU sync:** packs everything into the fixed buffers `GeometryPass`/`PbrPass` expect
   (`meshVertexBuffer`, `meshIndexBuffer`, materials SSBO + texture arrays with the 48-byte
   `GpuMaterialLayout`, the lights buffer, the camera UBO, skin palettes), and `flushDirty()`. General
   *for the engine's deferred pipeline*, but it hard-codes that pipeline's data contract.
3. **Editor session:** `EditorMode`, `SelectionManager`, the "edited mesh", the points/heatmap buffers for
   the vertex tools, and the light-visual quads. Editor-only.

  Job 3 is why it doesn't feel general: `initialize()` requires a camera, at least one mesh, an input
  handler, and builds a `SelectionManager`, none of which a Python renderer necessarily wants.

**Proposed approach, in three steps, each useful on its own:**
- **7a — CPU scene in Python (no `SceneManager`).**
  - `lr.load_scene(path) -> lr.SceneAsset`, owning a `Scene` + `MeshStore` + `MaterialStore` filled by
    `SceneLoader`.
  - Python can walk objects and hierarchy and read world transforms. Meshes come out as numpy (vertex
    and index arrays already flattened the way `MeshUploader` packs them, per face group with its
    material); materials as parameters plus texture images as numpy; plus lights, cameras, and animation
    playback via `AnimatorComponent`.
  - A Python renderer then uploads whatever it needs with the existing `ResourceRegistry` API.
  - Needs the `_lr` module to link the animation/skinning feature libraries, which `SceneLoader` already
    depends on.
- **7b — split `SceneManager` in C++.**
  - **`SceneGpuSync`** (job 2): stores + uploaders + `flushDirty`, no editor state.
  - **`EditorSession`** (job 3): selection, edit modes, vertex-tool buffers, light visuals.
  - `main.cpp` keeps today's behaviour by owning both, so the C++ editor is unchanged. This is the
    "move the glue out of `main.cpp`" item from the very first analysis.
- **7c — engine passes as Python building blocks.** With `SceneGpuSync` exposed, Python can add the
  engine's `GeometryPass` / `AmbientOcclusionPass` / `PbrPass` / `FinalPass` to its own frame graph and
  insert its own passes between them, reading the G-buffer and lights. In effect it extends the real
  renderer.

**Decision (user, after the proposal):** Python builds its *own* renderers. The engine provides building
blocks, including its existing passes, each with clear documentation of what it takes in and what it
outputs. So all three parts are wanted, in order:
- **7a** gives Python the scene data.
- **7b** makes "scene → engine-standard GPU buffers" a reusable block, separate from the editor.
- **7c** wraps each engine pass with a documented input/output contract.

**7a, 7b, 7c — done ✅** (each one's details follow)

**7a scope:**
- **`lr::conventions`** (`core/scene/EngineConventions.hpp`): the attribute, texture and material-parameter
  names the loaders write and `GeometryPass` reads, the default material and the material capacity. These
  were literals in `main.cpp`; `main.cpp` now uses the header too, so C++ and Python can't disagree.
- **`lr::SceneAssets`** (`core/scene/SceneAssets.hpp`): a `Scene` + `MeshStore` + `MaterialStore`
  filled by `SceneLoader`. 7b's GPU-sync block will consume this.
- **Python (CPU only):**
  - `lr.load_scene(path)` / `lr.Scene().load(path)`.
  - Scene objects with hierarchy, name, local and world transforms.
  - Meshes as numpy: corner-domain `positions`, every per-vertex attribute, `indices`, per-face
    `face_materials`, and the unique-position topology.
  - Materials: parameters, plus textures as numpy.
  - Lights, and animation playback (`scene.update(dt)`).

**7a demo:**
```
PYTHONPATH=build/python python examples/python/scene_viewer.py [model.glb|.gltf|.obj]
```
[scene_viewer.py](examples/python/scene_viewer.py) is a renderer written in Python that draws any scene the
engine can load:
- **Geometry:** every mesh's `positions` + `normal` + `uv` go into one vertex buffer. Faces are sorted by
  material into one index buffer, so each (object, material) pair is one `draw_indexed` range with a
  vertex offset.
- **Textures:** each material's `baseColorTexture` goes into a texture array (`upload_array_image` +
  `sampled_image_array`, the same mechanism `GeometryPass` uses).
- **Shading:** albedo = texture × `baseDiffuse`, as in `geometry.frag`. The fragment shader's array size
  is filled in at load time (`#define MATERIAL_COUNT`).
- **Camera:** the engine's `lr.OrbitCamera`, framed to the scene's world-space bounds.

**7a results (2026-10-04):**
- `ctest -C Debug` — 22/22 pass, with two new tests:
  - **`python.scene`** (CPU only, no window), 6 cases:
    - glTF meshes are render-ready (shapes and dtypes, unit normals, `positions ==
      unique_positions[position_indices]`, `KeyError` listing the available attributes);
    - materials carry the convention parameter and texture names;
    - hierarchy is consistent, and moving the root moves descendants' world matrices;
    - playing an animation moves joints;
    - a two-material OBJ keeps per-face materials and their colours;
    - several loads into one `Scene`, without a `Viewer`.
  - **`python.scene_viewer`:** the demo, for 120 frames.
- **Screenshots:**
  - `bird_orange.glb` renders textured and lit, framed automatically;
  - the lion head (47k triangles, 4k textures) loads and renders in about 5 s;
  - a two-material OBJ renders red and green.
- **Bug caught by that last check:** the OBJ first rendered both quads grey. The loader gives untextured
  materials a white placeholder `baseColorTexture`, and the demo ignored `baseDiffuse`. It now multiplies
  them as the engine does.
- **`renderer.exe` with `lr::conventions` in `main.cpp`:** unchanged (2 compiles, the same single
  warning).
- **Stubs:** all 326 public names match; mypy clean.

**7a notes:**
- **Skinned meshes:** `Mesh` data is the rest pose. Animation playback moves joint objects, but deforming
  the mesh needs the skin palettes; GPU skinning comes with 7b/7c through the engine's skin buffers.
- **Array ownership:** mesh and material arrays are *copies* (numpy owns them), so they stay valid
  however the scene changes. Owned arrays are returned as plain objects, because nanobind's property
  default (`reference_internal`) can't apply to an array that owns its data.
- **nanobind copy trait:** `SceneObject` needed a `nanobind::detail::is_copy_constructible` override.
  Its components live in a `std::unordered_map` of `unique_ptr`, whose copy constructor isn't
  constrained, so `std::is_copy_constructible` wrongly reports `true`.
- **New Python API for per-draw textures:** `ResourceRegistry.upload_array_image()` and
  `PassBuilder.sampled_image_array()`.
- **Linking:** `_lr` now also links `lr_animation_feature` and `lr_linear_blend_skinning_feature`, which
  `SceneLoader` already depends on.

**Files:**
- **Engine:** `core/scene/{EngineConventions.hpp/.cpp, SceneAssets.hpp}`; `main.cpp` uses
  `lr::conventions`.
- **Bindings:** `LrModule.cpp` (`bindScene`, array-image bindings).
- **Stubs:** `python/lr/__init__.pyi`.
- **Demo:** `examples/python/scene_viewer.py` + `shaders/scene.{vert,frag}`.
- **Tests:** `tests/python/test_scene.py`.

**7b — done ✅: `SceneManager` split; the GPU side is `lr::SceneGpu`**

**What it is:** `core/scene/SceneGpu.{hpp,cpp}` turns a `Scene` (plus its `MeshStore`/`MaterialStore`)
into the buffers the engine's passes read, and keeps them in sync.
- **Construction:** `SceneGpu(registry, scene, meshStore, materialStore)`.
- **Setup:** `addMeshObject`, `addLoaded(SceneLoadResult)`, `setCamera`, then
  `initialize(areaLightConfig, materialLayout, vertexAttributes)`. All three arguments default to
  `lr::conventions`.
- **Per frame:** `registerCallbacks(viewer)` handles aspect ratio, animations, skins and `flushDirty()`.
- **For GeometryPass:** `geometryPassConfig()` returns everything `GeometryPass` needs, and
  `indexRange(mesh)` gives a mesh's range in the shared index buffer.
- **No editor state.** The camera object can live outside the scene; `lr.OrbitCamera`'s does.

**What changed in `SceneManager`:** it keeps only the editor (edited mesh, `SelectionManager`, editor mode,
the points/heatmap buffers). It owns a `SceneGpu`, created in `setScene()`, and forwards its old GPU API to
it, so `main.cpp`, ARAP and Laplace-Beltrami compile unchanged apart from the lines below.

**New in `lr::conventions`** (replacing literals in `main.cpp`):
- `materialLayout()`: the 48-byte material SSBO layout plus the four texture formats.
- `geometryMeshLayout()`: GeometryPass's vertex input.
- `geometryVertexAttributes()`.
- `areaLightVisualConfig()`.

`main.cpp` now builds `GeometryPass` from `sceneManager.gpu().geometryPassConfig()` and those helpers;
about 50 lines of config are gone.

**7b results:**
- `renderer.exe` behaves as before: 2 compiles, the single pre-existing warning, and the screenshot
  matches.
- `ctest` stayed green.

**7c — done ✅: the engine's passes as Python building blocks**

**Demo:**
```
PYTHONPATH=build/python python examples/python/deferred_blocks.py [model.glb] [--hdri sky.hdr]
```
[deferred_blocks.py](examples/python/deferred_blocks.py) is a deferred renderer assembled from engine
blocks, with a pass written in Python in the middle:

`Ibl` → `SceneGpu` → `GeometryPass` → `AmbientOcclusionPass` → `PbrPass` → **fog (Python, reads `pbr` +
`gbufferDepth`, writes `fogged`)** → `CompositePass(input="fogged")`

- **Lights:** a directional light, plus an `image` light for ambient light from the HDRI.
- **Live controls:** ImGui sliders change the fog density and the AO radius while it runs.
- **Contracts:** at startup it prints every block's inputs and outputs, and asserts on them.
- **Screenshot:** the bird, PBR-lit, in front of the HDRI sky, with visible fog; about 410 fps.

**C++:**
- **`CompositePass`** (`core/passes/composite/`): FinalPass without the editor overlays. It draws the
  `ibl_env` sky where depth is 1 and the HDR input elsewhere, Reinhard tone mapped. The input and output
  images are configurable.
  - Its shader shares `utility/tonemap.glslh` with `final.frag`, which now uses that header too, with
    the same maths.
- **`ResourceRegistry::names()`**, used by the contract introspection.

**Python** (`LrModule.cpp`, `bindBuildingBlocks`):
- **`lr.SceneGpu(viewer, scene, camera)`:** uploads the scene, drives the camera UBO from an
  `OrbitCamera`, and runs per-frame sync. `.buffers` gives the buffer names by role.
- **`lr.Ibl(viewer, hdri, env_res, irr_res, pf_res, pf_mips)`:** IBL precompute, run immediately.
- **Passes:**
  - `lr.GeometryPass(viewer, gpu)`, with `.skinning`;
  - `lr.AmbientOcclusionPass(viewer, gpu, ...)`, whose parameters are live properties;
  - `lr.PbrPass(viewer, gpu, ibl)`;
  - `lr.CompositePass(viewer, gpu, input, output, output_format, name)`.
- **Shared base, `lr.EnginePass`:** `inputs`/`outputs` (`lr.ResourceUse`: name, kind, usage, format),
  `passes`, `pass_names` and `describe()`.
  - **Where the contracts come from:** the frame graph's pass declarations, plus a snapshot of the
    registry taken before the block uploads its own data. Private data (HBAO params, LTC tables, the
    HDRI) and intermediates between a block's own passes (HBAO's raw AO) are excluded, so the lists
    match the docs. One exception: Ibl's `ibl_env` is consumed internally, so the block marks it as an
    output explicitly.
- **Scenes:** `Scene.add_light(type, color, intensity, position, rotation, size, cones, name)`, for
  point, spot, area, directional and image lights. Two new `lr.Format` values: `R16G16_SFLOAT` and
  `R16G16B16A16_UNORM`.

**Docs:** [docs/python_building_blocks.md](docs/python_building_blocks.md) covers, for every block:
- what it reads and writes, with names, formats and meaning (G-buffer encoding, AO polarity,
  camera/light/material layouts);
- how blocks connect;
- how to write a pass between them.

**Lifetimes:**
- **Ownership:** the Viewer holds `SceneGpu` and the passes, which its frame loop calls into, the same
  way it holds Python callbacks, so `lr.GeometryPass(viewer, gpu)` works without keeping a reference.
  Each block keeps what it uses alive in turn: the Viewer, the scene and the camera.
- **First attempt:** `keep_alive` pointing both ways made reference cycles. nanobind reported leaks.
- **Second problem, found while proving the tests can fail:** a Viewer that never reached `run()` (an
  exception first) stayed alive. The next `lr.Viewer` then aborted on ImGui's one-context assert. This
  was already possible with plain callbacks; SceneGpu made it likely.
- **Fix:** lr supports one Viewer at a time. Creating one now releases a stale Viewer's holds and
  collects it, or raises a clear `RuntimeError` if that Viewer is still referenced.

**7c results (2026-10-04):**
- **`ctest -C Debug`:** 24/24 pass. New tests:
  - **`python.blocks`**, 6 cases:
    - the contracts, with exact formats;
    - the scene renders: a custom compute pass samples `pbr` and `gbufferDepth` at the centre, and finds
      depth in (0, 1) and a lit colour;
    - after the orbit camera moves mid-run, the camera UBO read back equals `camera.view_matrix()`;
    - AO parameter changes reach `hbao_params`;
    - misuse raises (an empty scene, a second SceneGpu, an unknown light type);
    - a never-run Viewer is released by the next.
  - **`python.deferred_blocks`:** the demo, for 120 frames, under the leak/error regex.
  - **`python.scene`** gained `test_add_light`.
- **The tests can fail:** temporarily skipping the camera re-upload, GeometryPass's draws, and the AO
  upload snapshot each failed its intended test, and the other tests still ran. All reverted, and no
  `TEMPORARY` markers remain.
- **Missing input:** `PbrPass` without an AO pass fails to compile with `image or image array 'hbao_ao'
  not found`.
- **`renderer.exe`** is unchanged after the `final.frag` refactor.
- **Stubs:** 365 public names match; mypy clean.
- **Stub fix:** `Light.color`/`area_size` now really return tuples, as the stubs said.

**Review fix: `SceneGpu` no longer modifies the scene.**
- **The problem:** it used to build light quads by adding a `MeshComponent` to each light object. The
  scene stayed changed after the `SceneGpu` was gone:
  - showing the same scene in a second Viewer threw "Component of this type already exists";
  - each `SceneGpu` leaked one `MaterialStore` slot per light;
  - `light.mesh` appeared in Python;
  - in the C++ editor, lights passed `isEditable`, so a light's quad could become the edited mesh and
    get vertex-edited.
- **The fix:** `SceneGpu` now owns the quads (one heap-allocated `Mesh` per light) and their material
  slots, and releases the slots when a light is removed and in its destructor. The light object stays
  in the draw list as the quad's scene object, with an identity model matrix.
- **Verification:**
  - New test `test_a_scene_can_be_shown_again` (two Viewers in a row, `light.mesh` stays `None`). It
    failed on the old build and passes now.
  - `ctest` 24/24.
  - `renderer.exe` log unchanged.
  - A screenshot shows the area-light quad still rendering.
- **Leftover:** the `MeshComponent` `hideFromGui` flag now has no users.

**Follow-up: lights added, removed and edited while running** (`examples/python/light_editor.py`)

**Demo:** an ImGui panel to add point/spot/area lights, remove the selected one, and edit it (type,
colour, intensity, cones, area size, position, orbit). `--scripted` is the `python.light_editor` ctest:
it adds, edits and removes lights and checks the GPU light buffer after each step.

**API, all thin layers over existing engine pieces:**
- `SceneObject.set_light(...)` uses the new `Light::set()`, which marks the light dirty so the existing
  `flushDirty` path re-uploads it.
- `Scene.remove(obj)` is `Scene::destroySceneObject`.
- `SceneGpu.max_lights`.

**Engine:**
- `SceneGpu::flushDirty` notices lights added or removed and calls `syncLights()`, which builds and
  drops quads, uploads the lights and rebuilds the geometry.
- New listeners, `onGeometryRebuilt` and `onLightsUploaded`, keep `GeometryPass`'s draw lists and
  `PbrPass`'s light count current. `main.cpp` registers them too, replacing its manual refresh calls.
- `LightUploader` raises a clear error above 16 lights instead of overflowing its buffer.

**Bugs found and fixed:**
- **`PbrPass::setNumLights()` had no effect after `build()`.** The count was copied into the execute
  callback once. So in the C++ editor, a deleted light kept lighting the scene.
- **Use-after-free from the previous fix.** Removing a light freed its quad mesh while the geometry
  list still pointed at it. Quads are now dropped only in `syncLights()`.
- **Frame-graph layout bug (pre-existing):** the compiler wrote end-of-frame image layouts into the
  registry *at compile time*. A recompile before the graph had ever executed (any change during frame
  1 or before `run()`) therefore skipped the transitions out of `UNDEFINED`, causing validation errors.
  Layouts are now recorded when `execute()` records a frame.

**Tests:**
- `test_bindings.test_recompile_before_the_first_frame_runs` (generic, no lights involved);
- `test_blocks.test_lights_added_before_the_first_frame`;
- `test_blocks.test_lights_added_and_removed_while_running_light_the_scene`: the centre pixel turns
  red when a red light is added and returns to its original value when the light is removed.

Each test failed with its bug temporarily restored. Results: `ctest` 25/25; `renderer.exe` log
unchanged.

**Area lights: facing fixed, and two-sided as an option** (reported from the light editor)

**The bug:** area lights pointed backwards. Measured with one light between the camera and the bird
and a black environment: aimed at the bird, it lit nothing and showed its bright face to the camera
behind it. The emission (the corner order in `pbr.frag`) and the quad's triangle order were both
wound for a +Z normal, while the light's forward axis is −Z, as for spot and directional lights.
Both are reversed now.

**The option:**
- `AreaLight::twoSided` (default `true`), with a "Two-Sided" checkbox in the Light inspector.
- **GPU side:** the light buffer entry grows from 64 to 80 bytes with a `flags` word, and
  `CalcAreaLight` passes the flag to `LTC_Evaluate`.
- **The quad** always has 8 vertices, a front face and a back face. A one-sided light collapses the
  back face to a point, so switching sides at runtime changes only vertex data (SceneGpu re-uploads
  vertices, not indices).
- **Python:** `add_light(two_sided=True)`, `set_light(two_sided=...)`, `Light.two_sided`, and a
  checkbox in the light editor.

**Test:** `test_area_lights_face_forward_or_both_ways` uses one Viewer, switches the option and the
aim while running, and samples both the bird and the quad. It fails with either winding restored.
`ctest` 25/25; `renderer.exe` log unchanged.

**Files:**
- **Engine:** `core/scene/SceneGpu.{hpp,cpp}`; `SceneManager.{hpp,cpp}`; `EngineConventions.{hpp,cpp}`;
  `core/passes/composite/*`; `utility/tonemap.glslh`; `final.frag`; `ResourceRegistry::names()`;
  `main.cpp`.
- **Bindings:** `LrModule.cpp`.
- **Stubs:** `python/lr/__init__.pyi`.
- **Docs:** `docs/python_building_blocks.md`.
- **Demos:** `examples/python/deferred_blocks.py` + `shaders/fog.frag`; `examples/python/light_editor.py`.
- **Light editing:** `Light.hpp` (`set`), `LightUploader`, `PbrPass`, `CompiledFrameGraph` +
  `FrameGraphCompiler` (layout commit).
- **Tests:** `tests/python/test_blocks.py`, `test_scene.py`.

## Step 8 — your own frame loop: `viewer.step()` ✅

**Goal:** Python owns the loop when it wants to, e.g. for simulations, scripted captures or
notebooks:
```python
while viewer.step():   # window events + one rendered frame; False once the window has closed
    simulate(dt)       # plain Python between frames
    res.update_buffer(...)
```

**Demo:**
```
PYTHONPATH=build/python python examples/python/step_loop.py
```
[step_loop.py](examples/python/step_loop.py) simulates a 6,000-particle fountain in numpy between steps,
uploads it to a dynamic storage buffer, and draws instanced billboards (`shaders/particles.{vert,frag}`).
- Callbacks still run inside each step: the orbit camera via `on_update`, and an ImGui panel tuning
  gravity, spread and bounce.
- Space pauses the simulation.
- After the loop it prints a summary, since the code after the loop is the script's own.
- **Timings:** about 0.13 ms to simulate and 0.4 ms per frame.

**C++:**
- `Viewer::step()`: compiles on the first call, renders one frame, and returns false once the window
  has closed, after waiting for the GPU; later calls do nothing.
- `run()` is now `while (step()) {}`.
- `isOpen()`.
- **`dt` fix:** the first frame's `dt` used to count from GLFW initialisation, so it included all setup
  time (over 2 s in a test). It now counts from the first frame.

**Python:**
- `Viewer.step() -> bool` and `Viewer.is_open`. `run()` and `step()` share `driveFrames()`:
  - callback exceptions and validation errors raised during a frame are re-raised from that
    `step()`, which closes the window;
  - callbacks and held objects are released once the window has closed.
- Mixing works: step a few frames, then `run()` to continue.

**Results:**
- `ctest` 26/26. New tests:
  - **`python.bindings`** gained 4:
    - frames run in your own loop, and closing ends it;
    - a CPU write made before each step reaches exactly that frame (GPU readback over 7 frames);
    - callback errors are re-raised from the step where they happened;
    - `step()` then `run()` continues the same loop.
  - **`python.step_loop`:** the demo, for 200 frames.
- **The tests can fail:** each of three temporary breaks failed its intended test, and all were
  reverted.
  1. Remove the `dt` fix.
  2. `step()` keeps returning true after the window closes.
  3. `step()` stops re-raising callback errors.
- **`renderer.exe` log:** unchanged.

**Files:**
- **C++:** `core/app/Viewer.{hpp,cpp}`.
- **Bindings:** `LrModule.cpp`.
- **Stubs:** `python/lr/__init__.pyi`.
- **Demo:** `examples/python/step_loop.py`, `shaders/particles.{vert,frag}`.
- **Tests:** `tests/python/test_bindings.py`, `CMakeLists.txt`.

## Split: `lr` (frame graph) vs `lr.engine` (the engine's renderer) ✅

**Why:** so it's clear which parts are the general-purpose frame graph and which are the engine's own,
opinionated renderer.

**What lives where:**
- **`lr`:** the frame graph and its tools, with no opinion on scenes or shading. Viewer, FrameGraph,
  PassBuilder, ResourceRegistry, the enums, shader compilation, input, plus the helpers
  `lr.OrbitCamera`, `lr.gui` and `lr.transforms`.
- **`lr.engine`:**
  - scenes: `load_scene`, `Scene`, `SceneObject`, `Mesh`, `Material`, `Light`, `Animator`;
  - GPU layout: `SceneGpu`;
  - passes: `Ibl`, `GeometryPass`, `AmbientOcclusionPass`, `PbrPass`, `CompositePass`, and their base
    `EnginePass` / `ResourceUse`.
- **`OrbitCamera` stays in `lr`.** It only turns input into view/projection matrices, and plain
  frame-graph renderers (`spinning_torus`, `interactive`) use it.

**How it's built:**
- `LrModule.cpp` registers `bindScene` and `bindBuildingBlocks` on a native submodule,
  `lr._lr.engine`.
- [python/lr/engine.py](python/lr/engine.py) re-exports it by name, with a docstring describing the
  layer. `lr/__init__.py` drops the native `engine` its star import picks up, so `lr.engine` is that
  Python module.
- **Stubs:** split into `__init__.pyi` and `engine.pyi`. `test_stubs.py` now runs stubgen recursively,
  checks both files, and checks that `engine.py` exports every native name.
  - **First attempt:** I relabelled the classes `lr.engine.X` (via `__module__`). That made stubgen
    document them as imports, so the engine stubs were silently unchecked (1 reference name instead of
    91). The classes therefore keep `lr._lr.engine`, like `lr._lr.Viewer`.
- **Usages updated:** examples and tests now use `from lr import engine` / `engine.X`; the docs intro
  explains the split.
- `test_scene.test_engine_layer_is_its_own_submodule` keeps the names from drifting back.

**Results:** `ctest` 26/26; stubs: 371 names (280 + 91). A renamed stub class and an unexported name
each failed the stub check.

## Step 6 — interaction from Python ☑️ (committed deb6235)

**Goal:** a Python renderer can be interactive:
- read keyboard and mouse state (`viewer.input`);
- draw ImGui panels from Python (`viewer.on_gui()` + `lr.gui`), guarded so misuse raises instead of
  tripping an ImGui assert;
- drive the engine's own orbit camera (`lr.OrbitCamera`).

**Demo:**
```
PYTHONPATH=build/python python examples/python/interactive.py
PYTHONPATH=build/python python examples/python/interactive.py --scripted   # self-test
```
[interactive.py](examples/python/interactive.py) shows a torus with **the engine's own orbit camera**
(`lr.OrbitCamera`, the C++ renderer's `SphericalCameraController`): middle-drag orbits, Shift + middle-drag
pans, scroll zooms and R resets, the same controls as `renderer.exe`. The "Controls" panel has:
- albedo colour picker, light yaw and pitch sliders, spin speed;
- a wireframe-overlay checkbox (it just skips that pass's draw);
- a "Frame torus" button;
- live readouts of fps and `compile_count`.

Colour and light go through a dynamic uniform buffer (step 5's per-frame copies). `--scripted` runs without
the panel (the camera ignores the mouse over UI, and a real cursor resting on the panel would swallow the
injected input). It injects a 30-frame middle-drag and 3 scroll notches, then checks the result against the
controller's constants: `camera azimuth 0.00 -> -1.74, radius 4.00 -> 3.005 (expected 3.005)`. The drag
moves 0.01 rad/px over 30 × 6 px; the zoom is radius ÷ 1.1 per notch.

**Revision — use the engine camera (requested after review).** The first version shipped a separate
pure-Python `lr.camera.OrbitCamera` with different controls (left-drag orbit, its own speeds and
projection). That was replaced by a binding of the engine's `SphericalCameraController` + `Camera`
component, so Python and C++ share one implementation:
- **Engine change (behaviour-preserving):** `SphericalCameraController` gained
  `orbitState()` / `setOrbitState()` (target, radius, azimuth, elevation, clamped like `update()`), and
  its pose code moved into `applyPose()`, shared by both. `renderer.exe`'s view is unchanged (screenshot
  identical to step 5's).
- **`lr.OrbitCamera(viewer)`** owns a one-object `Scene` holding `Camera` + `TransformComponent`, as
  `main.cpp` sets up the renderer's camera:
  - `update(dt)`;
  - `view_matrix()` and `projection_matrix(aspect)`, as row-major numpy, using the engine's `Camera`
    projection: Vulkan depth [0, 1], Y flip;
  - `matrices(extent)`, packed for a `mat4 view; mat4 proj;` block;
  - `position`, plus read/write `target`, `radius`, `azimuth`, `elevation`, `fov_y_degrees`,
    `near_plane`, `far_plane`, `orthographic`, `ortho_height`.
- **Test:** `test_orbit_camera_is_the_engine_camera` checks the placement formula, that the view maps the
  target to `(0, 0, −radius)`, that the projection maps near/far to depth 0/1, the `matrices()` layout, and
  the clamping.
- **Removed:** `python/lr/camera.py`.

**Result (2026-10-04):**
- `ctest -C Debug` — 20/20 pass:
  - new `python.interactive`;
  - 4 new `python.bindings` cases: gui outside `on_gui` raises; every widget works, and a never-ended
    `begin()` is closed automatically; an exception inside `with gui.window()` closes it and reaches
    `run()`; injected mouse, scroll and key events show up in `viewer.input`;
  - a new `python.failures` case for plain `assert()` (Debug only).
- `python.stubs` checks 258 public names (the new `Key` members included), and `python.typecheck` is clean.
- A screenshot shows the panel and the wireframe overlay. The panel shows "1 compile(s)", so interacting
  never recompiles.
- **Proved the window-balancing test can fail.** With auto-close disabled, the test hit ImGui's "Missing
  End()" assert and the process aborted (exit code 3). Reverted (no `TEMPORARY` markers).
- **That experiment exposed a gap in step 3b:** a plain `assert()` from inside `lr` (as ImGui uses)
  aborted with *no* message and no traceback. It goes through `_wassert` and the debug runtime's own
  `abort()`, bypassing all three 3b hooks. Fixed: Debug builds now call `_set_error_mode(_OUT_TO_STDERR)`
  and install a `SIGABRT` handler in the module's runtime that forwards to the interpreter's `abort()`. The
  same failure now prints `Fatal Python error: Aborted` and the Python line that called into `lr`. Covered by
  the new `python.failures` case.

**API:**
- **`viewer.input`** (`lr.Input`):
  - `is_key_down(lr.Key.X)`, `is_mouse_down(lr.MouseButton.LEFT)`;
  - `mouse_position`, `mouse_delta`, `scroll_delta`, `shift`, `ctrl`, `alt`.
  - `lr.Key` covers A–Z, `DIGIT_0`–`DIGIT_9`, F1–F12, arrows, modifiers, space, enter, escape, tab,
    backspace and delete.
- **`viewer.on_gui(callback)` + `lr.gui`:**
  - `window(title, size=, position=)` (context manager), `begin`/`end`, `text`, `button`, `checkbox`,
    `slider_float` (incl. logarithmic), `slider_int`, `drag_float`, `color_edit3/4`, `combo`,
    `collapsing_header`, `separator`, `same_line`, `spacing`;
  - also `want_capture_mouse`, `want_capture_keyboard`, `framerate`, which are safe anywhere.
  - Value widgets return `(changed, value)`.
  - Typed Python wrappers live in `python/lr/gui.py` over a private native `_gui` module.
- **`lr.OrbitCamera(viewer)`:** the engine's camera; see "Revision" above.
- **Test-only:** `lr._lr._testing.inject_mouse_move/inject_mouse_button/inject_scroll/inject_key`.

**Guarding ImGui from Python.** An ImGui assert would abort the interpreter, so the bindings stop misuse
first:
- widgets outside `on_gui` raise `RuntimeError`;
- `on_gui` closes any windows a callback left open, so the frame stays balanced even when the callback
  raises;
- empty window titles and empty combo lists raise `ValueError`.

**Stubs:** `__init__.pyi` gained `Key`, `MouseButton`, `Input`, `OrbitCamera`, `Viewer.on_gui` and
`Viewer.input`, and now re-exports the `gui` and `transforms` submodules. Without that re-export mypy
couldn't see `lr.gui`. `ShaderInterfaceError` and `VulkanValidationError` now subclass `RuntimeError`, as at runtime.
`gui.py` is typed Python, so it needs no stub.

**Build note:** new `python/lr/*.py` files are only copied into `build/python/lr` after a CMake
**reconfigure**: the copy uses a configure-time file list, and building only the `lr_python` target doesn't
re-run it.

**Files:** `src/python/LrModule.cpp` (`Key`/`MouseButton`/`Input`, the `_gui` submodule, `on_gui`/`input`,
input injection, the `assert()` abort routing, `OrbitCamera`); `SphericalCameraController.{hpp,cpp}`;
`python/lr/{gui.py, __init__.py, __init__.pyi}`;
`examples/python/interactive.py`, `shaders/lit.frag`; `tests/python/{test_bindings.py, test_failures.py}`.

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
  when the window closes (`run()` returning, or `step()` returning False) and at interpreter exit
  (`atexit`). As a result, a `Viewer` runs once. Creating a new `Viewer` also releases an earlier one
  that never finished.
- `PassContext` is only valid inside its execute callback; keeping it afterwards is undefined (documented
  in its docstring, not enforced).
- Push-constant sizes and descriptor bindings are checked against the shader since step 3.
- `python/lr/__init__.pyi` (type stubs) is maintained by the user. It declares the `lr` exceptions as
  `Exception` subclasses, but at runtime `ShaderInterfaceError` and `VulkanValidationError` subclass
  `RuntimeError`.
- **Crash tests:** never run native-crash tests (`_testing.crash`) without the debug-runtime setup from
  step 3b in place. Without it they pop modal dialogs that someone has to click away.
