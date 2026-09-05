# Python API contract examples

These programs describe the intended Python API for the lightweight renderer. They
are executable specifications, not runnable examples yet: the
`lightweight_renderer` module has not been implemented.

The examples deliberately stop before the deferred PBR and ARAP pipeline. They
establish the lower-level contract that pipeline will eventually use:

1. `01_numpy_data.py` — create, update, and read back a generic buffer.
2. `02_compute_pass.py` — dispatch a custom compute shader over NumPy data.
3. `03_geometry_pass.py` — describe vertex input and render into an image.
4. `04_runtime_shader.py` — load shaders at runtime and report compilation errors.
5. `05_window_output.py` — use an opaque backbuffer and automatic presentation.
6. `06_replace_pass.py` — replace one pass while retaining the rest of a pipeline.
7. `07_load_gltf.py` — load a glTF asset into generic NumPy-backed mesh data.
8. `08_gltf_wireframe.py` — turn loaded triangle data into edges and display it.

## Contract established by these examples

- `Renderer` owns the Vulkan device and all low-level execution machinery. It can
  be used headlessly or create a window.
- Buffers and images are opaque, strongly typed handles. NumPy arrays and Python
  buffer objects are accepted without requiring a mesh abstraction.
- A `FrameGraphBuilder` describes work. `compile()` returns an immutable
  `CompiledFrameGraph`.
- Pass setup uses semantic resource operations. Python never supplies Vulkan
  layouts, stage masks, access masks, semaphores, or barriers.
- A pass callback receives a restricted `CommandEncoder`, not a Vulkan command
  buffer.
- A compiled graph retains strong references to setup and execution callbacks.
  Releasing the compiled graph releases those references. Callbacks cannot be
  changed during execution.
- `renderer.run(window, graph)` owns acquire, execution, submission, and present.
- Graph construction and shader failures raise subclasses of `lr.Error` with the
  pass, binding, resource, and shader context needed to fix the problem.
- Optional asset loaders return CPU-side data. Loading a model does not implicitly
  allocate GPU resources or require the renderer's scene/editor abstractions.

The API names are intentionally provisional. Changes should be made here before
binding large portions of the C++ API.

## Running these later

Once bindings exist, each file should run directly from the repository root:

```powershell
python examples/python/01_numpy_data.py
```

The examples should also be executed by the Python integration-test suite. Their
shaders live beside them so externally located shader files are exercised rather
than hidden engine shader paths.
