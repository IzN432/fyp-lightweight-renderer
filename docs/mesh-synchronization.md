# Mesh synchronization

The mesh is canonical CPU data. GPU vertex/index buffers are independent materialized views
of that data, not something editor commands update directly. This adopts the useful Polyscope
separation between source data and render representations; it is not a port of its OpenGL backend.

## Frame lifecycle

```text
GUI / editing / undo / ARAP / analysis
    -> Mesh setters publish source revisions (no Vulkan calls)
    -> late update: SceneGpu::flushDirty (shared geometry buffers), then
       SceneManager (the selected mesh's editor overlays)
    -> MeshUploader compares each buffer's own dependency stamp
    -> stale views are packed once and queued in ResourceRegistry
    -> frame-graph execution flushes transfers, waits, then records rendering
```

`MeshBufferCache` records the revision successfully **queued**, not GPU completion. A failed
packing/upload callback does not advance the stamp. `ResourceRegistry` owns transfer completion.
Buffers are initially materialized during scene initialization. Inactive selected-mesh overlays
retain their own stamps and catch up when their editor mode becomes active.

## Which edits affect which buffers?

| CPU edit | GPU views checked/updated |
| --- | --- |
| `setPositionAt`, either `setPositions` overload | Shared geometry positions; selected points in Edit; selected heatmap positions in Analysis |
| Per-render-vertex attribute setter | Shared attribute buffer, only for attributes included in its config |
| Unique `color` setter (selection highlighting) | Selected points in Edit |
| Unique `heatmapColors` setter | Selected heatmap in Analysis |
| `setFaceGroups` | Shared face-group buffer |
| Same-sized `setTopology` | Render-domain positions/attributes and indices; face groups are reset |
| Ordinary object transform | Draw-time model matrix; does not rewrite mesh positions |
| Area-light geometry edit | CPU quad positions/attributes, then normal late synchronization |

Several mutations before preparation collapse into one pack/upload per stale view. No source
revision is globally cleared: updating geometry cannot consume an edit on behalf of points,
heatmaps, or analysis. Meshes outside the current selection are observed too. Shared buffers
still repack the entire configured mesh list when stale; this is not per-mesh suballocation.

The heatmap's unique colors are gathered through `positionIndices` while packing its render-vertex
buffer. There is no second per-render-vertex heatmap attribute to keep synchronized on the CPU.
Authored normals/tangents are preserved after position edits; regeneration remains explicit.
Laplace-Beltrami results independently compare position/topology revisions to report stale analysis.

## Mutation and structural boundaries

Mesh reads return const references/spans. Use setters instead of modifying returned data. Indexed
position batches validate all indices before writing, snapshot aliased inputs, and publish one
revision; unchanged positions do not advance the position revision. Attribute writes publish
their named attribute revision. Revisions are local to a mesh and caches also include its identity.
Registered mesh assets must remain at their `MeshStore` addresses; do not move-assign a registered
asset in place. Use its setters or register a new asset and explicitly rebuild scene bindings.

Content synchronization keeps allocations, counts, layouts, and draw ranges fixed. A changed
per-mesh count or buffer layout throws instead of silently retaining invalid offsets. For scene
imports or topology/size changes, rebuild geometry explicitly (`SceneGpu::rebuildGeometry`). Passes
registered through `SceneGpu::onGeometryRebuilt` refresh their draw ranges; replaced resources are
retired safely and the frame graph recompiles descriptors that referenced them. Lights added to or
removed from the scene trigger this rebuild automatically. The rebuild also refreshes the selected
mesh's overlays and the selection's position span (`SceneManager`). Callers must rebind editor
features that retain topology or selection-derived state. Switching the edited object uses the
replace/rebind path.

Skin influence CSR is deliberately structural: changing a skinned mesh's topology/weights or its
skin binding requires a geometry rebuild. Palette updates reject stale structural source data.
Position-only edits do not rebuild influences. Moving/resizing an existing light quad preserves
its connectivity, so it is a content edit rather than a topology edit.

## Vulkan scope

Static uploads still use staging buffers and a blocking graphics-queue submission/fence wait.
Buffer copies have dependencies before and after transfer, copy only the staged byte count, and
pending copies targeting a replaced allocation are discarded. This is conservative synchronization,
not an asynchronous uploader. Replacements retire the old allocation until the frames in flight
that may use it have completed (`ResourceRegistry::retire`), so callers need not wait for the GPU.

Persistently mapped camera/material/light/joint buffers are dynamic buffers with one copy per frame
in flight, so host writes never overlap a frame that is still reading. Transfer batching without
blocking waits, redraw caching, and partial-range uploads remain follow-up work.

`mesh.buffer_sync` tests revision mutation, atomic batch validation, aliasing, independent consumers,
edit coalescing, inactive-view catch-up, seam expansion, failure/retry, and structural replacement.
It is a CPU-only test. The opt-in `mesh_gpu_sync_smoke` executable creates a headless Vulkan
context with validation requested, checks device readback for initial/partial/replaced uploads,
and exercises `MeshUploader` synchronization and replacement. Build/run it explicitly:

```powershell
cmake --build build --config Debug --target mesh_gpu_sync_smoke
.\build\Debug\mesh_gpu_sync_smoke.exe
```

This smoke check does not exercise interactive drawing or the multi-frame mapped-buffer lifetime.
