# Offline fluid simulations

`OfflineFluidSimulation` is the engine-native representation of an immutable baked particle clip.
External files are import formats only: components and `.lrscene` files never retain a Blender or
VDB path.

## Data model

A clip owns:

- a display name, frame rate, and suggested particle radius;
- a sorted frame table containing playback time, original source-frame number, sample offset/count,
  and bounds;
- one frame-major contiguous `float3` position array;
- an optional, equally sized `float3` velocity array.

The velocity channel is all-or-nothing for a clip. Frame times are strictly increasing and frame
ranges exactly partition the sample arrays. This makes corruption detectable and keeps one frame a
single span suitable for upload. Particle order has no identity across frames.

`OfflineFluidSimulationStore` owns clips at stable addresses. Runtime components hold dense
`OfflineFluidSimulationHandle` values; `.lrscene` persistence uses UUIDs and resolves them back to
handles during load. The file stores frame metadata in its JSON manifest and sample arrays in the
existing raw binary payload. At 12 bytes per position sample, file cost is deterministic before any
future compression is added. Positions plus velocities cost 24 bytes per particle-frame; the
renderer expands the selected frame to a 32-byte `ParticleGpuData` record in GPU memory.

As a concrete reference — a denser real Blender bake than the committed sample below, not checked
in — `fluid_data_0001.vdb` in that cache contains 6,776 liquid
particles and is 73,080 bytes on disk. Its uncompressed `.lrscene` sample payload is 81,312 bytes
for positions only or 162,624 bytes with velocities, plus a small JSON frame record and container
header. The matching gzip-compressed `.uni` file is 79 bytes (204 bytes after decompression).

`FluidComponent` is a tagged `Live`/`Baked` value. A baked component contains only its clip handle,
playback controls, and per-instance appearance. A particle radius of zero uses the clip's imported
suggestion.

## Playback

Baked playback expands the selected frame into the renderer's existing 32-byte
`ParticleGpuData` records and uploads only when the frame or appearance changes. The live GPU solver
and baked playback publish the same named buffer, so screen-space liquid rendering is unchanged.

Playback intentionally uses discrete frames. Blender/Mantaflow can add, delete, and spatially
reorder particles, so matching array index `i` between neighbouring frames would create false
trajectories. A later interpolation mode must use velocities and explicitly handle births/deaths.

Baked playback reads positions only. The velocity channel is groundwork for that interpolation
mode, so importing it is **off by default** (`importVelocities`) — enabling it would double clip
memory and `.lrscene` size for data nothing currently renders. The channel, its persistence and its
all-or-nothing invariant all remain in place; only the import default changed.

## Importers

- `VdbFluidImporter` reads one OpenVDB `PointDataGrid` (default `particles`), including `P` and,
  when `importVelocities` is set, the optional `particles_velocity` attribute, into a one-frame
  clip. Coordinates stay in the file's own axes: OpenVDB has no up-axis convention, so the generic
  reader applies no rotation and a format-specific importer owns that decision.
- `BlenderUniImporter` reads Blender's gzip-compressed 204-byte Mantaflow domain configuration,
  reporting every field in Blender's own axes — it is a file reader, not a conversion.
- `BlenderFluidCacheImporter` pairs `data/fluid_data_####.vdb` with
  `config/config_####.uni`, converts frames to domain-centred local coordinates, and creates one
  clip. The source MAC velocity volume is deliberately ignored; it is solver state rather than
  render data.

### Up-axis conversion

Blender is Z-up and this engine is Y-up (`SphericalCameraController` and the shadow passes all take
+Y as world up), so `BlenderFluidCacheImporter` rotates imported coordinates by -90° about X:

```
(x, y, z)_blender  ->  (x, z, -y)_engine
```

This is a rotation, not a mirror. Negating a single axis would also put height on +Y but would flip
handedness, turning the clip into its own reflection. Positions and velocities take the same
transform, since both are directions in Blender's space. Centring (`centerDomain`) happens first,
in Blender's axes, because the domain metadata it uses is authored there.

`convertUpAxis` turns it off for inspecting a cache as authored; a clip imported that way renders
lying on its side. Without this conversion a liquid bake falls sideways on screen — which is what
the reference-cache test asserts against, by checking that the settling fluid's top drops along +Y
while the horizontal extents stay pinned to the domain walls.

## The shipped sample

`assets/samples/fluid/dna_helix` is a committed 24-frame Blender-format cache: a rotating DNA
double helix of 1,808 particles, authored on a 64³ domain at 24fps so the clip is exactly one
second and loops seamlessly. It is about 1.5 MB, tracked in Git LFS alongside the other binary
sample assets.

It exists because the only other cache this code was developed against was a local Blender bake
that no other machine has. The sample makes the import path reproducible: `fluid.importers` reads
it on every machine (skipping only if LFS content has not been fetched), and it is what the
editor's **Import Fluid...** button opens by default.

`tools/fluid/GenerateDnaHelixCache.cpp` writes it, via the `generate_dna_helix_cache` target
(`EXCLUDE_FROM_ALL` — the output is checked in, so it only runs when the sample changes):

```
cmake --build build --target generate_dna_helix_cache --config Release
./build/Release/generate_dna_helix_cache assets/samples/fluid/dna_helix
```

The generator writes real OpenVDB `PointDataGrid` files and Blender's 204-byte gzip-compressed
Mantaflow configuration records, so the sample goes through `BlenderFluidCacheImporter` by exactly
the path a Blender bake does rather than through a shortcut. Positions are written in VDB world
units with the origin at the domain's lower corner, as Blender writes them, and the importer's
`centerDomain` re-centres them.

Crucially the sample is authored in **Blender's Z-up axes**, with the helix running along +Z and
rotating about Z, so importing it exercises the up-axis conversion above. Authoring it Y-up would
produce a file no Blender would ever write and would hide that conversion — the sample's own test
asserts the imported clip's long axis lands on +Y. The VDBs also carry a velocity attribute, as a
real bake does, even though the importer skips it by default; it is the rigid-rotation velocity, so
it stays consistent with the motion rather than being filler.

## Importing in the editor

**Import Fluid...** beside the hierarchy's **Import...** button opens a directory picker (a cache is
a directory of per-frame files, not a single file). Confirming it imports the cache into the scene's
clip store and creates a new object carrying a `Baked` `FluidComponent`, then selects it — the fluid
runtime follows the selected object carrying a `FluidComponent`, so playback starts immediately. The
cache path is not retained: the object saves into an `.lrscene` with the samples embedded.

The Blender importer currently assumes primary liquid particles. Secondary foam, spray, and bubble
VDB streams can later become an optional particle-type channel. Object rotation/non-uniform scale
also remain outside the fluid render contract, which currently applies translation and a uniform
render scale only.
