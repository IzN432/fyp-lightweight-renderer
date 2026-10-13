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

As a concrete reference, `fluid_data_0001.vdb` in the supplied Blender cache contains 6,776 liquid
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

## Importers

- `VdbFluidImporter` reads one OpenVDB `PointDataGrid` (default `particles`), including `P` and the
  optional `particles_velocity` attribute, into a one-frame clip.
- `BlenderUniImporter` reads Blender's gzip-compressed 204-byte Mantaflow domain configuration.
- `BlenderFluidCacheImporter` pairs `data/fluid_data_####.vdb` with
  `config/config_####.uni`, converts frames to domain-centred local coordinates, and creates one
  clip. The source MAC velocity volume is deliberately ignored; it is solver state rather than
  render data.

The Blender importer currently assumes primary liquid particles. Secondary foam, spray, and bubble
VDB streams can later become an optional particle-type channel. Object rotation/non-uniform scale
also remain outside the fluid render contract, which currently applies translation and a uniform
render scale only.
