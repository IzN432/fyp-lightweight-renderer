# Fluid rendering roadmap

This document splits fluid work into two independent features:

1. a Lagrangian liquid-data player rendered from reference particles; and
2. an Eulerian gas solver rendered by volume ray marching.

They share renderer conventions, camera/depth resources, controls, and profiling, but not a solver
interface. A particle set and a sampled grid have different lifetimes and access patterns; forcing
both through one abstraction would make either feature harder to validate.

## Scope and ordering

The liquid track comes first. It uses the existing `FluidSolverCpu` as a deterministic producer,
then treats its output as immutable test data. This separates renderer debugging from simulation
debugging and gives every visual regression a reproducible input.

The gas track begins after the renderer can composite a custom fluid layer correctly. It first adds
the missing 3D-resource support to the framegraph, then builds a small CPU reference solver and a GPU
solver that can be checked against it.

Both tracks should start with one deliberately small canonical scene and grow only after that scene
is correct. Performance work comes after numerical and visual baselines exist.

---

# Track 1: Lagrangian liquid reference-data player

## Goal

Run a CPU particle simulation ahead of rendering, derive liquid and secondary-particle attributes,
write a versioned cache, and play that cache through a screen-space particle renderer. From the
renderer onward, cached frames—not a live solver—are the normal test input.

The first useful result is visible camera-facing spheres/billboards. The final target is:

- a smooth liquid surface reconstructed in screen space;
- foam and spray drawn as secondary billboards;
- bubbles visible within the liquid using depth-aware billboards; and
- deterministic playback suitable for screenshots, profiling, and regression tests.

## Architecture and ownership

Add the following under `src/features/fluid/`:

- `LiquidParticleFrame`: one decoded frame of render-facing particle data.
- `LiquidBakeWriter` / `LiquidBakeReader`: versioned binary cache I/O.
- `LiquidPlayback`: frame selection, interpolation, looping, and playback time.
- `LiquidDerivedParticles`: classifies/generates foam, spray, and bubbles during the CPU bake.
- `LiquidParticleUploader`: owns render buffers and updates them from a playback frame.
- `LiquidRenderPass`: owns the liquid screen-space pass chain.

Keep `FluidSolverCpu` independent of file I/O and Vulkan. A separate bake command or test fixture
drives it, extracts a render frame, generates secondary particles, and passes the result to the
writer. This makes it possible to replace the simple PBF solver with SPlisHSPlasH later without
changing the cache reader or renderer.

## Render-facing data contract

Use a little-endian binary format with an explicit magic value and version. Do not dump C++ structs
directly because GLM padding, compiler ABI, and future fields would make those files fragile.

File header:

- magic and format version;
- coordinate convention and units (`+Y` up, metres);
- fixed simulation step and capture rate;
- frame count and frame-table offset;
- liquid particle radius and smoothing radius;
- feature flags describing optional channels;
- bake configuration hash and optional source-scene identifier.

Per-frame table entry:

- timestamp;
- byte offset and compressed/uncompressed size;
- counts for liquid, foam, spray, and bubble particles;
- world-space bounds;
- checksum for corruption and deterministic-test detection.

Per liquid particle, initially store:

- stable `uint32` ID;
- `float3` position and velocity;
- density;
- optional curvature/free-surface score.

Per secondary particle, store:

- stable `uint32` ID;
- `float3` position and velocity;
- radius, age, and lifetime;
- type (`foam`, `spray`, or `bubble`);
- opacity or confidence.

Use structure-of-arrays on disk and in the upload staging representation. It lets a shader bind only
the channels a pass needs. The first implementation may remain uncompressed; add per-frame LZ4 or
Zstd only after profiling proves disk bandwidth or file size is a problem.

Stable IDs are mandatory even if a future CPU solver reorders particles for neighbour-search cache
locality. Playback interpolation and temporal tests must match particles by ID, never by array index.

## CPU bake pipeline

### Phase L1 — deterministic base simulation

Extend the existing CPU reference path without coupling it to the UI:

1. describe a canonical emitter/container scene in a small bake configuration;
2. create particles with deterministic ordering;
3. run a fixed time step (begin with `1/120 s`);
4. capture at a fixed rate (begin with 60 frames/s);
5. write position, velocity, density, and stable IDs;
6. record solver settings and a configuration hash in the cache.

The current brute-force neighbour search is appropriate for correctness tests but not for large
bakes. Replace it behind `findNeighbours()` with a uniform-grid search once both implementations
produce equivalent neighbour sets on small inputs. Preserve the brute-force implementation as the
test oracle.

### Phase L2 — secondary-particle generation

Derive secondary material during the bake, not during rendering. This keeps test frames stable and
lets generation use solver neighbourhood information.

Classify source regions from a weighted combination of:

- low neighbour count / free-surface score;
- trapped-air potential from converging neighbours;
- wave-crest potential from surface curvature and vertical acceleration;
- kinetic energy and velocity relative to the local mean;
- depth below the reconstructed surface.

Then apply these behaviours:

- **foam:** spawn near energetic free surfaces; advect near the local liquid velocity; short-to-medium
  lifetime; float toward the surface;
- **spray/splash:** spawn at high-energy crests; integrate ballistically with gravity and drag; kill
  or convert to foam on re-entry;
- **bubbles:** spawn from trapped-air potential below the surface; advect with fluid velocity plus
  buoyancy; shrink/fade or convert to foam at the surface.

Use seeded random sampling derived from `(bake seed, frame, source particle ID)` so a repeated bake is
byte-for-byte reproducible. Put spawn thresholds in the bake configuration and write them into the
metadata.

## Playback and upload

`LiquidPlayback` owns a small decode window: previous, current, and next frames. It selects frames
from render time and interpolates matching IDs. New or deleted secondary particles fade rather than
pop. Seeking clears interpolation history.

`LiquidParticleUploader` packs GPU records as aligned `vec4` values to avoid `float3` layout traps:

- position + radius;
- velocity + type/flags encoded separately;
- age/lifetime/opacity/auxiliary scalar;
- stable ID where picking or debugging needs it.

Begin with `ResourceRegistry::replaceUploadedBuffer` on count/capacity changes and `reuploadBuffer`
when the allocation is large enough. Before optimizing, measure whether its staging/upload path is
acceptable for the chosen particle count. If not, add a persistently mapped, per-frame streaming
buffer API rather than inserting device-idle waits.

## Screen-space renderer

Implement the renderer incrementally so every pass has a visible debug mode.

### Phase L3 — reference billboards

Draw one camera-facing quad per particle, generated from `gl_VertexIndex` and `gl_InstanceIndex`, so
no quad mesh or index buffer is needed. In the fragment shader:

- discard outside the projected circle;
- reconstruct the view-space sphere normal and surface position;
- write the sphere's corrected depth rather than the quad depth;
- colour by particle type or a diagnostic channel.

This phase validates cache playback, buffer layout, camera transforms, radius scaling, occlusion
against `gbufferDepth`, and instance counts. Keep it permanently as a debug visualization.

### Phase L4 — liquid surface reconstruction

Build a pass chain in this order:

1. **particle depth:** nearest sphere surface into single-sample `R32_SFLOAT` plus a validity mask;
2. **thickness:** additive front/back path-length approximation into `R16_SFLOAT`;
3. **depth smoothing:** separable bilateral or narrow-range filter, rejecting background and large
   depth discontinuities;
4. **normal reconstruction:** screen-space derivatives of the smoothed view-space position;
5. **liquid shading/composite:** Fresnel reflection, absorption from thickness, refraction of the
   already-lit scene, environment reflection, and optional direct-light specular.

Do not put liquid into the opaque G-buffer. It is a separate HDR layer composited after opaque PBR
and before editor overlays/final presentation. Copy or sample the opaque lit scene into a distinct
input resource; a pass must never sample and write the same image without an explicitly supported
feedback mechanism.

Start with one liquid layer and no self-sorting. The nearest surface depth plus accumulated thickness
is enough for the target effect and avoids attempting general transparent-particle ordering.

### Phase L5 — foam, spray, and bubbles

- Draw foam and spray after liquid shading as premultiplied-alpha billboards.
- Foam samples the liquid surface depth and is clipped/softened around the surface.
- Spray tests against opaque and liquid depth and uses velocity-stretched billboards as an optional
  improvement.
- Bubbles are drawn before the liquid composite or folded into its optical term. They must be clipped
  to pixels where the bubble depth lies between the liquid front surface and estimated back surface;
  a plain alpha overlay would incorrectly show bubbles outside the liquid.
- Use soft depth intersections and stochastic/blue-noise alpha for dense secondary particles.

## Liquid tests and acceptance gates

Unit tests:

- cache header/version rejection, truncated data, frame seeking, and round-trip equality;
- deterministic bake hash and deterministic secondary spawning;
- interpolation by stable ID with births/deaths;
- CPU billboard projection/depth math at centre, edge, and near-plane cases.

GPU/framegraph tests:

- shader-interface test for every pass;
- graph topology proves depth -> smooth -> normal/shade -> secondary/composite ordering;
- tiny offscreen smoke test contains finite depth/thickness and non-empty output;
- resize and buffer-capacity growth do not invalidate descriptors.

Visual acceptance scenes:

1. still particle sphere grid (projection and depth);
2. dam break with no secondary particles (surface continuity);
3. falling stream into a pool (thickness and refraction);
4. energetic splash (spray and foam);
5. underwater/bubble-heavy frame (bubble clipping).

Track performance separately for decode, CPU-to-GPU upload, each render pass, and total GPU time.
Define the target particle count only after the baseline hardware is measured; do not hide a low
upload ceiling by reducing the test scene.

## Liquid delivery milestones

- **L0:** freeze cache v1 and generate a 2–3 second canonical bake.
- **L1:** cache reader/player with ID-based interpolation and inspection tests.
- **L2:** debug particle billboards with correct sphere depth and opaque occlusion.
- **L3:** smoothed liquid surface with thickness, normals, and a debug-mode selector.
- **L4:** physically motivated composite using scene colour, depth, and environment.
- **L5:** deterministic foam, spray, and bubble bake plus depth-aware billboard rendering.
- **L6:** performance pass, visual baselines, documentation, and representative captures.

---

# Track 2: Eulerian gas solver and ray-marched renderer

## Goal

Simulate smoke/gas on a fixed Eulerian grid and render density, temperature, and emission by
ray marching the volume. Build a small CPU solver first as a numerical reference, then execute the
interactive solver on Vulkan compute.

The minimum solver is incompressible, buoyant smoke in a rectangular domain. Combustion, sparse
grids, moving obstacles, and two-way rigid-body coupling are later extensions, not prerequisites.

## Numerical model

Use a staggered Marker-and-Cell (MAC) grid:

- scalar cell centres: density, temperature, pressure, divergence, obstacle mask;
- velocity faces: `u`, `v`, and `w`;
- configurable cell size, domain origin, and resolution;
- solid boundary face velocities stored separately when moving obstacles arrive.

One fixed simulation step executes:

1. inject density, temperature, and source velocity;
2. advect velocity;
3. add buoyancy and optional vorticity confinement;
4. enforce solid boundary velocities;
5. compute divergence;
6. solve the pressure Poisson equation;
7. subtract the pressure gradient (projection);
8. advect density and temperature;
9. apply dissipation/cooling and clamp non-finite values.

Start with semi-Lagrangian RK2 backtracing and trilinear sampling. It is dissipative but stable and
easy to compare. Add MacCormack/BFECC with neighbourhood clamping only after the reference tests pass.
Use a fixed simulation step independent of render delta and cap the number of catch-up steps.

For pressure, implement red-black Gauss-Seidel on the CPU reference and ping-pong Jacobi on the GPU
first. Jacobi is not the fastest choice, but it maps cleanly to compute and provides a clear baseline.
Only consider multigrid or PCG when profiling shows pressure projection dominates and the residual
metric is already trustworthy.

## Gas architecture

Add a separate feature subtree, for example `src/features/gas/`:

- `GasGridDesc`: resolution, cell size, origin, and boundary mode.
- `GasSolverCpu`: small-grid reference implementation.
- `GasSolverGpu`: owns volume resources and dispatch sequencing.
- `GasEmitter`: source shape and rates, initially sphere and box.
- `GasSimulationPasses`: framegraph compute-pass construction.
- `GasRaymarchPass`: volume shading and HDR compositing.
- `GasDebugSlicePass`: scalar/vector slices and residual views.

The public solver output is a `GasVolumeView` containing density, temperature, velocity, bounds, and
the simulation-to-world transform. The renderer consumes that view, not the solver object.

## Required renderer infrastructure

The current `ResourceRegistry` supports 2D images, arrays, and cubemaps but not 3D images. Before the
GPU solver or `sampler3D` ray marcher, add:

- persistent and transient 3D image registration (`VkExtent3D`, `VK_IMAGE_TYPE_3D`, 3D image view);
- extent/format/usage validation for 3D resources;
- storage-image and sampled-image descriptor support for 3D views;
- resize/replacement and deferred-destruction coverage;
- framegraph tests for compute write -> compute read -> fragment sample barriers;
- shader compilation/interface tests using `image3D` and `sampler3D`.

Do not overload `ExtentSpec`'s swapchain-relative 2D meaning. Introduce a distinct volume extent or a
dimension-aware resource descriptor. This avoids accidental volume reallocation on window resize.

For the first solver, use `R16_SFLOAT` scalar volumes where precision is adequate and
`R16G16B16A16_SFLOAT` for packed velocity/scratch data if separate MAC-face images prove cumbersome.
The CPU reference must remain a true MAC grid even if the first GPU storage layout packs fields for
implementation convenience.

## GPU compute passes

Build each step as explicit framegraph passes with ping-pong resources:

- `gas_inject`;
- `gas_advect_velocity`;
- `gas_buoyancy`;
- optional `gas_vorticity` and `gas_vorticity_force`;
- `gas_divergence`;
- `gas_pressure_N` repeated for the configured iteration count;
- `gas_project`;
- `gas_advect_scalars`;
- optional `gas_dissipate` if it is not folded into advection.

Dispatch in 3D workgroups (begin with `8x8x8`) and bounds-check every invocation. Keep previous and
next fields distinct whenever a pass samples neighbours. Expose debug resource names consistently so
tests and the slice viewer can inspect any stage.

Pressure convergence is an acceptance criterion, not just an iteration count. Periodically compute
or read back an L2/max residual in debug/test builds. Projection must demonstrably reduce divergence.

## Ray-marched gas renderer

Render a proxy box for the volume or use a fullscreen pass with ray/AABB intersection. A fullscreen
pass is simplest for one volume and naturally handles a camera inside the domain.

For each pixel:

1. reconstruct the world-space camera ray;
2. intersect it with the gas world-space AABB;
3. clamp the exit distance using opaque scene depth;
4. jitter the first sample with blue noise to suppress banding;
5. sample density and temperature from the 3D textures;
6. integrate front-to-back transmittance and in-scattered/emissive radiance;
7. terminate when transmittance is negligible or the ray exits the volume;
8. output premultiplied HDR colour and opacity into the fluid/volume layer.

Begin with Beer-Lambert absorption and a constant or Henyey-Greenstein phase term. Add one directional
or scene light only after the unlit density rendering is correct. For volumetric shadows, first use a
short secondary march toward the dominant light at a lower step count; later cache light
transmittance in a volume if it is too expensive.

Temperature maps to black-body-like emission through an artist-controlled ramp, not literal Kelvin
colour in the first version. Keep density extinction, albedo, anisotropy, emission scale, step size,
and maximum steps visible in the settings UI.

Composite gas after opaque PBR and before ordinary transparent meshes unless a later order study
shows the reverse is required. Correct interleaving with arbitrary transparent surfaces is outside
the first milestone; document that limitation rather than silently producing inconsistent ordering.

## Gas tests and acceptance gates

CPU numerical tests:

- constant and linear field trilinear sampling;
- advection of a translated density blob;
- zero-velocity invariance and dissipation bounds;
- closed-boundary no-through-flow;
- pressure residual decreases;
- projection lowers max/L2 divergence by a defined factor;
- no NaN/Inf through a long small-grid run;
- total density changes only by configured source/dissipation within tolerance.

GPU comparison tests on small grids:

- each individual GPU stage compared with its CPU equivalent;
- pressure and projected divergence compared by tolerance, not bit equality;
- odd/non-workgroup-multiple grid sizes exercise bounds checks;
- barriers and ping-pong selection verified through framegraph topology tests.

Raymarch tests:

- empty volume is transparent;
- uniform-density box matches analytic Beer-Lambert transmittance within tolerance;
- opaque depth truncates the ray;
- camera outside and inside the volume both render finite results;
- step-size changes converge rather than changing brightness arbitrarily.

Visual acceptance scenes:

1. static uniform cube (optical integration);
2. rising smoke plume (buoyancy and projection);
3. side wind crossing a plume (velocity advection);
4. obstacle in the flow (boundary handling, later milestone);
5. hot emissive source (temperature rendering).

Track GPU time per simulation stage, raymarch time, grid memory, pressure residual, maximum divergence,
and average ray steps. Default quality must fit the measured target GPU; expose resolution and render
step size independently so simulation cost and rendering cost can be studied separately.

## Gas delivery milestones

- **G0:** write CPU MAC-grid reference with sampling, advection, projection, and numerical tests.
- **G1:** add general 3D image resources and barrier/interface tests to the renderer.
- **G2:** render a static uploaded density volume with depth-clipped ray marching.
- **G3:** reproduce CPU injection/advection/buoyancy on GPU at small resolutions.
- **G4:** add GPU pressure projection with residual/divergence validation.
- **G5:** run the complete fixed-step GPU simulation and debug-slice UI interactively.
- **G6:** add scattering, temperature emission, blue-noise jitter, and quality controls.
- **G7:** profile, tune formats/workgroups/iterations, capture visual baselines, and document limits.

---

# Shared integration decisions

## Composition order

Use explicit HDR layers rather than modifying `TransparentPass` into a catch-all:

1. opaque geometry and deferred PBR;
2. gas volume layer;
3. reconstructed liquid surface;
4. liquid secondary particles;
5. ordinary transparent meshes (with the ordering limitation documented);
6. final environment/background composite and editor overlays.

The exact ordering of gas, liquid, and transparent meshes should be configurable during development.
Their first versions will not perform per-fragment order-independent transparency, so overlapping
volumes are a known limitation.

## Controls and diagnostics

Provide one Fluid settings panel with separate Liquid Playback and Gas Simulation sections:

- pause/play/step/reset and deterministic frame index;
- cache path, loop mode, interpolation, and particle counts;
- per-pass liquid debug views (raw depth, smooth depth, thickness, normals, type);
- gas resolution, fixed step, pressure iterations, source controls, and reset;
- gas slices for density, temperature, pressure, divergence, and velocity magnitude;
- render controls and per-stage timings.

No solver should advance when the application is paused or while a one-frame diagnostic capture is
being compared. Separate simulation time from wall-clock/render time.

## Recommended implementation order

The shortest path to evidence is:

1. L0–L2: deterministic liquid data and correct particle billboards;
2. L3–L4: liquid surface and composition;
3. G0–G2: CPU gas reference, 3D resource support, and static volume raymarch;
4. G3–G5: GPU gas simulation;
5. L5: secondary liquid material once the base surface is stable;
6. G6 plus joint performance/quality work.

This ordering produces a renderable result early, proves the framegraph composition path before the
gas solver depends on it, and prevents simulation defects from being mistaken for rendering defects.

## Explicit non-goals for the first complete version

- live high-particle-count CPU liquid simulation;
- a shared particle/grid solver abstraction;
- liquid marching cubes;
- gas combustion chemistry;
- sparse or adaptive gas grids;
- fully correct ordering of intersecting liquid, gas, and transparent meshes;
- two-way fluid/rigid-body coupling;
- production compression or streaming before baseline measurements exist.

