# SPlisHSPlasH integration plan

Working notes for integrating [SPlisHSPlasH](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH)
(MIT, v2.18.1) as a fluid simulation backend. Worktree: `../FYP-fluid-sim`, branch `fluid-sim`.

## What SPlisHSPlasH actually is

A **CPU-only, OpenMP-parallelised SPH solver**. The only GPU part is an optional neighbourhood
search (`cuNSearch`, needs CUDA). Pressure solve, viscosity and surface tension all stay on the CPU.
This single fact drives most of the decisions below.

Solvers available: WCSPH, PCISPH, PBF, IISPH, DFSPH, PF, ICSPH (`Simulation::setSimulationMethod`).

## Revised step order

The original plan had live-vs-offline as step 5. It has to move to the front, because the solver
being CPU-only largely settles it, and the answer determines what the consumption path in steps 3-4
even looks like. Likewise the renderer side (step 4) is the project's actual contribution and the
part most likely to be hard, so it should not sit behind three investigation steps.

### Step 0 — decide offline vs live (was step 5)

Assume **offline bake plus playback as the shipping path**, with live stepping as a
low-particle-count authoring preview. Rationale:

- CPU SPH with iterative pressure solvers (DFSPH/IISPH) is linear in particle count times solver
  iterations. Interactive rates are plausible only at low counts; it degrades to
  seconds-per-frame well before a million particles.
- Baking decouples the renderer's frame rate from solve cost, which is what makes step 6 (foam,
  spray, surface reconstruction) affordable at all.
- A bake also gives reproducible results for the writeup — same input, same frames.

Needs a measurement before committing: build the bundled `SPHSimulator` and time a bundled scene
on the target machine at a few particle counts. That number decides the live-preview budget.

### Step 1 — build integration

Two findings overturn the original approach:

- **There is no vcpkg port** (`splishsplash`, `partio`, `discregrid`, `compactnsearch` all 404).
  Every other third-party dep in this repo comes from `vcpkg.json`, so this is the first one that
  will not.
- **`add_subdirectory` is not viable.** `CMake/Common.cmake` sets `CMAKE_CXX_STANDARD 11`
  (clobbering our C++20), overwrites `CMAKE_CXX_FLAGS_RELEASE` globally with `/MD /fp:fast`,
  redirects `CMAKE_RUNTIME_OUTPUT_DIRECTORY` to `${CMAKE_SOURCE_DIR}/bin` (i.e. into *our* source
  tree) as `CACHE INTERNAL`, forces debug postfixes, and uses `add_definitions` / `link_libraries`
  for `-DEIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT`, `-DUSE_AVX` and OpenMP. There is also an unquoted
  `if (NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug")` that is a hard configure error when
  `CMAKE_BUILD_TYPE` is empty — the normal case for multi-config generators.

There are also **no `install()` rules and no package config anywhere** in the project, so
`find_package(SPlisHSPlasH)` does not exist as an option.

Plan: add it as a submodule under `external/SPlisHSPlasH`, but **build it out-of-tree** and consume
the artefacts through our own `IMPORTED` targets. Configure once with
`-DSPH_LIBS_ONLY=ON -DCMAKE_BUILD_TYPE=Release`; that switch skips GLFW/ImGui/NFD/pybind, `Tools`,
`Simulator` and `Tests`. Note `SPH_LIBS_ONLY` is never declared via `option()`, it is only tested
with `if (NOT SPH_LIBS_ONLY)`, so pass it as a plain cache var.

Libraries to wrap: `SPlisHSPlasH`, `Utilities`, `tinyexpr`, `MD5`, `partio`, `zlib`, plus
`CompactNSearch` and `Discregrid`. Watch the `_d` / `_rd` debug postfixes and the forced `/MD`.

Dependencies arrive by **build-time `ExternalProject_Add` git clone** (no FetchContent) into
`${CMAKE_BINARY_DIR}/extern/install`: `Discregrid`, `GenericParameters`, `PositionBasedDynamics`
and the neighbourhood search, each pinned to a commit. Consequences: the build needs network
access on first configure, and the include dirs do not exist at configure time. That last point is
the single most-reported problem for external consumers — see upstream issue #221, a developer
hitting exactly our use case, failing on `ParameterObject.h: No such file or directory`. It was
closed as a discussion rather than fixed. Our include path must cover the repo root plus
`build/extern/install/{GenericParameters,Discregrid,NeighborhoodSearch}/include` and Eigen.

Eigen3 is the only `find_package` it needs, and vcpkg can provide it — add `eigen3` to `vcpkg.json`.

Also: **this repo has no `README.md`.** The original plan's "add it to the README" cannot happen as
written. Either create one or document here; currently documenting here.

### Step 2 — feeding scene data in

Better than the original plan assumed. The JSON scene path is **optional** and lives in the app
layer we are excluding. The core API takes raw particle arrays directly:

    Simulation::init(particleRadius, sim2D);
    Simulation::addFluidModel(id, nParticles, Vector3r* positions, Vector3r* velocities,
                              unsigned int* objectIds, nMaxEmitterParticles);

`SimulatorBase::initFluidData` itself just passes `std::vector<Vector3r>::data()`. `Vector3r` is
`Eigen::Vector3f` by default (12 bytes, single precision). So our raw position data can go straight
in — no `.obj`, no JSON, no `SceneLoader`. `Simulation` is a singleton via `Simulation::getCurrent()`.

To sample from our existing meshes instead, `Utilities::VolumeSampling::sampleMesh` and
`PoissonDiskSampling::sampleMesh` are standalone and return `std::vector<Vector3r>`.

**The real cost here is boundaries, not fluid.** `RigidBodyObject` is abstract, and both usable
implementations (`StaticRigidBody`, and the PBD-backed dynamic one) live in `Simulator/` — the app
layer `SPH_LIBS_ONLY` excludes. We will need our own `RigidBodyObject` subclass bridging to the
existing `lr::PhysicsWorld` colliders. Budget real time for this. Boundary models:
`BoundaryModel_Akinci2012` (particle-based, simplest) vs `_Koschier2017` / `_Bender2019` (need
Discregrid SDF precomputation, also done app-layer).

### Step 3 — getting output back

For a renderer, skip the file exporters (app-layer anyway) and read memory directly:

    FluidModel::numParticles();
    FluidModel::getPosition(i);   // FORCE_INLINE over a contiguous std::vector<Vector3r>
    FluidModel::getVelocity(i);

`&model->getPosition(0)` is a valid contiguous base pointer — the library's own AVX paths rely on
this — so it is a straight `memcpy` into a Vulkan staging buffer.

**Trap:** `performNeighborhoodSearchSort()` reorders particles in memory for cache locality, so a
particle index does not identify a particle across steps. Anything needing stable identity (trails,
per-particle colour, foam seeding) must use the `objectId` array or disable sorting.

`FluidModel::addField` / `getFields` is a reflection system exposing per-particle arrays by name
(density, velocity, vorticity) — useful for shading inputs without hardcoding.

For the offline path, decide a bake format. Writing our own flat binary (frame header plus float3
array) is likely simpler than pulling in partio's `.bgeo`, and avoids partio's licence obligation.

### Step 4 — rendering the output

The actual contribution. Options, cheapest first:

1. Point/sphere splatting with depth — fast, reads as particles not fluid. Good first milestone.
2. Screen-space fluid rendering (van der Laan / Green): render sphere depth, bilateral-blur the
   depth buffer, reconstruct normals, shade with refraction and thickness. No mesh, fits our
   deferred pipeline, and the thickness pass composes with existing volume rendering.
3. Marching-cubes surface reconstruction — heavier, offline-only, nicer silhouettes.

Option 2 is the sweet spot and plugs into the existing G-buffer / deferred structure.

### Step 5 — feature wiring

Follow the established repo pattern: a `lr_fluid_feature` static library under
`src/features/fluid/`, with a `FluidComponent` plus solver bridge, codecs in the serialisation
target, and tests under `tests/fluid/`. Mirror `src/features/rigid_body/` closely — it is the
nearest analogue (third-party solver, component-based, with `PhysicsWorld` owning fixed-timestep
stepping and interpolation).

### Step 6 — foam, spray, bubbles

Last, as planned. Ihmsen et al. 2012 diffuse material: classify by trapped air / wave crest /
kinetic energy, advect secondary particles. Needs stable particle identity (see the step 3 trap).
Only worth it once step 4 looks right.

## Licensing

- SPlisHSPlasH: MIT.
- **partio: Disney 3-clause BSD, not MIT** — adds a binary attribution obligation (reproduce the
  notice in documentation). Avoidable by not using partio for the bake format.
- Discregrid / GenericParameters / PositionBasedDynamics / CompactNSearch believed MIT, zlib and
  tinyexpr zlib-licensed — **not yet verified, check before relying on it.**

## Open questions

- Measured solve cost on the target machine (gates step 0).
- Does the forced `/MD` conflict with our runtime library setting?
- Two-way coupling with `lr::PhysicsWorld`, or is one-way (rigid bodies push fluid) enough?
