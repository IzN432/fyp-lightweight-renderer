"""Type stubs for lr.engine: the engine's preassembled renderer (scenes, their GPU layout, IBL and the
deferred passes). Contracts are documented in docs/python_building_blocks.md."""

from __future__ import annotations

import os
from typing import Sequence

import numpy
from numpy.typing import NDArray

from . import Format, OrbitCamera, PassHandle, Viewer

__all__: list[str]

# ---------------------------------------------------------------------------
# Scenes (CPU side): what the engine's loaders produce
# ---------------------------------------------------------------------------

class Light:
    """A light's own parameters. Its position and orientation are its scene object's transform (the
    engine's light buffer uses that object's local position/rotation)."""

    @property
    def type(self) -> str:
        """'point', 'spot', 'area', 'directional' or 'image'."""
        ...
    @property
    def color(self) -> tuple[float, float, float]: ...
    @property
    def intensity(self) -> float: ...
    @property
    def inner_cone_degrees(self) -> float | None:
        """Spot lights only."""
        ...
    @property
    def outer_cone_degrees(self) -> float | None:
        """Spot lights only."""
        ...
    @property
    def range(self) -> float | None: ...
    @property
    def shadow_near_plane(self) -> float | None: ...
    @property
    def source_radius(self) -> float | None:
        """Spot lights only: emitter radius in world units, which widens the shadow penumbra."""
        ...
    @property
    def angular_radius_degrees(self) -> float | None:
        """Directional lights only: half the angle the emitter subtends, which widens the shadow
        penumbra with distance from the caster."""
        ...
    @property
    def area_size(self) -> tuple[float, float] | None:
        """Area lights only: (width, height) in world units."""
        ...
    @property
    def two_sided(self) -> bool | None:
        """Area lights only: emits from both faces (True) or only along its forward axis."""
        ...
    @property
    def spread_angle_degrees(self) -> float | None:
        """Area lights only: emission and shadow half-angle."""
        ...

class Material:
    """A material from the scene's material store. The loaders use the engine's names: parameters
    'baseDiffuse' (RGBA), 'baseEmissive' (RGB), 'baseRoughness', 'baseMetallic'; textures
    'baseColorTexture', 'normalTexture', 'metallicRoughnessTexture', 'emissiveTexture'. The engine's
    albedo is baseColorTexture * baseDiffuse."""

    @property
    def name(self) -> str: ...
    @property
    def parameters(self) -> dict[str, float | tuple[float, ...]]:
        """{name: float or colour tuple}, e.g. {'baseDiffuse': (r, g, b, a), 'baseRoughness': 0.5}."""
        ...
    @property
    def texture_names(self) -> list[str]:
        """Names of the textures this material has, e.g. ['baseColorTexture', 'normalTexture']."""
        ...
    def texture(self, name: str) -> NDArray[numpy.uint8]:
        """The texture's pixels as a (height, width, channels) uint8 array, ready for upload_image()."""
        ...

class Mesh:
    """Mesh data in the engine's layout. Render vertices are corners: each has a position (shared through
    position_indices) and its own attributes (normal, uv, …), so positions/attribute()/indices are
    directly usable as vertex and index buffers. Skinned meshes are in their rest pose."""

    @property
    def vertex_count(self) -> int: ...
    @property
    def face_count(self) -> int: ...
    @property
    def positions(self) -> NDArray[numpy.float32]:
        """(vertex_count, 3) float32: each render vertex's position."""
        ...
    @property
    def indices(self) -> NDArray[numpy.uint32]:
        """(face_count, 3) uint32 triangle vertex indices."""
        ...
    @property
    def face_materials(self) -> NDArray[numpy.uint32]:
        """(face_count,) uint32: each face's material handle (see Scene.material()); empty if the mesh has none."""
        ...
    @property
    def attribute_names(self) -> list[str]:
        """Per-vertex attributes this mesh has, e.g. ['normal', 'tangent', 'uv']."""
        ...
    def attribute(self, name: str) -> NDArray[numpy.generic]:
        """A per-vertex attribute: (vertex_count, components), e.g. 'normal' -> (V, 3) float32,
        'tangent' -> (V, 4) float32 (w = handedness), 'uv' -> (V, 2) float32. Raises KeyError if missing."""
        ...
    @property
    def unique_positions(self) -> NDArray[numpy.float32]:
        """(unique_position_count, 3) float32: positions shared between corners (for topology/geometry processing)."""
        ...
    @property
    def position_indices(self) -> NDArray[numpy.uint32]:
        """(vertex_count,) uint32: each render vertex's index into unique_positions."""
        ...

class Animator:
    """Animation playback for an imported animated object (advance it with Scene.update)."""

    @property
    def clip_names(self) -> list[str]: ...
    @property
    def playing(self) -> bool: ...
    def play(self, clip_index: int = 0) -> None: ...
    def pause(self) -> None: ...
    def stop(self) -> None: ...
    def seek(self, seconds: float) -> None: ...

class SceneObject:
    """An object in a Scene: a name, a place in the hierarchy, a transform, and optionally a mesh, a light
    or an animator."""

    @property
    def id(self) -> str:
        """The object's stable identity, as a UUID string. It survives saving and loading the scene,
        so it is safe to hold on to across a reload."""
    @property
    def name(self) -> str: ...
    @name.setter
    def name(self, value: str) -> None: ...
    @property
    def parent(self) -> SceneObject | None: ...
    @property
    def children(self) -> list[SceneObject]: ...
    @property
    def world_matrix(self) -> NDArray[numpy.float32]:
        """4x4 float32 object-to-world matrix (row-major numpy, acting on column vectors)."""
        ...
    @property
    def position(self) -> tuple[float, float, float]:
        """Local position (relative to the parent)."""
        ...
    @position.setter
    def position(self, value: Sequence[float]) -> None: ...
    @property
    def rotation(self) -> tuple[float, float, float, float]:
        """Local rotation as a unit quaternion (x, y, z, w)."""
        ...
    @rotation.setter
    def rotation(self, value: Sequence[float]) -> None: ...
    @property
    def scale(self) -> tuple[float, float, float]:
        """Local scale."""
        ...
    @scale.setter
    def scale(self, value: Sequence[float]) -> None: ...
    @property
    def mesh(self) -> Mesh | None:
        """The object's mesh, or None."""
        ...
    @property
    def light(self) -> Light | None:
        """The object's light, or None."""
        ...
    def set_light(
        self,
        type: str | None = None,
        color: Sequence[float] | None = None,
        intensity: float | None = None,
        size: Sequence[float] | None = None,
        inner_cone_degrees: float | None = None,
        outer_cone_degrees: float | None = None,
        range: float | None = None,
        shadow_near_plane: float | None = None,
        source_radius: float | None = None,
        angular_radius_degrees: float | None = None,
        two_sided: bool | None = None,
        spread_angle_degrees: float | None = None,
    ) -> None:
        """Change this object's light; parameters left as None keep their current values. Move or turn
        it with position/rotation. A SceneGpu showing the scene picks the change up on the next frame."""
        ...
    @property
    def animator(self) -> Animator | None:
        """The object's animation player, or None."""
        ...

class Scene:
    """A scene loaded with the engine's loaders: objects, meshes and materials, on the CPU. Build GPU
    buffers from it with ResourceRegistry.upload_buffer()/upload_image()."""

    def __init__(self) -> None: ...
    def save(self, path: str | os.PathLike[str]) -> None:
        """Save this complete scene to a single .lrscene file."""
        ...
    def load(self, path: str | os.PathLike[str]) -> SceneObject:
        """Load an OBJ, glTF or GLB file into this scene; returns the new root object it was placed under."""
        ...
    @property
    def objects(self) -> list[SceneObject]:
        """Every object, in creation order."""
        ...
    @property
    def roots(self) -> list[SceneObject]:
        """Objects without a parent."""
        ...
    def find(self, name: str) -> SceneObject | None:
        """The first object with this name, or None."""
        ...
    def material(self, handle: int) -> Material:
        """The material a mesh's face_materials entry refers to."""
        ...
    def update(self, dt: float) -> None:
        """Advance every playing animation by dt seconds (moves the animated objects' transforms)."""
        ...
    def add_light(
        self,
        type: str,
        color: Sequence[float] = (1.0, 1.0, 1.0),
        intensity: float = 1.0,
        position: Sequence[float] = (0.0, 0.0, 0.0),
        rotation: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
        size: Sequence[float] = (1.0, 1.0),
        inner_cone_degrees: float = 15.0,
        outer_cone_degrees: float = 30.0,
        range: float = 100.0,
        shadow_near_plane: float = 0.1,
        source_radius: float = 0.0,
        angular_radius_degrees: float = 0.0,
        two_sided: bool = True,
        spread_angle_degrees: float = 60.0,
        name: str = "Light",
    ) -> SceneObject:
        """Add a light object: 'point', 'spot', 'area', 'directional' or 'image' (environment lighting
        from an Ibl, scaled by color * intensity). It shines along its rotation's forward axis (spot,
        area, directional; area lights shine from both faces unless two_sided=False); rotation is a
        quaternion (x, y, z, w); size is an area light's (width, height). A SceneGpu showing this scene picks it up on the next frame."""
        ...
    def remove(self, object: SceneObject) -> None:
        """Remove an object and its descendants from the scene. A SceneGpu showing this scene stops
        drawing them, and removed lights stop lighting it, from the next frame. The Python objects
        remain but no longer appear in objects/roots."""
        ...

def load_scene(path: str | os.PathLike[str]) -> Scene:
    """Load an .lrscene file, OBJ, glTF or GLB file into a new Scene."""
    ...

# ---------------------------------------------------------------------------
# Engine building blocks: the C++ renderer's scene upload and passes
# (contracts in docs/python_building_blocks.md)
# ---------------------------------------------------------------------------

class ResourceUse:
    """A resource a building block reads or writes."""

    @property
    def name(self) -> str:
        """Its name in the frame graph / ResourceRegistry."""
        ...
    @property
    def kind(self) -> str:
        """'image' or 'buffer'."""
        ...
    @property
    def usage(self) -> str:
        """How the block uses it: images 'sampled', 'sampled_depth', 'sampled_array', 'storage',
        'color_attachment', 'depth_attachment' or 'uploaded'; buffers 'uniform', 'storage', 'vertex',
        'index' or 'indirect'."""
        ...
    @property
    def format(self) -> Format | None:
        """Image format, when known; None for buffers."""
        ...

class EnginePass:
    """Base class of the engine's passes and of Ibl. Constructing one declares its passes in the
    viewer's frame graph; inputs/outputs list what they read and write, taken from those declarations."""

    @property
    def name(self) -> str: ...
    @property
    def passes(self) -> list[PassHandle]:
        """Handles of the frame-graph passes it declared, e.g. for depends_on()."""
        ...
    @property
    def pass_names(self) -> list[str]: ...
    @property
    def inputs(self) -> list[ResourceUse]:
        """Resources it reads that something else must provide (another block, or your own pass)."""
        ...
    @property
    def outputs(self) -> list[ResourceUse]:
        """Resources it writes, for later passes to read."""
        ...
    def describe(self) -> str:
        """Inputs and outputs as readable text."""
        ...

class SceneGpu:
    """A Scene's GPU buffers in the engine's layout — what the engine's passes read: camera UBO,
    lights, mesh vertex/index buffers, materials and textures, skins. Kept in sync every frame (camera
    moves, animations, skinning). One per Viewer."""

    def __init__(self, viewer: Viewer, scene: Scene, camera: OrbitCamera) -> None:
        """Upload `scene` (its meshes and lights as they are now) and keep it in sync with `camera`.
        Every light also gets a quad mesh, drawn with the scene (bright for area lights, invisible
        otherwise). The quads belong to the SceneGpu: `scene` itself isn't modified and can be shown
        again later."""
        ...
    @property
    def camera_buffer(self) -> str:
        """Camera UBO: mat4 view, proj, viewProj, invView, invProj; vec4 position (std140, 336 bytes)."""
        ...
    @property
    def light_buffer(self) -> str:
        """Light SSBO, as pbr.frag reads it."""
        ...
    @property
    def num_lights(self) -> int:
        """Lights currently in the light buffer."""
        ...
    @property
    def max_lights(self) -> int:
        """Most lights the light buffer holds; more raises an error on the next frame."""
        ...
    @property
    def mesh_count(self) -> int: ...
    @property
    def buffers(self) -> dict[str, str]:
        """Names of the buffers it keeps, by role ('camera', 'lights', 'positions', 'attributes',
        'indices', 'face_groups', 'materials', 'joint_matrices'), for your own passes to bind."""
        ...

class Ibl(EnginePass):
    """Image-based lighting, precomputed once from an HDR environment (its compute passes run
    immediately, in their own frame graph). Outputs ibl_env, ibl_irradiance, ibl_prefiltered and
    ibl_brdf_lut. PbrPass lights the scene with it; CompositePass draws ibl_env as the sky."""

    def __init__(
        self,
        viewer: Viewer,
        hdri: str | os.PathLike[str] | None = None,
        env_res: int = 2048,
        irr_res: int = 32,
        pf_res: int = 2048,
        pf_mips: int = 8,
    ) -> None:
        """hdri: an equirectangular .hdr image (None: a black environment). env_res/pf_res: cubemap
        face sizes; irr_res: irradiance face size; pf_mips: prefiltered roughness levels."""
        ...
    @property
    def pf_mips(self) -> int: ...

class GeometryPass(EnginePass):
    """The engine's G-buffer pass: draws every SceneGpu mesh with its material into gbufferAlbedo,
    gbufferNormal, gbufferMaterial, gbufferEmissive and gbufferDepth."""

    def __init__(self, viewer: Viewer, scene_gpu: SceneGpu) -> None: ...
    @property
    def skinning(self) -> bool:
        """Draw skinned meshes posed (True, default) or in their rest pose."""
        ...
    @skinning.setter
    def skinning(self, value: bool) -> None: ...

class AmbientOcclusionPass(EnginePass):
    """Horizon-based ambient occlusion from gbufferDepth into hbao_ao (R32F occlusion: 0 = open,
    1 = fully occluded), then a bilateral blur. Parameters can change while running."""

    def __init__(
        self,
        viewer: Viewer,
        scene_gpu: SceneGpu,
        sphere_radius: float = 0.5,
        num_steps: int = 16,
        num_dirs: int = 8,
        tan_angle_bias: float = 0.364,
        ao_scalar: float = 2.0,
    ) -> None:
        """sphere_radius: world-space sampling radius (scale it to the model); num_steps/num_dirs:
        samples per direction / directions; tan_angle_bias: ignores horizons below this slope;
        ao_scalar: strength."""
        ...
    @property
    def sphere_radius(self) -> float: ...
    @sphere_radius.setter
    def sphere_radius(self, value: float) -> None: ...
    @property
    def num_steps(self) -> int: ...
    @num_steps.setter
    def num_steps(self, value: int) -> None: ...
    @property
    def num_dirs(self) -> int: ...
    @num_dirs.setter
    def num_dirs(self, value: int) -> None: ...
    @property
    def tan_angle_bias(self) -> float: ...
    @tan_angle_bias.setter
    def tan_angle_bias(self, value: float) -> None: ...
    @property
    def ao_scalar(self) -> float: ...
    @ao_scalar.setter
    def ao_scalar(self, value: float) -> None: ...

class PbrPass(EnginePass):
    """The engine's deferred lighting into pbr (RGBA16F, HDR): Cook-Torrance BRDF for the SceneGpu's
    lights (area lights via LTC). The Ibl lights the scene through 'image' lights (Scene.add_light),
    darkened by hbao_ao. Background pixels are black (CompositePass draws the sky there)."""

    def __init__(self, viewer: Viewer, scene_gpu: SceneGpu, ibl: Ibl) -> None: ...

class CompositePass(EnginePass):
    """Tone maps an HDR image (Reinhard) to the window, with the Ibl environment as the sky wherever
    GeometryPass drew nothing."""

    def __init__(
        self,
        viewer: Viewer,
        scene_gpu: SceneGpu,
        input: str = "pbr",
        output: str = "swapchain",
        output_format: Format | None = None,
        name: str = "composite",
    ) -> None:
        """input: the HDR image to show (PbrPass's 'pbr', or your own pass's output). output /
        output_format: where to write (default: the window)."""
        ...
