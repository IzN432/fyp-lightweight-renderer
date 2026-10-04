"""Type stubs for the lr native extension (_lr)."""

from __future__ import annotations

import enum
import os
import pathlib
from typing import Callable, Sequence, overload

import numpy
from numpy.typing import NDArray

# Anything exposing the buffer protocol: numpy arrays, bytes, bytearray, memoryview.
from collections.abc import Buffer

# Pure-Python submodules imported by lr/__init__.py (typed in their own .py files).
from . import gui as gui, transforms as transforms

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Format(enum.Enum):
    UNDEFINED = ...
    R8_UNORM = ...
    R8G8B8A8_UNORM = ...
    R8G8B8A8_SRGB = ...
    B8G8R8A8_UNORM = ...
    B8G8R8A8_SRGB = ...
    R16_SFLOAT = ...
    R16G16_SFLOAT = ...
    R16G16B16A16_SFLOAT = ...
    R16G16B16A16_UNORM = ...
    R32_SFLOAT = ...
    R32G32_SFLOAT = ...
    R32G32B32_SFLOAT = ...
    R32G32B32A32_SFLOAT = ...
    R32_UINT = ...
    D32_SFLOAT = ...

class BufferUsage(enum.IntFlag):
    TRANSFER_SRC = ...
    TRANSFER_DST = ...
    UNIFORM = ...
    STORAGE = ...
    INDEX = ...
    VERTEX = ...
    INDIRECT = ...

class ImageUsage(enum.IntFlag):
    TRANSFER_SRC = ...
    TRANSFER_DST = ...
    SAMPLED = ...
    STORAGE = ...
    COLOR_ATTACHMENT = ...
    DEPTH_STENCIL_ATTACHMENT = ...

class Stage(enum.IntFlag):
    VERTEX = ...
    FRAGMENT = ...
    COMPUTE = ...
    ALL_GRAPHICS = ...

class Topology(enum.Enum):
    POINT_LIST = ...
    LINE_LIST = ...
    LINE_STRIP = ...
    TRIANGLE_LIST = ...
    TRIANGLE_STRIP = ...

class LoadOp(enum.Enum):
    LOAD = ...
    CLEAR = ...
    DONT_CARE = ...

class BlendMode(enum.Enum):
    OPAQUE = ...
    ALPHA = ...
    PREMULTIPLIED_ALPHA = ...
    ADDITIVE = ...

class PolygonMode(enum.Enum):
    FILL = ...
    LINE = ...
    POINT = ...

class CullMode(enum.IntFlag):
    NONE = ...
    FRONT = ...
    BACK = ...
    FRONT_AND_BACK = ...

class FrontFace(enum.Enum):
    COUNTER_CLOCKWISE = ...
    CLOCKWISE = ...

class CompareOp(enum.Enum):
    NEVER = ...
    LESS = ...
    EQUAL = ...
    LESS_OR_EQUAL = ...
    GREATER = ...
    NOT_EQUAL = ...
    GREATER_OR_EQUAL = ...
    ALWAYS = ...

class PassType(enum.Enum):
    GEOMETRY = ...
    FULLSCREEN = ...
    COMPUTE = ...
    CUSTOM = ...

class Key(enum.Enum):
    """Keyboard keys (GLFW key codes)."""
    A = ...
    B = ...
    C = ...
    D = ...
    E = ...
    F = ...
    G = ...
    H = ...
    I = ...
    J = ...
    K = ...
    L = ...
    M = ...
    N = ...
    O = ...
    P = ...
    Q = ...
    R = ...
    S = ...
    T = ...
    U = ...
    V = ...
    W = ...
    X = ...
    Y = ...
    Z = ...
    DIGIT_0 = ...
    DIGIT_1 = ...
    DIGIT_2 = ...
    DIGIT_3 = ...
    DIGIT_4 = ...
    DIGIT_5 = ...
    DIGIT_6 = ...
    DIGIT_7 = ...
    DIGIT_8 = ...
    DIGIT_9 = ...
    F1 = ...
    F2 = ...
    F3 = ...
    F4 = ...
    F5 = ...
    F6 = ...
    F7 = ...
    F8 = ...
    F9 = ...
    F10 = ...
    F11 = ...
    F12 = ...
    SPACE = ...
    ESCAPE = ...
    ENTER = ...
    TAB = ...
    BACKSPACE = ...
    DELETE = ...
    LEFT = ...
    RIGHT = ...
    UP = ...
    DOWN = ...
    LEFT_SHIFT = ...
    RIGHT_SHIFT = ...
    LEFT_CONTROL = ...
    RIGHT_CONTROL = ...
    LEFT_ALT = ...
    RIGHT_ALT = ...

class MouseButton(enum.Enum):
    LEFT = ...
    RIGHT = ...
    MIDDLE = ...

class ShaderStage(enum.Enum):
    VERTEX = ...
    FRAGMENT = ...
    COMPUTE = ...

# ---------------------------------------------------------------------------
# Handles
# ---------------------------------------------------------------------------

class ImageHandle:
    def __bool__(self) -> bool: ...
    def __eq__(self, other: object) -> bool: ...

class BufferHandle:
    def __bool__(self) -> bool: ...
    def __eq__(self, other: object) -> bool: ...

class PassHandle:
    def __bool__(self) -> bool: ...
    def __eq__(self, other: object) -> bool: ...

# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------

class Extent:
    @staticmethod
    def swapchain() -> Extent:
        """Same size as the swapchain (the default)."""
        ...
    @staticmethod
    def absolute(width: int, height: int) -> Extent:
        """A fixed size in pixels."""
        ...
    @staticmethod
    def relative(numerator: int, denominator: int) -> Extent:
        """A fraction of the swapchain size, e.g. relative(1, 2) for half-res."""
        ...

class VertexBinding:
    binding: int
    stride: int
    def __init__(self, binding: int, stride: int, per_instance: bool = False) -> None: ...

class VertexAttribute:
    location: int
    binding: int
    format: Format
    offset: int
    def __init__(self, location: int, format: Format, offset: int = 0, binding: int = 0) -> None: ...

# ---------------------------------------------------------------------------
# Command buffer
# ---------------------------------------------------------------------------

class CommandBuffer:
    """Thin wrapper around a Vulkan command buffer. Only valid during a pass's execute callback."""

    def draw(
        self,
        vertex_count: int,
        instance_count: int = 1,
        first_vertex: int = 0,
        first_instance: int = 0,
    ) -> None:
        """Record a non-indexed draw call."""
        ...

    def draw_indexed(
        self,
        index_count: int,
        instance_count: int = 1,
        first_index: int = 0,
        vertex_offset: int = 0,
        first_instance: int = 0,
    ) -> None:
        """Record an indexed draw call."""
        ...

    def dispatch(self, x: int, y: int = 1, z: int = 1) -> None:
        """Dispatch a compute shader."""
        ...

    def set_viewport(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        min_depth: float = 0.0,
        max_depth: float = 1.0,
    ) -> None: ...

    def set_scissor(self, x: int, y: int, width: int, height: int) -> None: ...

# ---------------------------------------------------------------------------
# Pass context
# ---------------------------------------------------------------------------

class PassContext:
    """Handed to a pass's execute callback. Only valid during that call — don't keep it."""

    @property
    def cmd(self) -> CommandBuffer:
        """The command buffer to record draw/dispatch calls into."""
        ...

    @property
    def rendering_extent(self) -> tuple[int, int]:
        """Current render-target size as (width, height)."""
        ...

    def extent(self, image: ImageHandle) -> tuple[int, int]:
        """Size of a specific image as (width, height)."""
        ...

    def push_constants(
        self,
        stages: Stage,
        data: Buffer,
        offset: int = 0,
    ) -> None:
        """Push `data` (e.g. a float32 numpy array) into the pass's push-constant block."""
        ...

    def draw_indirect(
        self,
        buffer: BufferHandle,
        draw_count: int = 1,
        offset: int = 0,
        stride: int = 16,
    ) -> None:
        """vkCmdDrawIndirect from a buffer declared with indirect_buffer(); commands are 4 uint32s
        (vertex_count, instance_count, first_vertex, first_instance)."""
        ...

    def draw_indexed_indirect(
        self,
        buffer: BufferHandle,
        draw_count: int = 1,
        offset: int = 0,
        stride: int = 20,
    ) -> None:
        """vkCmdDrawIndexedIndirect; commands are 5 x 4 bytes (index_count, instance_count, first_index,
        vertex_offset, first_instance)."""
        ...

    def dispatch_indirect(self, buffer: BufferHandle, offset: int = 0) -> None:
        """vkCmdDispatchIndirect; the command is 3 uint32s (x, y, z)."""
        ...

# ---------------------------------------------------------------------------
# Pass builder
# ---------------------------------------------------------------------------

class PassBuilder:
    """Declares one pass. Every method returns the builder for chaining."""

    @property
    def handle(self) -> PassHandle: ...

    def type(self, type: PassType) -> PassBuilder: ...

    @overload
    def vert_shader(self, spirv: bytes) -> PassBuilder:
        """Set the vertex shader from raw SPIR-V bytes."""
        ...
    @overload
    def vert_shader(self, path: str | os.PathLike[str]) -> PassBuilder:
        """Set the vertex shader. Pass a path (.vert → GLSL compiled on-the-fly, .spv → SPIR-V)."""
        ...

    @overload
    def frag_shader(self, spirv: bytes) -> PassBuilder:
        """Set the fragment shader from raw SPIR-V bytes."""
        ...
    @overload
    def frag_shader(self, path: str | os.PathLike[str]) -> PassBuilder:
        """Set the fragment shader. Pass a path (.frag → GLSL compiled on-the-fly, .spv → SPIR-V)."""
        ...

    @overload
    def compute_shader(self, spirv: bytes) -> PassBuilder:
        """Set the compute shader from raw SPIR-V bytes."""
        ...
    @overload
    def compute_shader(self, path: str | os.PathLike[str]) -> PassBuilder:
        """Set the compute shader. Pass a path (.comp → GLSL compiled on-the-fly, .spv → SPIR-V)."""
        ...

    def push_constant_size(self, size: int, stages: Stage) -> PassBuilder: ...
    def topology(self, topology: Topology) -> PassBuilder: ...

    def vertex_layout(
        self,
        bindings: Sequence[VertexBinding],
        attributes: Sequence[VertexAttribute],
    ) -> PassBuilder: ...

    def blend(self, mode: BlendMode) -> PassBuilder:
        """Blend mode for all color attachments."""
        ...

    def polygon_mode(self, mode: PolygonMode) -> PassBuilder:
        """FILL (default), LINE (wireframe) or POINT."""
        ...

    def cull(
        self,
        mode: CullMode,
        front_face: FrontFace = FrontFace.COUNTER_CLOCKWISE,
    ) -> PassBuilder:
        """Default: BACK for geometry passes, NONE for fullscreen ones."""
        ...

    def depth(
        self,
        test: bool,
        write: bool,
        compare: CompareOp = CompareOp.LESS,
    ) -> PassBuilder:
        """Default: test and write on (compare LESS) exactly when the pass has a depth attachment."""
        ...

    def depth_bias(self, constant: float, slope: float = 0.0) -> PassBuilder:
        """Offset rasterized depth, e.g. negative values to draw a wireframe over its own solid surface."""
        ...

    def sampled_image(self, binding: int, image: ImageHandle, stages: Stage) -> PassBuilder: ...
    def sampled_depth(self, binding: int, image: ImageHandle, stages: Stage) -> PassBuilder: ...
    def sampled_image_array(self, binding: int, images: ImageHandle, count: int, stages: Stage) -> PassBuilder:
        """Bind the first `count` elements of an image array (see ResourceRegistry.upload_array_image) as
        `uniform sampler2D name[count]`."""
        ...
    def storage_image_read(self, binding: int, image: ImageHandle, stages: Stage) -> PassBuilder: ...
    def storage_image_write(self, binding: int, image: ImageHandle, stages: Stage) -> PassBuilder: ...
    def storage_image_read_write(self, binding: int, image: ImageHandle, stages: Stage) -> PassBuilder: ...
    def uniform_buffer(self, binding: int, buffer: BufferHandle, stages: Stage) -> PassBuilder: ...
    def storage_buffer_read(self, binding: int, buffer: BufferHandle, stages: Stage) -> PassBuilder: ...
    def storage_buffer_write(self, binding: int, buffer: BufferHandle, stages: Stage) -> PassBuilder: ...
    def storage_buffer_read_write(self, binding: int, buffer: BufferHandle, stages: Stage) -> PassBuilder: ...
    def vertex_buffer(self, binding: int, buffer: BufferHandle) -> PassBuilder: ...
    def index_buffer(self, buffer: BufferHandle) -> PassBuilder: ...

    def indirect_buffer(self, buffer: BufferHandle) -> PassBuilder:
        """Declare a buffer of draw/dispatch arguments for PassContext.draw_indirect() and friends."""
        ...

    def runs_last(self) -> PassBuilder:
        """Order this pass after every other pass sharing a resource with it, even ones declared later."""
        ...

    def color_attachment(
        self,
        image: ImageHandle,
        format: Format,
        load_op: LoadOp = LoadOp.CLEAR,
        clear_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        extent: Extent = ...,
    ) -> PassBuilder: ...

    def depth_attachment(
        self,
        image: ImageHandle,
        format: Format = Format.D32_SFLOAT,
        load_op: LoadOp = LoadOp.CLEAR,
        clear_depth: float = 1.0,
        extent: Extent = ...,
    ) -> PassBuilder: ...

    @overload
    def depends_on(self, dependency: PassHandle) -> PassBuilder: ...
    @overload
    def depends_on(self, dependencies: Sequence[PassHandle]) -> PassBuilder: ...

    def execute(self, callback: Callable[[PassContext], None]) -> PassBuilder:
        """Record this pass's commands: callback(ctx: PassContext), called every frame."""
        ...

# ---------------------------------------------------------------------------
# Frame graph
# ---------------------------------------------------------------------------

class FrameGraph:
    def add_pass(self, name: str) -> PassBuilder:
        """Start declaring a pass; passes run in dependency order, derived from the resources they use."""
        ...

    def image(self, name: str) -> ImageHandle:
        """Handle for a named image. Use \"swapchain\" for the window's back buffer."""
        ...

    def buffer(self, name: str) -> BufferHandle:
        """Handle for a named buffer."""
        ...

    def compile(self) -> None:
        """Rebuild pipelines and barriers after changing passes while running."""
        ...

    @property
    def needs_recompile(self) -> bool: ...

    @property
    def compile_count(self) -> int:
        """How many times the graph has been compiled."""
        ...

    def debug_dump(self) -> None: ...

# ---------------------------------------------------------------------------
# Resource registry
# ---------------------------------------------------------------------------

class ResourceRegistry:
    """Named GPU buffers and images. Passes refer to them by name via FrameGraph.buffer()/image()."""

    @property
    def extent(self) -> tuple[int, int]:
        """Current swapchain-sized extent as (width, height)."""
        ...

    def upload_buffer(self, name: str, data: Buffer, usage: BufferUsage) -> None:
        """Create a GPU-only buffer sized to `data` and upload it before the next frame."""
        ...

    def reupload_buffer(self, name: str, data: Buffer) -> None:
        """Overwrite an uploaded buffer; `data` must fit its original size."""
        ...

    def register_static_buffer(self, name: str, size: int, usage: BufferUsage) -> None:
        """Create an uninitialised GPU-only buffer (e.g. compute scratch)."""
        ...

    def register_dynamic_buffer(self, name: str, size: int, usage: BufferUsage) -> None:
        """Create a CPU-writable buffer for per-frame data such as uniforms; fill it with update_buffer()."""
        ...

    def update_buffer(self, name: str, data: Buffer) -> None:
        """Write `data` into a dynamic buffer (call from an on_update callback)."""
        ...

    def register_image(
        self,
        name: str,
        format: Format,
        usage: ImageUsage,
        extent: Extent = ...,
    ) -> None:
        """Create a transient image, reallocated when the window resizes."""
        ...

    def upload_image(
        self,
        name: str,
        data: Buffer,
        format: Format,
        generate_mipmaps: bool = False,
    ) -> None:
        """Create a sampled image from an array shaped (height, width[, channels])."""
        ...

    def upload_array_image(
        self,
        array_name: str,
        index: int,
        data: Buffer,
        format: Format,
        generate_mipmaps: bool = False,
    ) -> None:
        """Upload one element of an image array (elements may differ in size), for sampled_image_array():
        how a single pass picks a different texture per draw, as the engine's GeometryPass does."""
        ...

    def replace_image(
        self,
        name: str,
        data: Buffer,
        format: Format,
        generate_mipmaps: bool = False,
    ) -> None:
        """Replace an image created by upload_image() (any size/format). Safe while running: the old image
        is kept until in-flight frames finish, and passes sampling it recompile on the next frame."""
        ...

    def replace_buffer(self, name: str, data: Buffer, usage: BufferUsage) -> None:
        """Replace a buffer created by upload_buffer() with new contents of any size. Safe while running."""
        ...

    def replace_dynamic_buffer(self, name: str, size: int, usage: BufferUsage) -> None:
        """Reallocate a buffer created by register_dynamic_buffer() at a new size (contents start undefined).
        Same safety as replace_buffer()."""
        ...

    def read_buffer(self, name: str) -> NDArray[numpy.uint8]:
        """Copy a buffer back to the CPU as a uint8 numpy array (use .view(np.float32) etc.), as of the last
        submitted frame. Waits for the GPU to go idle: meant for tests and debugging, not every frame."""
        ...

    def has_buffer(self, name: str) -> bool: ...
    def has_image(self, name: str) -> bool: ...

# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------

class Input:
    """Keyboard and mouse state, updated once per frame before on_update. It reports the raw state even
    over UI; check lr.gui.want_capture_mouse() to leave the mouse to ImGui."""

    def is_key_down(self, key: Key) -> bool: ...
    def is_mouse_down(self, button: MouseButton) -> bool: ...
    @property
    def mouse_position(self) -> tuple[float, float]:
        """Cursor position in window pixels, (x, y)."""
        ...
    @property
    def mouse_delta(self) -> tuple[float, float]:
        """Cursor movement since the previous frame, (dx, dy) in pixels."""
        ...
    @property
    def scroll_delta(self) -> float:
        """Scroll-wheel movement since the previous frame."""
        ...
    @property
    def shift(self) -> bool: ...
    @property
    def ctrl(self) -> bool: ...
    @property
    def alt(self) -> bool: ...

class Viewer:
    """Window + Vulkan device + frame graph. Declare passes, then call run(), or call step() in your own
    loop."""

    def __init__(
        self,
        title: str = "lr",
        width: int = 1600,
        height: int = 900,
        validation: bool = True,
        raise_validation_errors: bool = True,
    ) -> None:
        """With validation on (the default), a validation-layer error closes the window and run() raises
        VulkanValidationError; pass raise_validation_errors=False to only log them."""
        ...

    @property
    def frame_graph(self) -> FrameGraph: ...

    @property
    def resources(self) -> ResourceRegistry: ...

    @property
    def swapchain_format(self) -> Format: ...

    def on_update(self, callback: Callable[[float, tuple[int, int]], None]) -> None:
        """callback(dt: float, extent: (width, height)), called every frame before rendering."""
        ...

    def on_late_update(self, callback: Callable[[float, tuple[int, int]], None]) -> None:
        """Like on_update, but after every on_update callback has run."""
        ...

    def on_gui(self, callback: Callable[[], None]) -> None:
        """callback(), called every frame to build ImGui panels with lr.gui (e.g. `with lr.gui.window(...)`)."""
        ...

    @property
    def input(self) -> Input:
        """Keyboard and mouse state for this window."""
        ...

    def run(self) -> None:
        """Compile the frame graph and run frames until the window closes. An exception raised in any
        callback closes the window and is re-raised here. Once the window has closed, its callbacks are
        released: a Viewer runs once."""
        ...

    def step(self) -> bool:
        """Render one frame (compiling the frame graph on the first call) and return True, or False once
        the window has closed: `while viewer.step(): ...` lets your code own the loop. The window only
        responds while it is being stepped. Callbacks run as with run(); an exception raised in one
        closes the window and is re-raised from this step. Once the window has closed, its callbacks
        are released."""
        ...

    @property
    def is_open(self) -> bool:
        """True until the window has closed."""
        ...

    def close(self) -> None:
        """Close the window after the current frame (run() returns, step() returns False)."""
        ...

class OrbitCamera:
    """The engine's orbit camera (SphericalCameraController + Camera), with the C++ renderer's controls:
    middle-drag orbits, Shift + middle-drag pans, the scroll wheel zooms and R resets. Ignores the mouse
    while it is over ImGui UI. Call update() once per frame, then use matrices(extent) for a
    `mat4 view; mat4 proj;` uniform block."""

    def __init__(self, viewer: Viewer) -> None: ...
    def update(self, dt: float) -> None:
        """Apply this frame's input (from on_update)."""
        ...
    def view_matrix(self) -> NDArray[numpy.float32]:
        """4x4 float32 view matrix (row-major numpy, acting on column vectors)."""
        ...
    def projection_matrix(self, aspect: float) -> NDArray[numpy.float32]:
        """4x4 float32 projection: Vulkan clip space, depth in [0, 1], Y flipped — as the engine uses."""
        ...
    def matrices(self, extent: tuple[int, int]) -> NDArray[numpy.float32]:
        """view and projection for a (width, height) extent, packed for a `mat4 view; mat4 proj;` block."""
        ...
    @property
    def position(self) -> tuple[float, float, float]:
        """Camera position in world space."""
        ...
    @property
    def target(self) -> tuple[float, float, float]:
        """The point orbited around."""
        ...
    @target.setter
    def target(self, value: Sequence[float]) -> None: ...
    @property
    def radius(self) -> float:
        """Distance from the target (clamped to [0.01, 1000])."""
        ...
    @radius.setter
    def radius(self, value: float) -> None: ...
    @property
    def azimuth(self) -> float:
        """Radians around +Y; 0 places the camera on the target's +Z side."""
        ...
    @azimuth.setter
    def azimuth(self, value: float) -> None: ...
    @property
    def elevation(self) -> float:
        """Radians above the horizontal (clamped to ±89°)."""
        ...
    @elevation.setter
    def elevation(self, value: float) -> None: ...
    @property
    def fov_y_degrees(self) -> float: ...
    @fov_y_degrees.setter
    def fov_y_degrees(self, value: float) -> None: ...
    @property
    def near_plane(self) -> float: ...
    @near_plane.setter
    def near_plane(self, value: float) -> None: ...
    @property
    def far_plane(self) -> float: ...
    @far_plane.setter
    def far_plane(self, value: float) -> None: ...
    @property
    def orthographic(self) -> bool:
        """Orthographic instead of perspective projection (height set by ortho_height)."""
        ...
    @orthographic.setter
    def orthographic(self, value: bool) -> None: ...
    @property
    def ortho_height(self) -> float: ...
    @ortho_height.setter
    def ortho_height(self, value: float) -> None: ...

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
    def area_size(self) -> tuple[float, float] | None:
        """Area lights only: (width, height) in world units."""
        ...
    @property
    def two_sided(self) -> bool | None:
        """Area lights only: emits from both faces (True) or only along its forward axis."""
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
    def id(self) -> int: ...
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
        two_sided: bool | None = None,
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
        two_sided: bool = True,
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
    """Load an OBJ, glTF or GLB file into a new Scene."""
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

# ---------------------------------------------------------------------------
# Shader utilities
# ---------------------------------------------------------------------------

class ShaderCompileError(Exception): ...
class ShaderInterfaceError(RuntimeError): ...
class VulkanValidationError(RuntimeError): ...

def compile_glsl(
    path: os.PathLike,
    stage: ShaderStage | None = None,
    include_dirs: Sequence[os.PathLike] = (),
) -> bytes:
    """Compile a GLSL file to SPIR-V bytes. Stage is inferred from .vert/.frag/.comp unless given."""
    ...

def compile_glsl_source(
    source: str,
    stage: ShaderStage,
    name: str = "<source>",
    include_dirs: Sequence[os.PathLike] = (),
) -> bytes:
    """Compile GLSL source text to SPIR-V bytes."""
    ...

SHADER_DIR: pathlib.Path
ASSET_DIR: pathlib.Path
