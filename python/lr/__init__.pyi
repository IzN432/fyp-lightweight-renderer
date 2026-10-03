"""Type stubs for the lr native extension (_lr)."""

from __future__ import annotations

import os
import pathlib
from typing import Callable, Sequence

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Format:
    UNDEFINED: Format
    R8_UNORM: Format
    R8G8B8A8_UNORM: Format
    R8G8B8A8_SRGB: Format
    B8G8R8A8_UNORM: Format
    B8G8R8A8_SRGB: Format
    R16_SFLOAT: Format
    R16G16B16A16_SFLOAT: Format
    R32_SFLOAT: Format
    R32G32_SFLOAT: Format
    R32G32B32_SFLOAT: Format
    R32G32B32A32_SFLOAT: Format
    R32_UINT: Format
    D32_SFLOAT: Format
    name: str

class BufferUsage:
    TRANSFER_SRC: BufferUsage
    TRANSFER_DST: BufferUsage
    UNIFORM: BufferUsage
    STORAGE: BufferUsage
    INDEX: BufferUsage
    VERTEX: BufferUsage
    INDIRECT: BufferUsage
    def __or__(self, other: BufferUsage) -> BufferUsage: ...
    def __and__(self, other: BufferUsage) -> BufferUsage: ...

class ImageUsage:
    TRANSFER_SRC: ImageUsage
    TRANSFER_DST: ImageUsage
    SAMPLED: ImageUsage
    STORAGE: ImageUsage
    COLOR_ATTACHMENT: ImageUsage
    DEPTH_STENCIL_ATTACHMENT: ImageUsage
    def __or__(self, other: ImageUsage) -> ImageUsage: ...
    def __and__(self, other: ImageUsage) -> ImageUsage: ...

class Stage:
    VERTEX: Stage
    FRAGMENT: Stage
    COMPUTE: Stage
    ALL_GRAPHICS: Stage
    def __or__(self, other: Stage) -> Stage: ...
    def __and__(self, other: Stage) -> Stage: ...

class Topology:
    POINT_LIST: Topology
    LINE_LIST: Topology
    LINE_STRIP: Topology
    TRIANGLE_LIST: Topology
    TRIANGLE_STRIP: Topology

class LoadOp:
    LOAD: LoadOp
    CLEAR: LoadOp
    DONT_CARE: LoadOp

class BlendMode:
    OPAQUE: BlendMode
    ALPHA: BlendMode
    PREMULTIPLIED_ALPHA: BlendMode
    ADDITIVE: BlendMode

class PolygonMode:
    FILL: PolygonMode
    LINE: PolygonMode
    POINT: PolygonMode

class CullMode:
    NONE: CullMode
    FRONT: CullMode
    BACK: CullMode
    FRONT_AND_BACK: CullMode
    def __or__(self, other: CullMode) -> CullMode: ...

class FrontFace:
    COUNTER_CLOCKWISE: FrontFace
    CLOCKWISE: FrontFace

class CompareOp:
    NEVER: CompareOp
    LESS: CompareOp
    EQUAL: CompareOp
    LESS_OR_EQUAL: CompareOp
    GREATER: CompareOp
    NOT_EQUAL: CompareOp
    GREATER_OR_EQUAL: CompareOp
    ALWAYS: CompareOp

class PassType:
    GEOMETRY: PassType
    FULLSCREEN: PassType
    COMPUTE: PassType
    CUSTOM: PassType

class ShaderStage:
    VERTEX: ShaderStage
    FRAGMENT: ShaderStage
    COMPUTE: ShaderStage

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
        data: object,  # any numpy array / bytes-like
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

    def vert_shader(self, path_or_spirv: os.PathLike | bytes) -> PassBuilder:
        """Set the vertex shader. Pass a path (.vert → GLSL compiled on-the-fly, .spv → SPIR-V) or raw SPIR-V bytes."""
        ...

    def frag_shader(self, path_or_spirv: os.PathLike | bytes) -> PassBuilder:
        """Set the fragment shader. Pass a path (.frag → GLSL compiled on-the-fly, .spv → SPIR-V) or raw SPIR-V bytes."""
        ...

    def compute_shader(self, path_or_spirv: os.PathLike | bytes) -> PassBuilder:
        """Set the compute shader."""
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

    def depends_on(self, dependency: PassHandle | Sequence[PassHandle]) -> PassBuilder: ...

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

    def upload_buffer(self, name: str, data: object, usage: BufferUsage) -> None:
        """Create a GPU-only buffer sized to `data` and upload it before the next frame."""
        ...

    def reupload_buffer(self, name: str, data: object) -> None:
        """Overwrite an uploaded buffer; `data` must fit its original size."""
        ...

    def register_static_buffer(self, name: str, size: int, usage: BufferUsage) -> None:
        """Create an uninitialised GPU-only buffer (e.g. compute scratch)."""
        ...

    def register_dynamic_buffer(self, name: str, size: int, usage: BufferUsage) -> None:
        """Create a CPU-writable buffer for per-frame data such as uniforms; fill it with update_buffer()."""
        ...

    def update_buffer(self, name: str, data: object) -> None:
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
        data: object,
        format: Format,
        generate_mipmaps: bool = False,
    ) -> None:
        """Create a sampled image from an array shaped (height, width[, channels])."""
        ...

    def replace_image(
        self,
        name: str,
        data: object,
        format: Format,
        generate_mipmaps: bool = False,
    ) -> None:
        """Replace an image created by upload_image() (any size/format). Safe while running: the old image
        is kept until in-flight frames finish, and passes sampling it recompile on the next frame."""
        ...

    def replace_buffer(self, name: str, data: object, usage: BufferUsage) -> None:
        """Replace a buffer created by upload_buffer() with new contents of any size. Safe while running."""
        ...

    def replace_dynamic_buffer(self, name: str, size: int, usage: BufferUsage) -> None:
        """Reallocate a buffer created by register_dynamic_buffer() at a new size (contents start undefined).
        Same safety as replace_buffer()."""
        ...

    def read_buffer(self, name: str) -> object:
        """Copy a buffer back to the CPU as a uint8 numpy array (use .view(np.float32) etc.), as of the last
        submitted frame. Waits for the GPU to go idle: meant for tests and debugging, not every frame."""
        ...

    def has_buffer(self, name: str) -> bool: ...
    def has_image(self, name: str) -> bool: ...

# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------

class Viewer:
    """Window + Vulkan device + frame graph. Declare passes, then call run()."""

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

    def run(self) -> None:
        """Compile the frame graph and run until the window closes."""
        ...

    def close(self) -> None:
        """Ask run() to return after the current frame."""
        ...

# ---------------------------------------------------------------------------
# Shader utilities
# ---------------------------------------------------------------------------

class ShaderCompileError(Exception): ...
class ShaderInterfaceError(Exception): ...
class VulkanValidationError(Exception): ...

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
