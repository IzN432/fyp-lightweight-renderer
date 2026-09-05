"""Contract: explicit geometry pipeline rendering to an offscreen image."""

from pathlib import Path

import numpy as np

import lightweight_renderer as lr


HERE = Path(__file__).resolve().parent
renderer = lr.Renderer()

# A structured dtype defines interleaved vertex memory without a Mesh class.
vertex_dtype = np.dtype(
    [
        ("position", np.float32, 2),
        ("color", np.float32, 3),
    ],
    align=False,
)
vertices = np.array(
    [
        ((-0.75, -0.60), (1.0, 0.1, 0.1)),
        ((0.75, -0.60), (0.1, 1.0, 0.1)),
        ((0.00, 0.75), (0.1, 0.3, 1.0)),
    ],
    dtype=vertex_dtype,
)
indices = np.array([0, 1, 2], dtype=np.uint32)

vertex_buffer = renderer.create_buffer(vertices, usage=lr.BufferUsage.VERTEX)
index_buffer = renderer.create_buffer(indices, usage=lr.BufferUsage.INDEX)

pipeline = renderer.create_graphics_pipeline(
    vertex_shader=renderer.load_shader(
        HERE / "shaders" / "triangle.vert", lr.ShaderStage.VERTEX
    ),
    fragment_shader=renderer.load_shader(
        HERE / "shaders" / "triangle.frag", lr.ShaderStage.FRAGMENT
    ),
    vertex_layout=lr.VertexLayout.from_numpy_dtype(
        vertex_dtype,
        locations={"position": 0, "color": 1},
    ),
    color_formats=[lr.Format.RGBA8_UNORM],
)

graph = lr.FrameGraphBuilder(renderer)
color = graph.create_image(
    "color",
    format=lr.Format.RGBA8_UNORM,
    extent=lr.Extent.absolute(800, 600),
    usage=lr.ImageUsage.COLOR_ATTACHMENT | lr.ImageUsage.READBACK,
)


def draw(encoder: lr.CommandEncoder) -> None:
    encoder.bind_pipeline(pipeline)
    encoder.bind_vertex_buffer(0, vertex_buffer)
    encoder.bind_index_buffer(index_buffer)
    encoder.draw_indexed(index_count=indices.size)


graph.add_render_pass(
    "triangle",
    color_attachments=[
        lr.ColorAttachment(color, load=lr.LoadOp.CLEAR, clear=(0.02, 0.02, 0.03, 1.0))
    ],
    reads=[lr.vertex_read(vertex_buffer), lr.index_read(index_buffer)],
    execute=draw,
)

compiled = graph.compile(outputs=[color])
renderer.execute_and_wait(compiled)
pixels = color.readback(dtype=np.uint8, shape=(600, 800, 4))
np.save("triangle.npy", pixels)
