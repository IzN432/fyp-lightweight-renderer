"""Contract: load a glTF model and display its triangle edges as wireframe."""

from pathlib import Path

import numpy as np

import lightweight_renderer as lr


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]
MODEL_PATH = REPOSITORY_ROOT / "assets" / "samples" / "models" / "lion_head_4k.glb"


def unique_triangle_edges(indices: np.ndarray) -> np.ndarray:
    """Return a flat uint32 line-list index array for triangle-list indices."""
    triangles = np.asarray(indices, dtype=np.uint32).reshape(-1, 3)
    edges = np.concatenate(
        (
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ),
        axis=0,
    )

    # Treat (a, b) and (b, a) as the same undirected edge.
    edges.sort(axis=1)
    return np.unique(edges, axis=0).ravel()


asset = lr.load_gltf(MODEL_PATH)
triangle_meshes = [
    mesh
    for mesh in asset.meshes
    if mesh.topology is lr.PrimitiveTopology.TRIANGLE_LIST
]
if not triangle_meshes:
    raise ValueError(f"{MODEL_PATH} contains no triangle-list mesh primitives")

# Normalize all primitives together so the complete asset fits in clip space.
all_positions = np.concatenate(
    [mesh.attributes["position"] for mesh in triangle_meshes], axis=0
).astype(np.float32, copy=False)
bounds_min = all_positions.min(axis=0)
bounds_max = all_positions.max(axis=0)
center = (bounds_min + bounds_max) * 0.5
scale = 1.8 / max(float((bounds_max - bounds_min).max()), 1e-8)

renderer = lr.Renderer()
window = renderer.create_window(title="glTF wireframe", width=1280, height=720)

draws = []
for mesh in triangle_meshes:
    positions = np.ascontiguousarray(
        (mesh.attributes["position"].astype(np.float32) - center) * scale
    )
    triangle_indices = (
        mesh.indices
        if mesh.indices is not None
        else np.arange(positions.shape[0], dtype=np.uint32)
    )
    edge_indices = unique_triangle_edges(triangle_indices)

    draws.append(
        (
            renderer.create_buffer(
                positions,
                usage=lr.BufferUsage.VERTEX,
                label=f"{mesh.name} positions",
            ),
            renderer.create_buffer(
                edge_indices,
                usage=lr.BufferUsage.INDEX,
                label=f"{mesh.name} edges",
            ),
            edge_indices.size,
        )
    )

graph = lr.FrameGraphBuilder(renderer)
backbuffer = graph.import_backbuffer(window)
depth = graph.create_image(
    "wireframe depth",
    format=lr.Format.D32_FLOAT,
    extent=lr.Extent.match(backbuffer),
    usage=lr.ImageUsage.DEPTH_ATTACHMENT,
)

pipeline = renderer.create_graphics_pipeline(
    vertex_shader=renderer.load_shader(
        HERE / "shaders" / "wireframe.vert", lr.ShaderStage.VERTEX
    ),
    fragment_shader=renderer.load_shader(
        HERE / "shaders" / "wireframe.frag", lr.ShaderStage.FRAGMENT
    ),
    vertex_layout=lr.VertexLayout(
        bindings=[lr.VertexBinding(binding=0, stride=3 * np.dtype(np.float32).itemsize)],
        attributes=[
            lr.VertexAttribute(
                location=0,
                binding=0,
                format=lr.Format.RGB32_FLOAT,
                offset=0,
            )
        ],
    ),
    topology=lr.PrimitiveTopology.LINE_LIST,
    color_formats=[backbuffer.format],
    depth_format=depth.format,
    depth_test=True,
    depth_write=True,
)


def draw_wireframe(encoder: lr.CommandEncoder) -> None:
    encoder.bind_pipeline(pipeline)
    for position_buffer, edge_buffer, index_count in draws:
        encoder.bind_vertex_buffer(0, position_buffer)
        encoder.bind_index_buffer(edge_buffer)
        encoder.draw_indexed(index_count=index_count)


graph.add_render_pass(
    "wireframe",
    color_attachments=[
        lr.ColorAttachment(
            backbuffer,
            load=lr.LoadOp.CLEAR,
            clear=(0.015, 0.018, 0.025, 1.0),
        )
    ],
    depth_attachment=lr.DepthAttachment(
        depth,
        load=lr.LoadOp.CLEAR,
        clear=1.0,
    ),
    reads=[
        access
        for position_buffer, edge_buffer, _ in draws
        for access in (
            lr.vertex_read(position_buffer),
            lr.index_read(edge_buffer),
        )
    ],
    execute=draw_wireframe,
)

compiled = graph.compile(outputs=[backbuffer])
renderer.run(window, compiled)
