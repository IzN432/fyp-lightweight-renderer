"""Per-pass pipeline state — three passes drawing into the same window with different fixed-function state.

  pass "solid": opaque, lit torus (default state: back-face culling, depth test + write)
  pass "wire":  the same mesh again as a wireframe on top of it — polygon_mode(LINE), depth test with
                LESS_OR_EQUAL and no depth writes, plus a depth bias so lines win over their own surface
  pass "glass": three overlapping alpha-blended quads, drawn back to front — blend(ALPHA), no culling,
                depth-tested against the torus (so they cut into it) but not writing depth

"wire" and "glass" load the swapchain and depth images that "solid" wrote, which also orders them after it.

Run from the repo root after building:
    PYTHONPATH=build/python python examples/python/pipeline_state.py [--frames N]
"""

import argparse
import pathlib
from typing import Any

import numpy as np

import lr
from lr import transforms as tf
from meshes import make_torus

SHADERS = pathlib.Path(__file__).parent / "shaders"


def make_glass_quads():
    """Three overlapping RGBA quads facing the camera, ordered back to front: (18, 7) float32 rows of pos + rgba."""
    quads = [
        # centre, half-size, colour (premultiplied by nothing — alpha blending uses straight alpha)
        ((-0.55, 0.25, 0.7), 0.6, (0.95, 0.2, 0.2, 0.45)),
        ((0.0, -0.15, 0.95), 0.6, (0.2, 0.9, 0.3, 0.45)),
        ((0.55, 0.25, 1.2), 0.6, (0.25, 0.4, 1.0, 0.45)),
    ]
    rows = []
    for (cx, cy, cz), h, rgba in quads:
        corners = [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]
        for k in (0, 1, 2, 0, 2, 3):
            x, y = corners[k]
            rows.append((x, y, cz, *rgba))
    return np.array(rows, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int, default=0, help="close after this many frames (0 = run until closed)")
    args = parser.parse_args()

    viewer = lr.Viewer(title="lr - pipeline state (Python)", width=1280, height=720)
    fg, res = viewer.frame_graph, viewer.resources
    swapchain, depth = fg.image(lr.SWAPCHAIN), fg.image("depth")

    vertices, indices = make_torus()
    glass = make_glass_quads()
    res.upload_buffer("torus_vertices", vertices, lr.BufferUsage.VERTEX)
    res.upload_buffer("torus_indices", indices, lr.BufferUsage.INDEX)
    res.upload_buffer("glass_vertices", glass, lr.BufferUsage.VERTEX)
    res.register_dynamic_buffer("camera", 2 * 64, lr.BufferUsage.UNIFORM)

    state: dict[str, Any] = {"time": 0.0, "model": np.identity(4, dtype=np.float32), "frames": 0}
    torus_layout = (
        [lr.VertexBinding(0, stride=vertices.strides[0])],
        [
            lr.VertexAttribute(0, lr.Format.R32G32B32_SFLOAT, offset=0),
            lr.VertexAttribute(1, lr.Format.R32G32B32_SFLOAT, offset=12),
        ],
    )

    def draw_torus(ctx):
        ctx.push_constants(lr.Stage.VERTEX, tf.to_gpu(state["model"]))
        ctx.cmd.draw_indexed(len(indices))

    def torus_pass(name, frag):
        return (
            fg.add_pass(name)
            .type(lr.PassType.GEOMETRY)
            .vert_shader(SHADERS / "torus.vert")
            .frag_shader(SHADERS / frag)
            .vertex_layout(*torus_layout)
            .vertex_buffer(0, fg.buffer("torus_vertices"))
            .index_buffer(fg.buffer("torus_indices"))
            .uniform_buffer(0, fg.buffer("camera"), lr.Stage.VERTEX)
            .push_constant_size(64, lr.Stage.VERTEX)
            .execute(draw_torus)
        )

    (
        torus_pass("solid", "solid.frag")
        .color_attachment(swapchain, viewer.swapchain_format, clear_color=(0.09, 0.09, 0.11, 1.0))
        .depth_attachment(depth)
    )

    (
        torus_pass("wire", "wire.frag")
        .polygon_mode(lr.PolygonMode.LINE)
        .depth(test=True, write=False, compare=lr.CompareOp.LESS_OR_EQUAL)
        .depth_bias(constant=-1.0, slope=-1.0)
        .color_attachment(swapchain, viewer.swapchain_format, load_op=lr.LoadOp.LOAD)
        .depth_attachment(depth, load_op=lr.LoadOp.LOAD)
    )

    (
        fg.add_pass("glass")
        .type(lr.PassType.GEOMETRY)
        .vert_shader(SHADERS / "glass.vert")
        .frag_shader(SHADERS / "glass.frag")
        .vertex_layout(
            [lr.VertexBinding(0, stride=glass.strides[0])],
            [
                lr.VertexAttribute(0, lr.Format.R32G32B32_SFLOAT, offset=0),
                lr.VertexAttribute(1, lr.Format.R32G32B32A32_SFLOAT, offset=12),
            ],
        )
        .vertex_buffer(0, fg.buffer("glass_vertices"))
        .uniform_buffer(0, fg.buffer("camera"), lr.Stage.VERTEX)
        .blend(lr.BlendMode.ALPHA)
        .cull(lr.CullMode.NONE)
        .depth(test=True, write=False)
        .color_attachment(swapchain, viewer.swapchain_format, load_op=lr.LoadOp.LOAD)
        .depth_attachment(depth, load_op=lr.LoadOp.LOAD)
        .execute(lambda ctx: ctx.cmd.draw(len(glass)))
    )

    def update(dt, extent):
        width, height = extent
        state["time"] += dt
        state["model"] = tf.rotation((0, 1, 0), state["time"] * 0.4) @ tf.rotation((1, 0, 0), 0.5)

        view = tf.look_at(eye=(0.0, 1.0, 3.6), target=(0.0, 0.0, 0.0))
        proj = tf.perspective(45.0, width / max(height, 1), 0.1, 100.0)
        res.update_buffer("camera", tf.to_gpu(view, proj))

        state["frames"] += 1
        if args.frames and state["frames"] >= args.frames:
            viewer.close()

    viewer.on_update(update)
    viewer.run()


if __name__ == "__main__":
    main()
