"""Spinning torus — a two-pass renderer written entirely in Python on the lr frame graph.

  pass "torus": draws a mesh generated with numpy into an HDR colour target + depth buffer
  pass "post":  samples that target, tone-maps it and writes the window's swapchain image

Neither attachment is created by hand: the frame graph allocates "torus_color"/"torus_depth" from
the pass declarations (and resizes them with the window), orders "post" after "torus" because it
samples what "torus" wrote, and inserts the barrier between them.

Run from the repo root after building:
    PYTHONPATH=build/python python examples/python/spinning_torus.py [--frames N]
"""

import argparse
import pathlib

import numpy as np

import lr
from lr import transforms as tf

SHADERS = pathlib.Path(__file__).parent / "shaders"


def make_torus(major_radius=1.0, minor_radius=0.4, rings=96, sides=48):
    """Interleaved (position, normal) float32 vertices and a flat uint32 triangle index array."""
    u = np.linspace(0.0, 2.0 * np.pi, rings, endpoint=False)
    v = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing="ij")

    normals = np.stack([np.cos(vv) * np.cos(uu), np.sin(vv), np.cos(vv) * np.sin(uu)], axis=-1)
    centres = np.stack([major_radius * np.cos(uu), np.zeros_like(uu), major_radius * np.sin(uu)], axis=-1)
    positions = centres + minor_radius * normals
    vertices = np.concatenate([positions, normals], axis=-1).reshape(-1, 6).astype(np.float32)

    i, j = np.meshgrid(np.arange(rings), np.arange(sides), indexing="ij")
    a = i * sides + j
    b = ((i + 1) % rings) * sides + j
    c = ((i + 1) % rings) * sides + (j + 1) % sides
    d = i * sides + (j + 1) % sides
    # Counter-clockwise seen from outside, the engine's front face.
    triangles = np.stack([a, d, c, a, c, b], axis=-1).reshape(-1)
    return vertices, triangles.astype(np.uint32)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int, default=0, help="close after this many frames (0 = run until closed)")
    args = parser.parse_args()

    viewer = lr.Viewer(title="lr - spinning torus (Python)", width=1280, height=720)
    fg, res = viewer.frame_graph, viewer.resources

    vertices, indices = make_torus()
    res.upload_buffer("torus_vertices", vertices, lr.BufferUsage.VERTEX)
    res.upload_buffer("torus_indices", indices, lr.BufferUsage.INDEX)
    res.register_dynamic_buffer("camera", 2 * 64, lr.BufferUsage.UNIFORM)  # view + proj

    state = {"time": 0.0, "model": np.identity(4, dtype=np.float32), "frames": 0}

    def draw_torus(ctx):
        ctx.push_constants(lr.Stage.VERTEX, tf.to_gpu(state["model"]))
        ctx.cmd.draw_indexed(len(indices))

    (
        fg.add_pass("torus")
        .type(lr.PassType.GEOMETRY)
        .vert_shader(SHADERS / "torus.vert")
        .frag_shader(SHADERS / "torus.frag")
        .vertex_layout(
            [lr.VertexBinding(0, stride=vertices.strides[0])],
            [
                lr.VertexAttribute(0, lr.Format.R32G32B32_SFLOAT, offset=0),  # position
                lr.VertexAttribute(1, lr.Format.R32G32B32_SFLOAT, offset=12),  # normal
            ],
        )
        .vertex_buffer(0, fg.buffer("torus_vertices"))
        .index_buffer(fg.buffer("torus_indices"))
        .uniform_buffer(0, fg.buffer("camera"), lr.Stage.VERTEX)
        .push_constant_size(64, lr.Stage.VERTEX)
        .color_attachment(fg.image("torus_color"), lr.Format.R16G16B16A16_SFLOAT, clear_color=(0.02, 0.02, 0.03, 1.0))
        .depth_attachment(fg.image("torus_depth"))
        .execute(draw_torus)
    )

    apply_gamma = 0.0 if viewer.swapchain_format.name.endswith("SRGB") else 1.0
    post_params = np.array([1.2, 0.6, apply_gamma], dtype=np.float32)  # exposure, vignette, applyGamma

    def draw_post(ctx):
        ctx.push_constants(lr.Stage.FRAGMENT, post_params)
        ctx.cmd.draw(3)  # fullscreen triangle

    (
        fg.add_pass("post")
        .type(lr.PassType.FULLSCREEN)
        .vert_shader(lr.SHADER_DIR / "fullscreen.vert.spv")  # the engine's own, prebuilt
        .frag_shader(SHADERS / "post.frag")
        .sampled_image(0, fg.image("torus_color"), lr.Stage.FRAGMENT)
        .push_constant_size(post_params.nbytes, lr.Stage.FRAGMENT)
        .color_attachment(fg.image("swapchain"), viewer.swapchain_format)
        .execute(draw_post)
    )

    def update(dt, extent):
        width, height = extent
        state["time"] += dt
        state["model"] = tf.rotation((0, 1, 0), state["time"] * 0.7) @ tf.rotation((1, 0, 0), 0.6)

        view = tf.look_at(eye=(0.0, 1.2, 3.4), target=(0.0, 0.0, 0.0))
        proj = tf.perspective(45.0, width / max(height, 1), 0.1, 100.0)
        res.update_buffer("camera", tf.to_gpu(view, proj))

        state["frames"] += 1
        if args.frames and state["frames"] >= args.frames:
            viewer.close()

    viewer.on_update(update)
    viewer.run()


if __name__ == "__main__":
    main()
