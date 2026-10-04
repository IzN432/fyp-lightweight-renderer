"""GPU-driven instancing — compute shaders decide what to draw, one indirect call draws it, and the
results are read back into numpy and checked.

  pass "reset" (compute): writes a fresh indexed-indirect draw command (36 indices, 0 instances)
  pass "cull"  (compute): animates a 100x100 grid of instances and compacts the ones inside a breathing
                          radius into the "instances" buffer, atomically counting them into the command
  pass "draw":            one draw_indexed_indirect() of a cube, reading "instances" as a per-instance
                          vertex buffer

The CPU never learns how many instances are visible. The frame graph orders the three passes from the
buffers they share and inserts the barriers: storage write -> storage read/write -> vertex-attribute and
indirect-command reads.

Every 120 frames the animation holds still for a few frames, then read_buffer() copies both buffers back,
and they're checked against a numpy re-implementation of instances_cull.comp.

Run from the repo root after building:
    PYTHONPATH=build/python python examples/python/gpu_instancing.py [--frames N]
"""

import argparse
import pathlib
from typing import Any

import numpy as np

import lr
from lr import transforms as tf
from meshes import make_cube

SHADERS = pathlib.Path(__file__).parent / "shaders"

GRID = 100  # GRID x GRID instances
SPACING = 0.5
VISIBLE_RADIUS = 22.0
INSTANCE_FLOATS = 8  # vec4 offsetScale + vec4 color
VERIFY_EVERY = 120
HOLD_FRAMES = 4  # > frames in flight, so every submitted frame used the same time


def cull_push_constants(time):
    pc = np.array([time, 0.0, SPACING, VISIBLE_RADIUS], dtype=np.float32)
    pc.view(np.uint32)[1] = GRID
    return pc


def cell_index(xz):
    """Integer grid cell of a position (cells sit at half-integer multiples of SPACING)."""
    return tuple(np.rint(np.asarray(xz) / SPACING + 0.5 * (GRID - 1)).astype(int))


def reference_instances(time):
    """numpy version of instances_cull.comp: (sure, ambiguous) dicts keyed by grid cell -> (8,) float32 row.
    "ambiguous" holds instances so close to the radius that float rounding may decide either way."""
    i = np.arange(GRID * GRID)
    cell = np.stack([i % GRID, i // GRID], axis=-1).astype(np.float32) - 0.5 * (GRID - 1)
    xz = cell * SPACING
    d = np.linalg.norm(xz, axis=-1)
    radius = VISIBLE_RADIUS * (0.75 + 0.25 * np.sin(time * 0.7))
    y = 0.6 * np.sin(d * 0.35 - time * 2.0)
    color = 0.5 + 0.5 * np.cos(np.array([0.0, 2.1, 4.2])[None, :] + (d * 0.15 - time)[:, None])
    rows = np.concatenate(
        [xz[:, :1], y[:, None], xz[:, 1:], np.full((len(i), 1), 0.4), color, np.ones((len(i), 1))], axis=-1
    ).astype(np.float32)
    keys = [cell_index(p) for p in xz]
    sure = {keys[k]: rows[k] for k in np.nonzero(d < radius - 1e-3)[0]}
    ambiguous = {keys[k]: rows[k] for k in np.nonzero(np.abs(d - radius) <= 1e-3)[0]}
    return sure, ambiguous


def verify_readback(res, time, index_count):
    args = res.read_buffer("draw_args").view(np.uint32)
    count = int(args[1])
    assert args[0] == index_count, f"indexCount {args[0]} != {index_count}"
    gpu = res.read_buffer("instances").view(np.float32).reshape(-1, INSTANCE_FLOATS)[:count]
    sure, ambiguous = reference_instances(time)

    assert len(sure) <= count <= len(sure) + len(ambiguous), f"GPU kept {count}, numpy expects {len(sure)}"
    gpu_by_cell = {cell_index(row[[0, 2]]): row for row in gpu}
    assert len(gpu_by_cell) == count, "two GPU instances landed on the same grid cell"
    missing = sure.keys() - gpu_by_cell.keys()
    assert not missing, f"{len(missing)} instances numpy keeps are missing on the GPU"
    unexpected = gpu_by_cell.keys() - sure.keys() - ambiguous.keys()
    assert not unexpected, f"{len(unexpected)} instances the GPU kept should have been culled"
    for cell, row in gpu_by_cell.items():
        expected = sure.get(cell, ambiguous.get(cell))
        assert np.allclose(row, expected, atol=2e-3), f"cell {cell}: GPU {row} vs numpy {expected}"
    return count


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--frames", type=int, default=0, help="close after this many frames (0 = run until closed)")
    args = parser.parse_args()

    viewer = lr.Viewer(title="lr - GPU instancing (Python)", width=1280, height=720)
    fg, res = viewer.frame_graph, viewer.resources

    cube_vertices, cube_indices = make_cube()
    res.upload_buffer("cube_vertices", cube_vertices, lr.BufferUsage.VERTEX)
    res.upload_buffer("cube_indices", cube_indices, lr.BufferUsage.INDEX)
    res.register_static_buffer(
        "instances", GRID * GRID * INSTANCE_FLOATS * 4, lr.BufferUsage.STORAGE | lr.BufferUsage.VERTEX
    )
    res.register_static_buffer("draw_args", 5 * 4, lr.BufferUsage.STORAGE | lr.BufferUsage.INDIRECT)
    res.register_dynamic_buffer("camera", 2 * 64, lr.BufferUsage.UNIFORM)
    instances, draw_args = fg.buffer("instances"), fg.buffer("draw_args")

    state: dict[str, Any] = {"time": 0.0, "frames": 0, "hold": 0, "checks": 0}
    index_count = np.array([len(cube_indices)], dtype=np.uint32)

    def reset(ctx):
        ctx.push_constants(lr.Stage.COMPUTE, index_count)
        ctx.cmd.dispatch(1, 1)

    def cull(ctx):
        ctx.push_constants(lr.Stage.COMPUTE, cull_push_constants(state["time"]))
        ctx.cmd.dispatch((GRID * GRID + 63) // 64, 1)

    (
        fg.add_pass("reset")
        .type(lr.PassType.COMPUTE)
        .compute_shader(SHADERS / "instances_reset.comp")
        .storage_buffer_write(0, draw_args, lr.Stage.COMPUTE)
        .push_constant_size(4, lr.Stage.COMPUTE)
        .execute(reset)
    )
    (
        fg.add_pass("cull")
        .type(lr.PassType.COMPUTE)
        .compute_shader(SHADERS / "instances_cull.comp")
        .storage_buffer_write(0, instances, lr.Stage.COMPUTE)
        .storage_buffer_read_write(1, draw_args, lr.Stage.COMPUTE)
        .push_constant_size(16, lr.Stage.COMPUTE)
        .execute(cull)
    )
    (
        fg.add_pass("draw")
        .type(lr.PassType.GEOMETRY)
        .vert_shader(SHADERS / "instanced.vert")
        .frag_shader(SHADERS / "instanced.frag")
        .vertex_layout(
            [
                lr.VertexBinding(0, stride=cube_vertices.strides[0]),
                lr.VertexBinding(1, stride=INSTANCE_FLOATS * 4, per_instance=True),
            ],
            [
                lr.VertexAttribute(0, lr.Format.R32G32B32_SFLOAT, offset=0, binding=0),
                lr.VertexAttribute(1, lr.Format.R32G32B32_SFLOAT, offset=12, binding=0),
                lr.VertexAttribute(2, lr.Format.R32G32B32A32_SFLOAT, offset=0, binding=1),
                lr.VertexAttribute(3, lr.Format.R32G32B32A32_SFLOAT, offset=16, binding=1),
            ],
        )
        .vertex_buffer(0, fg.buffer("cube_vertices"))
        .vertex_buffer(1, instances)
        .index_buffer(fg.buffer("cube_indices"))
        .indirect_buffer(draw_args)
        .uniform_buffer(0, fg.buffer("camera"), lr.Stage.VERTEX)
        .color_attachment(fg.image(lr.SWAPCHAIN), viewer.swapchain_format, clear_color=(0.05, 0.05, 0.07, 1.0))
        .depth_attachment(fg.image("depth"))
        .execute(lambda ctx: ctx.draw_indexed_indirect(draw_args))
    )

    def update(dt, extent):
        frame = state["frames"]
        if frame % VERIFY_EVERY == VERIFY_EVERY - 1 - HOLD_FRAMES:
            state["hold"] = HOLD_FRAMES
        if state["hold"] > 0:
            state["hold"] -= 1
            if state["hold"] == 0:
                count = verify_readback(res, state["time"], len(cube_indices))
                state["checks"] += 1
                print(f"frame {frame}: GPU kept {count} of {GRID * GRID} instances; readback matches numpy")
        else:
            state["time"] += dt

        width, height = extent
        view = tf.look_at(eye=(0.0, 16.0, 30.0), target=(0.0, -2.0, 0.0))
        proj = tf.perspective(45.0, width / max(height, 1), 0.1, 200.0)
        res.update_buffer("camera", tf.to_gpu(view, proj))

        state["frames"] += 1
        if args.frames and state["frames"] >= args.frames:
            viewer.close()

    viewer.on_update(update)
    viewer.run()
    if args.frames >= VERIFY_EVERY:
        assert state["checks"] >= args.frames // VERIFY_EVERY, "expected a readback check every 120 frames"


if __name__ == "__main__":
    main()
