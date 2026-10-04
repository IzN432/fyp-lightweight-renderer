"""Per-frame data — checks that uniform data written with update_buffer() each frame is the data that
frame's GPU work reads, even while earlier frames are still executing.

  pass "busy"   (fragment): deliberately slow, so the GPU is still on frame N when the CPU starts N+1
  pass "record" (compute):  runs after "busy" (it samples its output), reads "frame_data" (a uniform
                            buffer Python updates every frame with the frame number), and stores what it
                            read in "history" at this frame's slot. The slot comes from a push constant,
                            which is baked into the frame's commands and can't race.
  pass "show":              copies "busy" to the window

Afterwards, read_buffer("history") shows, for each frame, which frame's uniform data it actually saw.
With a single shared uniform buffer, a slow frame reads the *next* frame's value, because the CPU
overwrote it mid-flight. With per-frame copies, every frame reads its own.

Run from the repo root after building:
    PYTHONPATH=build/python python examples/python/per_frame_data.py [--frames N] [--iterations K]
"""

import argparse
import pathlib
from typing import Any

import numpy as np

import lr

SHADERS = pathlib.Path(__file__).parent / "shaders"
HISTORY = 256


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--iterations", type=int, default=150_000, help="per-pixel work in the slow pass")
    parser.add_argument("--expect-race", action="store_true", help="exit non-zero if NO frame saw stale data")
    args = parser.parse_args()
    assert args.frames <= HISTORY

    viewer = lr.Viewer(title="lr - per-frame data (Python)", width=640, height=360)
    fg, res = viewer.frame_graph, viewer.resources
    busy = fg.image("busy")

    res.register_dynamic_buffer("frame_data", 16, lr.BufferUsage.UNIFORM)
    res.upload_buffer("history", np.full(HISTORY, 0xFFFFFFFF, dtype=np.uint32), lr.BufferUsage.STORAGE)

    state: dict[str, Any] = {"frame": 0, "time": 0.0}

    def draw_busy(ctx):
        pc = np.array([args.iterations, 0], dtype=np.uint32)
        pc.view(np.float32)[1] = state["time"]
        ctx.push_constants(lr.Stage.FRAGMENT, pc)
        ctx.cmd.draw(3)

    def record(ctx):
        ctx.push_constants(lr.Stage.COMPUTE, np.array([state["frame"], HISTORY], dtype=np.uint32))
        ctx.cmd.dispatch(1, 1)

    fullscreen = lr.SHADER_DIR / "fullscreen.vert.spv"
    (
        fg.add_pass("busy")
        .type(lr.PassType.FULLSCREEN)
        .vert_shader(fullscreen)
        .frag_shader(SHADERS / "busy.frag")
        .push_constant_size(8, lr.Stage.FRAGMENT)
        .color_attachment(busy, lr.Format.R8G8B8A8_UNORM)
        .execute(draw_busy)
    )
    (
        fg.add_pass("record")
        .type(lr.PassType.COMPUTE)
        .compute_shader(SHADERS / "record_frame.comp")
        .uniform_buffer(0, fg.buffer("frame_data"), lr.Stage.COMPUTE)
        .storage_buffer_read_write(1, fg.buffer("history"), lr.Stage.COMPUTE)
        .sampled_image(2, busy, lr.Stage.COMPUTE)
        .push_constant_size(8, lr.Stage.COMPUTE)
        .execute(record)
    )
    (
        fg.add_pass("show")
        .type(lr.PassType.FULLSCREEN)
        .vert_shader(fullscreen)
        .frag_shader(SHADERS / "show.frag")
        .sampled_image(0, busy, lr.Stage.FRAGMENT)
        .color_attachment(fg.image(lr.SWAPCHAIN), viewer.swapchain_format)
        .execute(lambda ctx: ctx.cmd.draw(3))
    )

    def update(dt, extent):
        state["frame"] += 1
        state["time"] += dt
        # This frame's number, written into CPU-visible memory the GPU reads later in the frame.
        res.update_buffer("frame_data", np.array([state["frame"], 0, 0, 0], dtype=np.uint32))
        if state["frame"] >= args.frames:
            viewer.close()

    viewer.on_update(update)
    viewer.run()

    seen = res.read_buffer("history").view(np.uint32)
    frames = np.arange(1, args.frames + 1)
    recorded = seen[frames % HISTORY]
    stale = frames[recorded != frames]
    print(f"{len(stale)} of {args.frames} frames read another frame's uniform data", end="")
    if len(stale):
        offsets = (recorded[recorded != frames].astype(np.int64) - stale).tolist()
        print(f" (e.g. frame {stale[0]} saw frame {stale[0] + offsets[0]}; offsets seen: {sorted(set(offsets))})")
    else:
        print()

    if args.expect_race:
        assert len(stale) > 0, "expected stale reads (race demonstration), saw none"
    else:
        assert len(stale) == 0, "frames must read their own uniform data"


if __name__ == "__main__":
    main()
