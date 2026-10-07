"""Live editing — changing resources and passes while the renderer runs, with no manual waits or recompiles.

  every frame: the "trail" vertex buffer is replaced with one more point (replace_buffer). The old
               buffer is kept until the frames still using it finish, and nothing recompiles: vertex,
               index and indirect buffers are looked up by name every frame.
  frame 120:   a new pass "dots" is added mid-run, drawing a quad per trail point (the same buffer, read
               as per-instance data). The graph recompiles on the next frame, and the ImGui overlay
               stays last.
  frame 240:   the "style" uniform buffer both passes read is reallocated at a new size
               (replace_dynamic_buffer). Their descriptor sets point at the old buffer, so the graph
               recompiles once more.

FrameGraph.compile_count is checked at each step.

Run from the repo root after building:
    PYTHONPATH=build/python python examples/python/live_edit.py [--frames N]
"""

import argparse
import pathlib
from typing import Any

import numpy as np

import lr

SHADERS = pathlib.Path(__file__).parent / "shaders"
ADD_DOTS_AT = 120
RESTYLE_AT = 240


def spiral(point_count):
    t = np.linspace(0.0, 1.0, point_count, dtype=np.float32)
    angle = t * 9.0 * np.pi
    radius = 0.05 + 0.8 * t
    return np.stack([radius * np.cos(angle) * 0.6, radius * np.sin(angle)], axis=-1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--frames", type=int, default=0, help="close after this many frames (0 = run until closed)")
    args = parser.parse_args()

    viewer = lr.Viewer(title="lr - live editing (Python)", width=1280, height=720)
    fg, res = viewer.frame_graph, viewer.resources
    swapchain, trail, style = fg.image(lr.SWAPCHAIN), fg.buffer("trail"), fg.buffer("style")

    state: dict[str, Any] = {"frames": 0, "points": spiral(2), "compiles": {}}
    res.upload_buffer("trail", state["points"], lr.BufferUsage.VERTEX)
    res.register_dynamic_buffer("style", 32, lr.BufferUsage.UNIFORM)
    res.update_buffer("style", np.array([0.9, 0.9, 0.95, 1, 1.0, 0.45, 0.2, 1], dtype=np.float32))

    (
        fg.add_pass("trail")
        .type(lr.PassType.GEOMETRY)
        .topology(lr.Topology.LINE_STRIP)
        .vert_shader(SHADERS / "trail.vert")
        .frag_shader(SHADERS / "trail.frag")
        .vertex_layout([lr.VertexBinding(0, stride=8)], [lr.VertexAttribute(0, lr.Format.R32G32_SFLOAT)])
        .vertex_buffer(0, trail)
        .uniform_buffer(0, style, lr.Stage.FRAGMENT)
        .color_attachment(swapchain, viewer.swapchain_format, clear_color=(0.06, 0.06, 0.09, 1.0))
        .execute(lambda ctx: ctx.cmd.draw(len(state["points"])))
    )

    def add_dots_pass():
        def draw_dots(ctx):
            width, height = ctx.rendering_extent
            ctx.push_constants(lr.Stage.VERTEX, np.array([0.008 * height / width, 0.008], dtype=np.float32))
            ctx.cmd.draw(6, instance_count=len(state["points"]))

        (
            fg.add_pass("dots")
            .type(lr.PassType.GEOMETRY)
            .cull(lr.CullMode.NONE)
            .vert_shader(SHADERS / "dots.vert")
            .frag_shader(SHADERS / "dots.frag")
            .vertex_layout(
                [lr.VertexBinding(0, stride=8, per_instance=True)], [lr.VertexAttribute(0, lr.Format.R32G32_SFLOAT)]
            )
            .vertex_buffer(0, trail)
            .uniform_buffer(0, style, lr.Stage.FRAGMENT)
            .push_constant_size(8, lr.Stage.VERTEX)
            .color_attachment(swapchain, viewer.swapchain_format, load_op=lr.LoadOp.LOAD)
            .execute(draw_dots)
        )

    def update(dt, extent):
        frame = state["frames"]
        state["compiles"][frame] = fg.compile_count  # as of the end of the previous frame

        state["points"] = spiral(min(frame + 3, 600))
        res.replace_buffer("trail", state["points"], lr.BufferUsage.VERTEX)

        if frame == ADD_DOTS_AT:
            add_dots_pass()
        if frame == RESTYLE_AT:
            res.replace_dynamic_buffer("style", 64, lr.BufferUsage.UNIFORM)
            res.update_buffer("style", np.array([0.3, 0.8, 1.0, 1, 1.0, 1.0, 1.0, 1], dtype=np.float32))

        state["frames"] += 1
        if args.frames and state["frames"] >= args.frames:
            viewer.close()

    update_connection = viewer.on_update(update)
    viewer.run()

    compiles = state["compiles"]
    if args.frames > RESTYLE_AT + 1:
        # run() compiles once; each change recompiles exactly once, on the frame it happened in.
        assert compiles[1] == 1, compiles[1]
        assert compiles[ADD_DOTS_AT] == 1, "replacing a vertex buffer every frame must not recompile"
        assert compiles[ADD_DOTS_AT + 1] == 2, "adding a pass should recompile once"
        assert compiles[RESTYLE_AT] == 2
        assert compiles[RESTYLE_AT + 1] == 3, "replacing a uniform buffer should recompile once"
        assert compiles[max(compiles)] == 3
        print(f"{state['frames']} frames, {len(state['points'])} trail points, {fg.compile_count} compiles: "
              "1 at start, 1 for the added pass, 1 for the replaced uniform buffer")


if __name__ == "__main__":
    main()
