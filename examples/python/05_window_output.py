"""Contract: opaque backbuffer, frame execution, and automatic presentation."""

from pathlib import Path

import lightweight_renderer as lr


HERE = Path(__file__).resolve().parent
renderer = lr.Renderer()
window = renderer.create_window(title="Python triangle", width=1280, height=720)

graph = lr.FrameGraphBuilder(renderer)
backbuffer = graph.import_backbuffer(window, label="window")

pipeline = renderer.create_graphics_pipeline(
    vertex_shader=renderer.load_shader(
        HERE / "shaders" / "fullscreen.vert", lr.ShaderStage.VERTEX
    ),
    fragment_shader=renderer.load_shader(
        HERE / "shaders" / "display.frag", lr.ShaderStage.FRAGMENT
    ),
    color_formats=[backbuffer.format],
)


def draw(encoder: lr.CommandEncoder) -> None:
    encoder.bind_pipeline(pipeline)
    encoder.draw(vertex_count=3)


graph.add_render_pass(
    "display",
    color_attachments=[
        lr.ColorAttachment(
            backbuffer,
            load=lr.LoadOp.CLEAR,
            clear=(0.02, 0.03, 0.05, 1.0),
        )
    ],
    execute=draw,
)

# Exporting as PRESENT is implicit for an imported backbuffer. The run loop owns
# acquire, external-image binding, resize, submission, and presentation.
compiled = graph.compile(outputs=[backbuffer])
renderer.run(window, compiled)
