"""Contract: replace a pass in a reusable Python-authored pipeline."""

from pathlib import Path

import lightweight_renderer as lr


HERE = Path(__file__).resolve().parent
renderer = lr.Renderer()
window = renderer.create_window(title="Replace a pass", width=1280, height=720)

graph = lr.FrameGraphBuilder(renderer)
backbuffer = graph.import_backbuffer(window)

default_pipeline = renderer.create_graphics_pipeline(
    vertex_shader=renderer.load_shader(
        HERE / "shaders" / "fullscreen.vert", lr.ShaderStage.VERTEX
    ),
    fragment_shader=renderer.load_shader(
        HERE / "shaders" / "display.frag", lr.ShaderStage.FRAGMENT
    ),
    color_formats=[backbuffer.format],
)


def default_display(encoder: lr.CommandEncoder) -> None:
    encoder.bind_pipeline(default_pipeline)
    encoder.draw(vertex_count=3)


display_pass = graph.add_render_pass(
    "display",
    color_attachments=[lr.ColorAttachment(backbuffer, load=lr.LoadOp.CLEAR)],
    execute=default_display,
)

replacement_pipeline = renderer.create_graphics_pipeline(
    vertex_shader=renderer.load_shader(
        HERE / "shaders" / "fullscreen.vert", lr.ShaderStage.VERTEX
    ),
    fragment_shader=renderer.load_shader(
        HERE / "shaders" / "inverted_display.frag", lr.ShaderStage.FRAGMENT
    ),
    color_formats=[backbuffer.format],
)


def inverted_display(encoder: lr.CommandEncoder) -> None:
    encoder.bind_pipeline(replacement_pipeline)
    encoder.draw(vertex_count=3)


# Replacement is legal only before compile, retains the original typed PassHandle,
# and re-runs validation. Resource declarations are supplied again rather than
# silently inherited from a potentially incompatible pass.
graph.replace_render_pass(
    display_pass,
    color_attachments=[lr.ColorAttachment(backbuffer, load=lr.LoadOp.CLEAR)],
    execute=inverted_display,
)

compiled = graph.compile(outputs=[backbuffer])
renderer.run(window, compiled)
