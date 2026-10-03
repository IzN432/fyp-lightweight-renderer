"""Behavioural tests for the lr Python bindings. Needs a GPU and a display (opens short-lived windows).

Run via ctest (`python.bindings`), or directly:
    PYTHONPATH=build/python python tests/python/test_bindings.py
"""

import gc

import numpy as np

import lr

CLEAR_VERT = """
#version 450
void main() { gl_Position = vec4(vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2) * 2.0 - 1.0, 0.0, 1.0); }
"""
CLEAR_FRAG = """
#version 450
layout(location = 0) out vec4 outColor;
layout(push_constant) uniform Tint { vec4 color; } pc;
void main() { outColor = pc.color; }
"""


def make_viewer(on_execute=None):
    """A viewer with one fullscreen pass (SPIR-V supplied as bytes) drawing to the swapchain."""
    viewer = lr.Viewer(title="lr binding test", width=320, height=240)
    fg = viewer.frame_graph
    tint = np.array([0.2, 0.4, 0.6, 1.0], dtype=np.float32)

    def execute(ctx):
        if on_execute:
            on_execute(ctx)
        ctx.push_constants(lr.Stage.FRAGMENT, tint)
        ctx.cmd.draw(3)

    (
        fg.add_pass("tint")
        .type(lr.PassType.FULLSCREEN)
        .vert_shader(lr.compile_glsl_source(CLEAR_VERT, lr.ShaderStage.VERTEX, "clear.vert"))
        .frag_shader(lr.compile_glsl_source(CLEAR_FRAG, lr.ShaderStage.FRAGMENT, "clear.frag"))
        .push_constant_size(tint.nbytes, lr.Stage.FRAGMENT)
        .color_attachment(fg.image("swapchain"), viewer.swapchain_format)
        .execute(execute)
    )
    return viewer


def test_compile_errors_raise_with_location():
    spirv = lr.compile_glsl_source(CLEAR_FRAG, lr.ShaderStage.FRAGMENT)
    assert int.from_bytes(spirv[:4], "little") == 0x07230203
    try:
        lr.compile_glsl_source("#version 450\nvoid main() { nope = 1; }", lr.ShaderStage.FRAGMENT, "bad.frag")
    except lr.ShaderCompileError as e:
        assert "bad.frag:2" in str(e) and "nope" in str(e), str(e)
    else:
        raise AssertionError("expected ShaderCompileError")


def test_runs_frames_and_closes():
    viewer = make_viewer()
    frames = []

    def update(dt, extent):
        assert extent == (320, 240) or extent[0] > 0
        frames.append(dt)
        if len(frames) == 10:
            viewer.close()

    viewer.on_update(update)
    viewer.run()
    assert len(frames) == 10


def test_update_callback_exception_is_reraised():
    viewer = make_viewer()
    calls = []

    def update(dt, extent):
        calls.append(1)
        if len(calls) == 3:
            raise ValueError("boom from on_update")

    viewer.on_update(update)
    try:
        viewer.run()
    except ValueError as e:
        assert str(e) == "boom from on_update"
    else:
        raise AssertionError("expected ValueError from run()")
    assert len(calls) == 3, "the window should close after the frame that raised"


def test_execute_callback_exception_is_reraised():
    def fail(ctx):
        raise KeyError("boom from execute")

    viewer = make_viewer(on_execute=fail)
    try:
        viewer.run()
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError from run()")


def test_resource_errors_raise_immediately():
    viewer = make_viewer()
    res = viewer.resources
    res.register_dynamic_buffer("small", 16, lr.BufferUsage.UNIFORM)
    res.update_buffer("small", np.zeros(4, dtype=np.float32))
    try:
        res.update_buffer("small", np.zeros(8, dtype=np.float32))
    except RuntimeError as e:
        assert "overflow" in str(e)
    else:
        raise AssertionError("expected RuntimeError for an oversized update")
    try:
        res.upload_image("img", np.zeros((4, 4, 3), dtype=np.uint8), lr.Format.R8G8B8A8_UNORM)
    except ValueError as e:
        assert "byte size" in str(e)
    else:
        raise AssertionError("expected ValueError for a mismatched image upload")


def main():
    tests = [
        test_compile_errors_raise_with_location,
        test_runs_frames_and_closes,
        test_update_callback_exception_is_reraised,
        test_execute_callback_exception_is_reraised,
        test_resource_errors_raise_immediately,
    ]
    for test in tests:
        test()
        gc.collect()  # each test's Viewer must be fully destroyed before the next creates one
        print(f"ok   {test.__name__}")
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    main()
