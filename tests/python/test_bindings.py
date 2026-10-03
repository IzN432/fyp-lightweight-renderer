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


def make_viewer(on_execute=None, configure=None):
    """A viewer with one fullscreen pass (SPIR-V supplied as bytes) drawing to the swapchain.
    `configure(builder)` can add extra state to that pass."""
    viewer = lr.Viewer(title="lr binding test", width=320, height=240)
    fg = viewer.frame_graph
    tint = np.array([0.2, 0.4, 0.6, 1.0], dtype=np.float32)

    def execute(ctx):
        if on_execute:
            on_execute(ctx)
        ctx.push_constants(lr.Stage.FRAGMENT, tint)
        ctx.cmd.draw(3)

    builder = (
        fg.add_pass("tint")
        .type(lr.PassType.FULLSCREEN)
        .vert_shader(lr.compile_glsl_source(CLEAR_VERT, lr.ShaderStage.VERTEX, "clear.vert"))
        .frag_shader(lr.compile_glsl_source(CLEAR_FRAG, lr.ShaderStage.FRAGMENT, "clear.frag"))
        .push_constant_size(tint.nbytes, lr.Stage.FRAGMENT)
        .color_attachment(fg.image("swapchain"), viewer.swapchain_format)
        .execute(execute)
    )
    if configure:
        configure(builder)
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


def test_pipeline_state_runs():
    def configure(builder):
        builder.blend(lr.BlendMode.ADDITIVE).cull(lr.CullMode.NONE)

    viewer = make_viewer(configure=configure)
    frames = []
    viewer.on_update(lambda dt, extent: (frames.append(dt), len(frames) >= 5 and viewer.close()))
    viewer.run()
    assert len(frames) == 5


def test_invalid_pipeline_state_raises_at_compile():
    viewer = make_viewer(configure=lambda builder: builder.depth(test=True, write=True))
    try:
        viewer.run()
    except RuntimeError as e:
        assert "'tint'" in str(e) and "no depth attachment" in str(e), str(e)
    else:
        raise AssertionError("expected RuntimeError for depth state without a depth attachment")


def test_read_and_replace_buffers():
    viewer = make_viewer()
    res = viewer.resources
    data = np.arange(16, dtype=np.float32)
    res.upload_buffer("numbers", data, lr.BufferUsage.STORAGE)
    assert np.array_equal(res.read_buffer("numbers").view(np.float32), data)

    bigger = np.linspace(0.0, 1.0, 100, dtype=np.float32)
    res.replace_buffer("numbers", bigger, lr.BufferUsage.STORAGE)
    assert np.array_equal(res.read_buffer("numbers").view(np.float32), bigger)

    res.register_dynamic_buffer("params", 16, lr.BufferUsage.UNIFORM)
    res.update_buffer("params", np.array([1, 2, 3, 4], dtype=np.uint32))
    assert res.read_buffer("params").view(np.uint32).tolist() == [1, 2, 3, 4]


def test_indirect_draw_requires_declaration():
    # The pass draws from "args" without declaring it with indirect_buffer(), so the frame graph
    # couldn't have synchronised it.
    viewer = make_viewer(on_execute=lambda ctx: ctx.draw_indirect(viewer.frame_graph.buffer("args")))
    viewer.resources.upload_buffer("args", np.array([3, 1, 0, 0], dtype=np.uint32), lr.BufferUsage.INDIRECT)
    try:
        viewer.run()
    except RuntimeError as e:
        assert "without declaring it with indirectBuffer()" in str(e), str(e)
    else:
        raise AssertionError("expected an error for an undeclared indirect buffer")


RECORD_COMP = """
#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0) uniform Data { uint value; } data;
layout(set = 0, binding = 1) buffer History { uint seen[]; } history;
layout(push_constant) uniform PC { uint frame; } pc;
void main() { history.seen[pc.frame] = data.value; }
"""


def make_recorder(writes_dynamic=False):
    """A viewer whose compute pass stores the dynamic buffer "data"'s value in "history"[frame]."""
    viewer = make_viewer()
    fg, res = viewer.frame_graph, viewer.resources
    res.register_dynamic_buffer("data", 16, lr.BufferUsage.UNIFORM | lr.BufferUsage.STORAGE)
    res.upload_buffer("history", np.zeros(16, dtype=np.uint32), lr.BufferUsage.STORAGE)
    frame = [0]

    def record(ctx):
        ctx.push_constants(lr.Stage.COMPUTE, np.array([frame[0]], dtype=np.uint32))
        ctx.cmd.dispatch(1)

    builder = fg.add_pass("record").type(lr.PassType.COMPUTE)
    if writes_dynamic:
        source = RECORD_COMP.replace("uniform Data", "buffer Data")
        builder.compute_shader(lr.compile_glsl_source(source, lr.ShaderStage.COMPUTE))
        builder.storage_buffer_read_write(0, fg.buffer("data"), lr.Stage.COMPUTE)
    else:
        builder.compute_shader(lr.compile_glsl_source(RECORD_COMP, lr.ShaderStage.COMPUTE))
        builder.uniform_buffer(0, fg.buffer("data"), lr.Stage.COMPUTE)
    (
        builder.storage_buffer_read_write(1, fg.buffer("history"), lr.Stage.COMPUTE)
        .push_constant_size(4, lr.Stage.COMPUTE)
        .execute(record)
    )

    def update(dt, extent):
        frame[0] += 1
        if frame[0] >= 8:
            viewer.close()

    viewer.on_update(update)
    return viewer


def test_dynamic_data_written_once_reaches_every_frame():
    # Each frame in flight has its own copy; data written before run() must still reach all of them.
    viewer = make_recorder()
    viewer.resources.update_buffer("data", np.array([1234], dtype=np.uint32))
    viewer.run()
    seen = viewer.resources.read_buffer("history").view(np.uint32)
    assert seen[1:8].tolist() == [1234] * 7, seen[:8].tolist()


def test_gpu_writes_to_dynamic_buffers_are_rejected():
    viewer = make_recorder(writes_dynamic=True)
    try:
        viewer.run()
    except RuntimeError as e:
        assert "writes dynamic buffer 'data'" in str(e), str(e)
    else:
        raise AssertionError("expected an error for a GPU write to a dynamic buffer")


def main():
    tests = [
        test_compile_errors_raise_with_location,
        test_runs_frames_and_closes,
        test_update_callback_exception_is_reraised,
        test_execute_callback_exception_is_reraised,
        test_resource_errors_raise_immediately,
        test_pipeline_state_runs,
        test_invalid_pipeline_state_raises_at_compile,
        test_read_and_replace_buffers,
        test_indirect_draw_requires_declaration,
        test_dynamic_data_written_once_reaches_every_frame,
        test_gpu_writes_to_dynamic_buffers_are_rejected,
    ]
    for test in tests:
        test()
        gc.collect()  # each test's Viewer must be fully destroyed before the next creates one
        print(f"ok   {test.__name__}")
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    main()
