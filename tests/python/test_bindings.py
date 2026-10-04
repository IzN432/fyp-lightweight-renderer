"""Behavioural tests for the lr Python bindings. Needs a GPU and a display (opens short-lived windows).

Run via ctest (`python.bindings`), or directly:
    PYTHONPATH=build/python python tests/python/test_bindings.py
"""

import gc
from typing import Any

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

    def update(dt, extent):
        frames.append(dt)
        if len(frames) >= 5:
            viewer.close()

    viewer.on_update(update)
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
    def draw(ctx):
        ctx.draw_indirect(viewer.frame_graph.buffer("args"))

    viewer = make_viewer(on_execute=draw)
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


def run_frames(viewer, count, per_frame=None):
    frame = [0]

    def update(dt, extent):
        if per_frame:
            per_frame(frame[0])
        frame[0] += 1
        if frame[0] >= count:
            viewer.close()

    viewer.on_update(update)
    viewer.run()


def test_gui_outside_on_gui_raises():
    make_viewer()  # an ImGui context exists, but no frame is being built
    try:
        lr.gui.text("hello")
    except RuntimeError as e:
        assert "on_gui" in str(e), str(e)
    else:
        raise AssertionError("expected RuntimeError outside on_gui")
    assert lr.gui.want_capture_mouse() in (True, False)  # safe anywhere


def test_gui_widgets_and_window_balancing():
    viewer = make_viewer()
    seen: dict[str, Any] = {}

    def gui():
        with lr.gui.window("Controls") as visible:
            seen["visible"] = visible
            lr.gui.text("text")
            seen["button"] = lr.gui.button("Button")
            seen["checkbox"] = lr.gui.checkbox("Check", True)
            seen["slider"] = lr.gui.slider_float("Float", 0.5, 0.0, 1.0)
            seen["slider_int"] = lr.gui.slider_int("Int", 3, 0, 10)
            seen["drag"] = lr.gui.drag_float("Drag", 1.5)
            seen["color"] = lr.gui.color_edit3("Color", (0.1, 0.2, 0.3))
            seen["color4"] = lr.gui.color_edit4("Color4", (0.1, 0.2, 0.3, 1.0))
            seen["combo"] = lr.gui.combo("Combo", 1, ["a", "b", "c"])
            seen["header"] = lr.gui.collapsing_header("Header")
            lr.gui.separator()
            lr.gui.same_line()
            lr.gui.spacing()
        lr.gui.begin("Left open")  # never ended: closed automatically after the callback

    viewer.on_gui(gui)
    run_frames(viewer, 3)
    assert seen["checkbox"] == (False, True), seen["checkbox"]
    assert seen["slider"] == (False, 0.5) and seen["slider_int"] == (False, 3)
    assert seen["combo"] == (False, 1) and seen["header"] is True
    assert seen["color"][1] == tuple(np.array([0.1, 0.2, 0.3], dtype=np.float32).tolist())


def test_gui_exception_closes_window_and_reraises():
    viewer = make_viewer()

    def gui():
        with lr.gui.window("Broken"):
            raise ValueError("boom from on_gui")

    viewer.on_gui(gui)
    try:
        run_frames(viewer, 50)
    except ValueError as e:
        assert str(e) == "boom from on_gui"
    else:
        raise AssertionError("expected ValueError from run()")


def test_input_reflects_injected_events():
    viewer = make_viewer()
    seen = []

    def per_frame(frame):
        if frame == 0:
            _testing_inject(viewer)
        elif frame == 1:  # events injected during frame 0 are visible from frame 1
            inp = viewer.input
            seen.append((inp.mouse_delta, inp.scroll_delta, inp.is_mouse_down(lr.MouseButton.LEFT),
                         inp.is_key_down(lr.Key.W), inp.shift))

    run_frames(viewer, 3, per_frame)
    delta, scroll, left, w, shift = seen[0]
    assert left and w and shift, seen[0]
    assert scroll == 2.0, scroll
    assert delta[0] != 0.0 or delta[1] != 0.0, delta


def test_orbit_camera_is_the_engine_camera():
    viewer = make_viewer()
    camera = lr.OrbitCamera(viewer)
    camera.target, camera.radius, camera.azimuth, camera.elevation = (1.0, 2.0, 3.0), 5.0, 0.5, 0.25

    # SphericalCameraController's placement: on a sphere around the target.
    expected = np.array([1.0, 2.0, 3.0]) + 5.0 * np.array(
        [np.cos(0.25) * np.sin(0.5), np.sin(0.25), np.cos(0.25) * np.cos(0.5)])
    assert np.allclose(camera.position, expected, atol=1e-5), (camera.position, expected)

    # The view puts the target straight ahead at `radius`; the projection maps near/far to depth 0/1.
    view = camera.view_matrix()
    assert np.allclose(view @ np.array([1.0, 2.0, 3.0, 1.0]), [0.0, 0.0, -5.0, 1.0], atol=1e-4)
    proj = camera.projection_matrix(16 / 9)
    for distance, depth in ((camera.near_plane, 0.0), (camera.far_plane, 1.0)):
        clip = proj @ np.array([0.0, 0.0, -distance, 1.0])
        assert abs(clip[2] / clip[3] - depth) < 1e-4, (distance, clip)

    # matrices() is (view, proj) in GLSL mat4 layout, as lr.transforms.to_gpu produces.
    assert np.allclose(camera.matrices((1600, 900)), lr.transforms.to_gpu(view, proj), atol=1e-6)

    camera.elevation, camera.radius = 3.0, 0.0  # clamped like the C++ controller
    assert abs(camera.elevation - np.radians(89.0)) < 1e-5 and camera.radius == np.float32(0.01)


def _testing_inject(viewer):
    from lr._lr import _testing

    x, y = viewer.input.mouse_position
    _testing.inject_mouse_move(viewer, x + 40.0, y + 25.0)
    _testing.inject_mouse_button(viewer, lr.MouseButton.LEFT, True)
    _testing.inject_scroll(viewer, 2.0)
    _testing.inject_key(viewer, lr.Key.W, True)
    _testing.inject_key(viewer, lr.Key.LEFT_SHIFT, True)


SAMPLE_FRAG = """
#version 450
layout(location = 0) out vec4 outColor;
layout(set = 0, binding = 0) uniform sampler2D source;
void main() { outColor = texture(source, gl_FragCoord.xy / vec2(textureSize(source, 0))); }
"""
UNIFORM_FRAG = """
#version 450
layout(location = 0) out vec4 outColor;
layout(set = 0, binding = 0) uniform Tint { vec4 color; } tint;
void main() { outColor = tint.color; }
"""


def test_recompile_before_the_first_frame_runs():
    # A recompile before the graph has executed once (here: a uniform buffer replaced in the first
    # frame's update) must not assume the images are already in the layouts a frame would leave them in.
    viewer = lr.Viewer(title="lr binding test", width=320, height=240)
    fg, res = viewer.frame_graph, viewer.resources
    res.upload_buffer("tint", np.array([1.0, 0.5, 0.25, 1.0], dtype=np.float32), lr.BufferUsage.UNIFORM)
    vert = lr.compile_glsl_source(CLEAR_VERT, lr.ShaderStage.VERTEX, "clear.vert")
    (
        fg.add_pass("offscreen")
        .type(lr.PassType.FULLSCREEN)
        .vert_shader(vert)
        .frag_shader(lr.compile_glsl_source(UNIFORM_FRAG, lr.ShaderStage.FRAGMENT, "uniform.frag"))
        .uniform_buffer(0, fg.buffer("tint"), lr.Stage.FRAGMENT)
        .color_attachment(fg.image("offscreen"), lr.Format.R16G16B16A16_SFLOAT)
        .execute(lambda ctx: ctx.cmd.draw(3))
    )
    (
        fg.add_pass("present")
        .type(lr.PassType.FULLSCREEN)
        .vert_shader(vert)
        .frag_shader(lr.compile_glsl_source(SAMPLE_FRAG, lr.ShaderStage.FRAGMENT, "sample.frag"))
        .sampled_image(0, fg.image("offscreen"), lr.Stage.FRAGMENT)
        .color_attachment(fg.image("swapchain"), viewer.swapchain_format)
        .execute(lambda ctx: ctx.cmd.draw(3))
    )
    frames = [0]

    def update(dt, extent):
        frames[0] += 1
        if frames[0] == 1:
            res.replace_buffer("tint", np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32), lr.BufferUsage.UNIFORM)
        if frames[0] == 3:
            viewer.close()

    viewer.on_update(update)
    viewer.run()  # raises VulkanValidationError if the recompiled graph skipped the initial transitions
    assert fg.compile_count == 2, fg.compile_count


def main():
    tests = [
        test_recompile_before_the_first_frame_runs,
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
        test_gui_outside_on_gui_raises,
        test_gui_widgets_and_window_balancing,
        test_gui_exception_closes_window_and_reraises,
        test_input_reflects_injected_events,
        test_orbit_camera_is_the_engine_camera,
    ]
    for test in tests:
        test()
        gc.collect()  # each test's Viewer must be fully destroyed before the next creates one
        print(f"ok   {test.__name__}")
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    main()
