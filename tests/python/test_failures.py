"""Failures originating inside lr itself (not in user code) — they must surface as Python errors or
clear crash reports, never hang, and never leave the GPU in use while the Viewer is torn down.

Uses the private failure-injection hooks in lr._lr._testing. Needs a GPU and a display.

Run via ctest (`python.failures`), or directly:
    PYTHONPATH=build/python python tests/python/test_failures.py
"""

import gc
import os
import subprocess
import sys

import numpy as np

import lr
from lr._lr import _testing

VERT = "#version 450\nvoid main() { gl_Position = vec4(vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2) * 2.0 - 1.0, 0.0, 1.0); }\n"
# A deliberately slow fragment shader (~100 ms/frame at 640x360 with 600k iterations on an RTX 5070 Ti),
# so earlier frames are still executing on the GPU when a failure unwinds the frame loop.
FRAG = """#version 450
layout(location = 0) out vec4 c;
layout(push_constant) uniform PC { uint iterations; } pc;
void main() {
    float v = gl_FragCoord.x * 0.001;
    for (uint i = 0u; i < pc.iterations; ++i) { v = fract(sin(v * 12.9898 + float(i)) * 43758.5453); }
    c = vec4(v, v, v, 1.0);
}
"""


def make_viewer(iterations=0):
    viewer = lr.Viewer(title="lr failure test", width=640, height=360)
    fg = viewer.frame_graph
    pc = np.array([iterations], dtype=np.uint32)

    def draw(ctx):
        ctx.push_constants(lr.Stage.FRAGMENT, pc)
        ctx.cmd.draw(3)

    (
        fg.add_pass("fill")
        .type(lr.PassType.FULLSCREEN)
        .vert_shader(lr.compile_glsl_source(VERT, lr.ShaderStage.VERTEX))
        .frag_shader(lr.compile_glsl_source(FRAG, lr.ShaderStage.FRAGMENT))
        .push_constant_size(4, lr.Stage.FRAGMENT)
        .color_attachment(fg.image("swapchain"), viewer.swapchain_format)
        .execute(draw)
    )
    return viewer


def run_frames(viewer, frames):
    count = [0]

    def update(dt, extent):
        count[0] += 1
        if count[0] >= frames:
            viewer.close()

    viewer.on_update(update)
    viewer.run()
    return count[0]


def test_library_exception_mid_frame_tears_down_cleanly():
    # Thrown after several frames, so earlier frames are still in flight on the GPU when it unwinds;
    # ctest fails this test on any "[error]" line, i.e. a validation error from destroying in-use objects.
    viewer = make_viewer(iterations=600_000)
    _testing.throw_in_frame_loop(viewer, after_frames=5)
    try:
        run_frames(viewer, frames=1000)
    except RuntimeError as e:
        assert "injected C++ failure inside the frame loop" in str(e), str(e)
    else:
        raise AssertionError("expected RuntimeError from run()")
    del viewer
    gc.collect()

    # The process is still healthy: a new Viewer renders normally.
    assert run_frames(make_viewer(), frames=5) == 5


def crash_in_subprocess(kind):
    """Run lr._testing.crash(kind) in a child interpreter; returns (exit code, stderr). Fails on a hang."""
    code = f"import lr\nfrom lr._lr import _testing\n_testing.crash({kind!r})\n"
    try:
        result = subprocess.run(
            # A hang here usually means a modal error dialog; the timeout kills the child, closing it.
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=20, env=dict(os.environ)
        )
    except subprocess.TimeoutExpired:
        raise AssertionError(f"crash({kind!r}) hung instead of exiting (e.g. on a modal error dialog)")
    return result.returncode, result.stderr


def test_access_violation_reports_python_location():
    returncode, stderr = crash_in_subprocess("access_violation")
    assert returncode != 0
    assert "access violation" in stderr.lower(), stderr
    assert 'File "<string>", line 3' in stderr, f"expected the Python line that called into lr:\n{stderr}"


def test_terminate_exits_with_report():
    returncode, stderr = crash_in_subprocess("terminate")
    assert returncode != 0
    assert 'File "<string>", line 3' in stderr, f"expected the Python line that called into lr:\n{stderr}"


def test_debug_assertion_exits_instead_of_hanging():
    if not _testing.DEBUG_BUILD:
        print("skip test_debug_assertion_exits_instead_of_hanging (release build)")
        return
    returncode, stderr = crash_in_subprocess("debug_assert")
    assert returncode != 0
    assert "out of range" in stderr, stderr
    assert 'File "<string>", line 3' in stderr, f"expected the Python line that called into lr:\n{stderr}"


def test_plain_assert_reports_python_location():
    # assert() reports through _wassert, separately from the CRT report hook (e.g. ImGui's IM_ASSERT).
    if not _testing.DEBUG_BUILD:
        print("skip test_plain_assert_reports_python_location (release build: assert is compiled out)")
        return
    returncode, stderr = crash_in_subprocess("assert")
    assert returncode != 0
    assert "plain assert" in stderr, stderr
    assert 'File "<string>", line 3' in stderr, f"expected the Python line that called into lr:\n{stderr}"


def main():
    tests = [
        test_library_exception_mid_frame_tears_down_cleanly,
        test_access_violation_reports_python_location,
        test_terminate_exits_with_report,
        test_debug_assertion_exits_instead_of_hanging,
        test_plain_assert_reports_python_location,
    ]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"ok   {test.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {test.__name__}: {e}")
        gc.collect()
    print(f"{len(tests) - failures} passed, {failures} failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
