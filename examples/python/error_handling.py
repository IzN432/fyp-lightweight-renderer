"""Error handling — common mistakes surface as Python exceptions that name the problem, and the
interpreter keeps running.

Each scenario builds a small one-pass renderer with one deliberate mistake, catches the exception
and prints it; the last one has no mistake and renders normally, in the same interpreter.

  1. GLSL syntax error              -> lr.ShaderCompileError, at the line declaring the shader
  2. binding number off by one      -> lr.ShaderInterfaceError, before any Vulkan object is created
  3. push-constant block too small  -> lr.ShaderInterfaceError
  4. pushing too many bytes         -> ValueError from the execute callback, re-raised by run()
  5. invalid Vulkan usage           -> lr.VulkanValidationError (a validation-layer error)

Run from the repo root after building:
    PYTHONPATH=build/python python examples/python/error_handling.py
"""

import gc

import numpy as np

import lr

VERT = """
#version 450
void main() { gl_Position = vec4(vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2) * 2.0 - 1.0, 0.0, 1.0); }
"""
FRAG = """
#version 450
layout(location = 0) out vec4 outColor;
layout(set = 0, binding = 0) uniform Tint { vec4 color; } tint;
layout(push_constant) uniform Scale { vec4 scale; } pc;   // 16 bytes
void main() { outColor = tint.color * pc.scale; }
"""


def run_pass(frag_source=FRAG, tint_binding=0, push_size=16, push_data=None, record_extra=None, frames=5):
    """Build and run a one-pass renderer: a fullscreen triangle tinted by a UBO and push constants."""
    viewer = lr.Viewer(title="lr - error handling", width=480, height=270)
    fg, res = viewer.frame_graph, viewer.resources
    res.register_dynamic_buffer("tint", 16, lr.BufferUsage.UNIFORM)
    res.update_buffer("tint", np.array([0.3, 0.6, 0.9, 1.0], dtype=np.float32))
    scale = push_data if push_data is not None else np.ones(4, dtype=np.float32)
    count = [0]

    def execute(ctx):
        if record_extra:
            record_extra(ctx)
        ctx.push_constants(lr.Stage.FRAGMENT, scale)
        ctx.cmd.draw(3)

    (
        fg.add_pass("tinted")
        .type(lr.PassType.FULLSCREEN)
        .vert_shader(lr.compile_glsl_source(VERT, lr.ShaderStage.VERTEX, "fullscreen.vert"))
        .frag_shader(lr.compile_glsl_source(frag_source, lr.ShaderStage.FRAGMENT, "tinted.frag"))
        .uniform_buffer(tint_binding, fg.buffer("tint"), lr.Stage.FRAGMENT)
        .push_constant_size(push_size, lr.Stage.FRAGMENT)
        .color_attachment(fg.image(lr.SWAPCHAIN), viewer.swapchain_format)
        .execute(execute)
    )

    def update(dt, extent):
        count[0] += 1
        if count[0] >= frames:
            viewer.close()

    update_connection = viewer.on_update(update)
    viewer.run()
    return count[0]


def expect(title, exception_type, needles, scenario):
    print(f"\n== {title} ==")
    try:
        scenario()
    except exception_type as e:
        message = str(e)
        print(f"{type(e).__module__}.{type(e).__name__}: {message}")
        missing = [n for n in needles if n not in message]
        assert not missing, f"message is missing {missing}"
    else:
        raise AssertionError(f"expected {exception_type.__name__}")
    finally:
        gc.collect()  # destroy this scenario's Viewer before the next one opens a window


def main():
    expect(
        "1. GLSL syntax error",
        lr.ShaderCompileError,
        ["tinted.frag:6", "colr"],
        lambda: run_pass(frag_source=FRAG.replace("outColor = tint.color", "outColor = tint.colr")),
    )
    expect(
        "2. uniform buffer declared at binding 1, shader reads binding 0",
        lr.ShaderInterfaceError,
        ["pass 'tinted'", "binding 0 ('tint', uniform buffer, fragment)", "not declared", "declares bindings no shader uses: 1"],
        lambda: run_pass(tint_binding=1),
    )
    expect(
        "3. push_constant_size smaller than the shader's block",
        lr.ShaderInterfaceError,
        ["pass 'tinted'", "the shaders use 16 bytes, but push_constant_size is 8"],
        lambda: run_pass(push_size=8),
    )
    expect(
        "4. pushing 32 bytes into a 16-byte push-constant range",
        ValueError,
        ["writes bytes [0, 32)", "push_constant_size(16)"],
        lambda: run_pass(push_data=np.ones(8, dtype=np.float32)),
    )
    expect(
        "5. zero-width viewport (caught by the Vulkan validation layer)",
        lr.VulkanValidationError,
        ["Vulkan validation error", "width"],
        lambda: run_pass(record_extra=lambda ctx: ctx.cmd.set_viewport(0, 0, 0, 0)),
    )

    print("\n== 6. no mistakes ==")
    frames = run_pass(frames=30)
    assert frames == 30
    print(f"rendered {frames} frames; every error above was recoverable")


if __name__ == "__main__":
    main()
