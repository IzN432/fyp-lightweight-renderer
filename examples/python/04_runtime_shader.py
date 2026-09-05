"""Contract: runtime shader files and useful shader compilation failures."""

from pathlib import Path

import lightweight_renderer as lr


HERE = Path(__file__).resolve().parent
renderer = lr.Renderer()

try:
    shader = renderer.load_shader(
        HERE / "shaders" / "display.frag",
        stage=lr.ShaderStage.FRAGMENT,
        entry_point="main",
        defines={"APPLY_GAMMA": "1"},
        include_directories=[HERE / "shaders"],
    )
except lr.ShaderCompilationError as error:
    # The exception contract includes the source path, stage and compiler
    # diagnostics with original source line numbers.
    print(error.path)
    print(error.stage)
    print(error.diagnostics)
    raise

print(shader.label, shader.entry_point)

# Loading is explicit. A shader object is immutable; applications request a new
# shader and pipeline when implementing hot reload.
reloaded_shader = renderer.load_shader(
    HERE / "shaders" / "display.frag",
    stage=lr.ShaderStage.FRAGMENT,
)
assert reloaded_shader is not shader
