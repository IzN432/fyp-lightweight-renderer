"""Contract: custom compute shader, semantic usages, and buffer readback."""

from pathlib import Path

import numpy as np

import lightweight_renderer as lr


HERE = Path(__file__).resolve().parent
renderer = lr.Renderer()

values = np.arange(1024, dtype=np.float32)
input_buffer = renderer.create_buffer(
    data=values,
    usage=lr.BufferUsage.STORAGE,
    label="compute input",
)
output_buffer = renderer.create_buffer(
    shape=values.shape,
    dtype=values.dtype,
    usage=lr.BufferUsage.STORAGE | lr.BufferUsage.READBACK,
    label="compute output",
)

shader = renderer.load_shader(
    HERE / "shaders" / "double.comp",
    stage=lr.ShaderStage.COMPUTE,
)
pipeline = renderer.create_compute_pipeline(
    shader=shader,
    bindings={0: lr.storage_buffer(read_only=True), 1: lr.storage_buffer()},
    label="double values",
)

graph = lr.FrameGraphBuilder(renderer)


def encode(encoder: lr.CommandEncoder) -> None:
    encoder.bind_pipeline(pipeline)
    encoder.bind_buffer(0, input_buffer)
    encoder.bind_buffer(1, output_buffer)
    encoder.dispatch_for(output_buffer, local_size=(64, 1, 1))


graph.add_compute_pass(
    "double",
    reads=[lr.storage_read(input_buffer)],
    writes=[lr.storage_write(output_buffer)],
    execute=encode,
)

compiled = graph.compile()
renderer.execute_and_wait(compiled)

result = output_buffer.readback(dtype=np.float32, shape=values.shape)
np.testing.assert_allclose(result, values * 2.0)
