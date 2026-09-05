"""Contract: generic NumPy upload, update, and readback without a Mesh."""

import numpy as np

import lightweight_renderer as lr


renderer = lr.Renderer()

points = np.array(
    [
        [-0.75, -0.50, 0.0],
        [0.75, -0.50, 0.0],
        [0.00, 0.75, 0.0],
    ],
    dtype=np.float32,
)

point_buffer = renderer.create_buffer(
    data=points,
    usage=lr.BufferUsage.STORAGE | lr.BufferUsage.VERTEX,
    label="triangle points",
)

# Updates accept any contiguous Python buffer. The renderer retains no borrowed
# pointer after this call returns.
points[:, 0] *= 0.5
point_buffer.update(points)

# Readback returns an owned NumPy array. It remains valid after the GPU buffer is
# released and preserves the requested logical shape and dtype.
result = point_buffer.readback(dtype=np.float32, shape=points.shape)
np.testing.assert_allclose(result, points)

renderer.wait_idle()
