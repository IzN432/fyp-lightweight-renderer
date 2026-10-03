"""4x4 transform helpers matching the engine's conventions.

Right-handed world space, Vulkan clip space (depth in [0, 1]) and the same Y flip as the engine's
Camera, so counter-clockwise triangles are front-facing. Matrices are row-major numpy float32 arrays
that act on column vectors (``clip = proj @ view @ model @ p``); use ``to_gpu`` to lay one out for a
GLSL ``mat4`` (column-major) in a uniform buffer or push-constant block.
"""

import numpy as np


def perspective(fov_y_degrees, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fov_y_degrees) * 0.5)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = -f  # Vulkan's clip-space Y points down.
    m[2, 2] = far / (near - far)
    m[2, 3] = near * far / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    eye, target, up = (np.asarray(v, dtype=np.float32) for v in (eye, target, up))
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    m = np.identity(4, dtype=np.float32)
    m[0, :3], m[1, :3], m[2, :3] = right, true_up, -forward
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def rotation(axis, angle_radians):
    axis = np.asarray(axis, dtype=np.float32)
    x, y, z = axis / np.linalg.norm(axis)
    c, s = np.cos(angle_radians), np.sin(angle_radians)
    t = 1.0 - c
    m = np.identity(4, dtype=np.float32)
    m[:3, :3] = [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]
    return m


def to_gpu(*matrices):
    """Concatenate matrices in GLSL mat4 (column-major) layout, ready for update_buffer/push_constants."""
    return np.concatenate([np.ascontiguousarray(m.T, dtype=np.float32).ravel() for m in matrices])
