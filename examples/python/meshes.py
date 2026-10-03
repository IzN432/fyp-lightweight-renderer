"""Procedural meshes shared by the Python examples."""

import numpy as np


def make_torus(major_radius=1.0, minor_radius=0.4, rings=96, sides=48):
    """Interleaved (position, normal) float32 vertices and a flat uint32 triangle index array."""
    u = np.linspace(0.0, 2.0 * np.pi, rings, endpoint=False)
    v = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing="ij")

    normals = np.stack([np.cos(vv) * np.cos(uu), np.sin(vv), np.cos(vv) * np.sin(uu)], axis=-1)
    centres = np.stack([major_radius * np.cos(uu), np.zeros_like(uu), major_radius * np.sin(uu)], axis=-1)
    positions = centres + minor_radius * normals
    vertices = np.concatenate([positions, normals], axis=-1).reshape(-1, 6).astype(np.float32)

    i, j = np.meshgrid(np.arange(rings), np.arange(sides), indexing="ij")
    a = i * sides + j
    b = ((i + 1) % rings) * sides + j
    c = ((i + 1) % rings) * sides + (j + 1) % sides
    d = i * sides + (j + 1) % sides
    # Counter-clockwise seen from outside, the engine's front face.
    triangles = np.stack([a, d, c, a, c, b], axis=-1).reshape(-1)
    return vertices, triangles.astype(np.uint32)


def make_cube(half_size=0.5):
    """A unit cube with flat per-face normals: 24 interleaved (position, normal) float32 vertices, 36 indices."""
    vertices: list[np.ndarray] = []
    indices: list[int] = []
    for axis in range(3):
        for sign in (1.0, -1.0):
            normal = np.zeros(3)
            normal[axis] = sign
            # Two tangent axes chosen so (u, v, normal) is right-handed, giving counter-clockwise
            # winding seen from outside.
            u = np.roll(normal, 1) if sign > 0 else np.roll(normal, 2)
            v = np.cross(normal, u)
            base = len(vertices)
            for du, dv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                vertices.append(np.concatenate([(normal + du * u + dv * v) * half_size, normal]))
            indices += [base, base + 1, base + 2, base, base + 2, base + 3]
    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)
