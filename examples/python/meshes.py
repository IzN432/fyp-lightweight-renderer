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
