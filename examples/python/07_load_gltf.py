"""Contract: the optional glTF loader returns generic NumPy-backed data."""

from pathlib import Path

import numpy as np

import lightweight_renderer as lr


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPOSITORY_ROOT / "assets" / "samples" / "models" / "lion_head_4k.glb"

# Loading is a CPU-side operation. It does not require a Renderer, create Vulkan
# resources, or insert objects into an engine-owned scene.
asset = lr.load_gltf(MODEL_PATH)

print(f"Loaded {len(asset.meshes)} mesh primitives")
print(f"Loaded {len(asset.materials)} materials")

mesh = asset.meshes[0]

# Attributes use stable semantic names and ordinary NumPy arrays. Optional glTF
# attributes are absent from the dictionary rather than filled with magic values.
positions = mesh.attributes["position"]
normals = mesh.attributes.get("normal")
texcoords = mesh.attributes.get("texcoord_0")
indices = mesh.indices

assert isinstance(positions, np.ndarray)
assert positions.dtype == np.float32
assert positions.ndim == 2 and positions.shape[1] == 3
assert isinstance(indices, np.ndarray)
assert indices.dtype in (np.uint16, np.uint32)

# Researchers can inspect, replace, or generate attributes before uploading them.
centered_positions = positions - positions.mean(axis=0, keepdims=True)

renderer = lr.Renderer()
position_buffer = renderer.create_buffer(
    centered_positions,
    usage=lr.BufferUsage.VERTEX | lr.BufferUsage.STORAGE,
    label=f"{mesh.name} positions",
)
index_buffer = renderer.create_buffer(
    indices,
    usage=lr.BufferUsage.INDEX,
    label=f"{mesh.name} indices",
)

print(position_buffer)
print(index_buffer)

if normals is not None:
    print("normal attribute:", normals.shape, normals.dtype)
if texcoords is not None:
    print("texture coordinates:", texcoords.shape, texcoords.dtype)

# Material and texture descriptions remain CPU-side until the application chooses
# how to upload and bind them. Mesh primitives reference them by optional index.
if mesh.material_index is not None:
    material = asset.materials[mesh.material_index]
    print("material:", material.name, material.parameters)
