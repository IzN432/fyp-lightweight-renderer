"""Scenes loaded with the engine's loaders, read from Python. CPU only: no window or GPU needed.

Run via ctest (`python.scene`), or directly:
    PYTHONPATH=build/python python tests/python/test_scene.py
"""

import pathlib
import sys
import tempfile

import numpy as np

import lr
from lr import engine

BIRD = pathlib.Path(lr.ASSET_DIR) / "samples" / "models" / "bird_orange.glb"
CONVENTION_PARAMETERS = {"baseDiffuse", "baseEmissive", "baseRoughness", "baseMetallic"}


def mesh_objects(scene):
    return [obj for obj in scene.objects if obj.mesh is not None]


def test_gltf_meshes_are_render_ready():
    scene = engine.load_scene(BIRD)
    (obj,) = mesh_objects(scene)
    mesh = obj.mesh
    assert mesh.attribute_names == ["normal", "tangent", "uv"] or set(mesh.attribute_names) >= {"normal", "uv"}
    v = mesh.vertex_count
    assert mesh.positions.shape == (v, 3) and mesh.positions.dtype == np.float32
    assert mesh.attribute("normal").shape == (v, 3)
    assert mesh.attribute("uv").shape == (v, 2)
    assert mesh.attribute("tangent").shape == (v, 4)
    assert mesh.indices.shape == (mesh.face_count, 3) and mesh.indices.dtype == np.uint32
    assert mesh.indices.max() < v
    assert np.allclose(np.linalg.norm(mesh.attribute("normal"), axis=1), 1.0, atol=1e-3)

    # positions are the unique positions gathered through position_indices
    assert np.array_equal(mesh.positions, mesh.unique_positions[mesh.position_indices])

    try:
        mesh.attribute("colour")
    except KeyError as e:
        assert "has: " in str(e), str(e)
    else:
        raise AssertionError("expected KeyError for a missing attribute")


def test_materials_use_engine_conventions():
    scene = engine.load_scene(BIRD)
    (obj,) = mesh_objects(scene)
    handles = set(obj.mesh.face_materials.tolist())
    assert len(obj.mesh.face_materials) == obj.mesh.face_count
    for handle in handles:
        material = scene.material(handle)
        assert CONVENTION_PARAMETERS <= set(material.parameters), material.parameters
        assert "baseColorTexture" in material.texture_names
        texture = material.texture("baseColorTexture")
        assert texture.ndim == 3 and texture.shape[2] == 4 and texture.dtype == np.uint8
    try:
        scene.material(10_000)
    except IndexError:
        pass
    else:
        raise AssertionError("expected IndexError for an out-of-range handle")


def test_hierarchy_and_transforms():
    scene = engine.load_scene(BIRD)
    (root,) = scene.roots
    assert root.parent is None and root.children
    for obj in scene.objects:
        for child in obj.children:
            assert child.parent is not None and child.parent.id == obj.id
    found = scene.find(root.name)
    assert found is not None and found.id == root.id and scene.find("no such object") is None

    # Moving the root moves every descendant's world matrix with it.
    (obj,) = mesh_objects(scene)
    before = obj.world_matrix.copy()
    root.position = (10.0, 0.0, 0.0)
    assert np.allclose(obj.world_matrix[:3, 3] - before[:3, 3], [10.0, 0.0, 0.0], atol=1e-5)
    assert np.allclose(obj.world_matrix[:3, :3], before[:3, :3], atol=1e-6)
    root.rotation = (0.0, 0.70710678, 0.0, 0.70710678)  # 90 degrees about +Y
    assert np.allclose(np.asarray(root.rotation), [0.0, 0.70710678, 0.0, 0.70710678], atol=1e-6)


def test_animation_moves_joints():
    scene = engine.load_scene(BIRD)
    animators = [obj.animator for obj in scene.objects if obj.animator is not None]
    assert animators and animators[0].clip_names
    animators[0].play(0)
    joints = [obj for obj in scene.objects if obj.mesh is None and obj.parent is not None]
    before = [obj.world_matrix.copy() for obj in joints]
    scene.update(0.25)
    moved = sum(not np.allclose(b, obj.world_matrix, atol=1e-6) for b, obj in zip(before, joints))
    assert moved > 0, "playing the clip should move some joints"


def test_multi_material_obj():
    with tempfile.TemporaryDirectory() as tmp:
        folder = pathlib.Path(tmp)
        (folder / "two.mtl").write_text("newmtl red\nKd 1 0 0\nnewmtl green\nKd 0 1 0\n")
        (folder / "two.obj").write_text(
            "mtllib two.mtl\n"
            "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 2 0 0\nv 2 1 0\n"
            "usemtl red\nf 1 2 3\nf 1 3 4\n"
            "usemtl green\nf 2 5 6\n"
        )
        scene = engine.load_scene(folder / "two.obj")

    (obj,) = mesh_objects(scene)
    face_materials = obj.mesh.face_materials
    assert obj.mesh.face_count == 3 and len(set(face_materials.tolist())) == 2, face_materials
    colors = {}
    for handle in face_materials:
        material = scene.material(int(handle))
        color = material.parameters["baseDiffuse"]
        assert isinstance(color, tuple), color
        colors[material.name] = color[:3]
    assert np.allclose(colors["red"], [1, 0, 0]) and np.allclose(colors["green"], [0, 1, 0]), colors
    # Faces keep their material: the last face is the green one.
    assert scene.material(int(face_materials[2])).name == "green"


def test_scene_without_viewer_and_multiple_loads():
    scene = engine.Scene()
    first = scene.load(BIRD)
    second = scene.load(BIRD)
    assert first.id != second.id and len(scene.roots) == 2
    assert len(mesh_objects(scene)) == 2


def test_add_light():
    scene = engine.Scene()
    spot = scene.add_light("spot", color=(1, 0.5, 0), intensity=3.0, position=(1, 2, 3),
                           rotation=(0.0, 0.70710678, 0.0, 0.70710678), outer_cone_degrees=40.0, name="Spot")
    assert spot.name == "Spot" and spot.parent is None and spot.mesh is None
    assert np.allclose(spot.position, (1, 2, 3)) and np.allclose(spot.rotation, (0.0, 0.70710678, 0.0, 0.70710678))
    light = spot.light
    assert light is not None and light.type == "spot" and light.intensity == 3.0
    assert np.allclose(light.color, (1, 0.5, 0)) and light.outer_cone_degrees == 40.0
    area = scene.add_light("area", size=(2.0, 0.5)).light
    assert area is not None and area.type == "area" and area.area_size == (2.0, 0.5)
    image = scene.add_light("image", intensity=0.5).light
    assert image is not None and image.type == "image" and image.intensity == 0.5
    assert len(scene.objects) == 3


def test_engine_layer_is_its_own_submodule():
    # The opinionated pieces live in lr.engine; lr itself is the general-purpose frame graph.
    for name in engine.__all__:
        assert hasattr(engine, name) and not hasattr(lr, name), name
    for name in ("Viewer", "FrameGraph", "PassBuilder", "ResourceRegistry", "OrbitCamera"):
        assert hasattr(lr, name) and not hasattr(engine, name), name


def main():
    tests = [
        test_engine_layer_is_its_own_submodule,
        test_add_light,
        test_gltf_meshes_are_render_ready,
        test_materials_use_engine_conventions,
        test_hierarchy_and_transforms,
        test_animation_moves_joints,
        test_multi_material_obj,
        test_scene_without_viewer_and_multiple_loads,
    ]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"ok   {test.__name__}")
        except Exception as e:  # report every failing test, not just the first
            failures += 1
            print(f"FAIL {test.__name__}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failures} passed, {failures} failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
