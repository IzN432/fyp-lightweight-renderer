"""The engine's preassembled renderer, as building blocks.

Everything else in ``lr`` is the general-purpose frame graph: buffers, images, passes and shaders you write
yourself, with no opinion on what a scene is or how it is lit. ``lr.engine`` is the engine's own, opinionated
layer on top of it — the same pieces the C++ renderer is made of:

  Scenes          load_scene, Scene, SceneObject, Mesh, Material, Light, Animator
                  the engine's scene model and loaders (OBJ, glTF, GLB), on the CPU
  GPU layout      SceneGpu
                  a scene uploaded in the engine's buffer layout and kept in sync every frame
  Passes          Ibl, GeometryPass, AmbientOcclusionPass, PbrPass, CompositePass
                  image-based lighting, the G-buffer, HBAO, PBR lighting and tone mapping, connected by
                  named frame-graph resources (see EnginePass.inputs/outputs and
                  docs/python_building_blocks.md)

The two layers mix freely: engine passes run in the same frame graph as yours, so your passes can read
their outputs and feed their inputs.

    import lr
    from lr import engine

    scene = engine.load_scene("model.glb")
    viewer = lr.Viewer()
    camera = lr.OrbitCamera(viewer)
    gpu = engine.SceneGpu(viewer, scene, camera)
    engine.GeometryPass(viewer, gpu)
    ...
"""

from ._lr import engine as _native

# Scenes (CPU side)
Scene = _native.Scene
SceneObject = _native.SceneObject
Mesh = _native.Mesh
Material = _native.Material
Light = _native.Light
Animator = _native.Animator
load_scene = _native.load_scene

# GPU layout
SceneGpu = _native.SceneGpu

# Passes
EnginePass = _native.EnginePass
ResourceUse = _native.ResourceUse
Ibl = _native.Ibl
GeometryPass = _native.GeometryPass
AmbientOcclusionPass = _native.AmbientOcclusionPass
PbrPass = _native.PbrPass
CompositePass = _native.CompositePass

__all__ = [
    "Scene", "SceneObject", "Mesh", "Material", "Light", "Animator", "load_scene",
    "SceneGpu",
    "EnginePass", "ResourceUse", "Ibl", "GeometryPass", "AmbientOcclusionPass", "PbrPass", "CompositePass",
]

# tests/python/test_stubs.py checks that every name the native submodule exports is listed above. (The
# classes keep their native __module__, lr._lr.engine, like lr's own classes keep lr._lr: renaming it
# would hide them from nanobind's stubgen, which the stub check is built on.)
