"""Deferred renderer from the engine's building blocks, with a pass of our own in the middle.

  lr.SceneGpu              the scene's GPU buffers in the engine's layout (camera, lights, meshes, materials)
  lr.Ibl                   image-based lighting precomputed from an HDR environment
  lr.GeometryPass          -> gbufferAlbedo, gbufferNormal, gbufferMaterial, gbufferEmissive, gbufferDepth
  lr.AmbientOcclusionPass  gbufferDepth -> hbao_ao
  lr.PbrPass               G-buffer + hbao_ao + lights + IBL -> pbr (HDR)
  fog (this file)          pbr + gbufferDepth -> fogged (HDR)
  lr.CompositePass         fogged -> swapchain: Reinhard tone map, the environment as the sky

Each block's inputs and outputs are printed at startup (and documented in docs/python_building_blocks.md).
Middle-drag orbits, Shift + middle-drag pans, the scroll wheel zooms.

Run from the repo root after building:
    PYTHONPATH=build/python python examples/python/deferred_blocks.py [path/to/model.glb] [--frames N]
"""

import argparse
import pathlib
from typing import Any

import numpy as np

import lr

SHADERS = pathlib.Path(__file__).parent / "shaders"
ASSETS = pathlib.Path(lr.ASSET_DIR) / "samples"
DEFAULT_MODEL = ASSETS / "models" / "bird_orange.glb"
DEFAULT_HDRI = ASSETS / "environments" / "cedar_bridge_sunset_2_4k.hdr"


def mesh_bounds(scene: lr.Scene) -> tuple[np.ndarray, np.ndarray]:
    lows, highs = [], []
    for obj in scene.objects:
        if obj.mesh is not None and obj.mesh.vertex_count:
            world = obj.mesh.positions @ obj.world_matrix[:3, :3].T + obj.world_matrix[:3, 3]
            lows.append(world.min(axis=0))
            highs.append(world.max(axis=0))
    return np.min(lows, axis=0), np.max(highs, axis=0)


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("model", nargs="?", default=str(DEFAULT_MODEL), help="OBJ, glTF or GLB file")
    parser.add_argument("--hdri", default=str(DEFAULT_HDRI), help="equirectangular .hdr environment")
    parser.add_argument("--frames", type=int, default=0, help="close after this many frames (0 = run until closed)")
    args = parser.parse_args()

    scene = lr.load_scene(args.model)
    low, high = mesh_bounds(scene)  # before SceneGpu adds the light quads
    size = float(np.linalg.norm(high - low))
    scene.add_light("directional", intensity=2.0, rotation=(-0.3826834, 0.0, 0.0, 0.9238795))  # 45 deg down
    scene.add_light("image", intensity=1.0)  # ambient light from the Ibl's environment

    viewer = lr.Viewer(title="lr - deferred renderer from building blocks (Python)", width=1280, height=720)
    fg = viewer.frame_graph

    camera = lr.OrbitCamera(viewer)
    camera.target = tuple(((low + high) / 2).tolist())
    camera.radius = size * 1.2
    camera.elevation = 0.3

    # Small cubemaps keep startup quick; the engine's own renderer uses 2048.
    ibl = lr.Ibl(viewer, hdri=args.hdri, env_res=512, pf_res=256, pf_mips=6)
    gpu = lr.SceneGpu(viewer, scene, camera)
    geometry = lr.GeometryPass(viewer, gpu)
    ao = lr.AmbientOcclusionPass(viewer, gpu, sphere_radius=size * 0.02)
    pbr = lr.PbrPass(viewer, gpu, ibl)

    # Our own pass: reads two engine outputs, writes an image the composite reads instead of "pbr".
    fog = np.array([0.55, 0.6, 0.7, 0.0, 0.0], dtype=np.float32)  # vec4 colour, float density
    fog_density = {"value": 0.15 / size}

    def draw_fog(ctx):
        ctx.push_constants(lr.Stage.FRAGMENT, fog)
        ctx.cmd.draw(3)

    (
        fg.add_pass("fog")
        .type(lr.PassType.FULLSCREEN)
        .vert_shader(pathlib.Path(lr.SHADER_DIR) / "fullscreen.vert.spv")
        .frag_shader(SHADERS / "fog.frag")
        .uniform_buffer(0, fg.buffer(gpu.camera_buffer), lr.Stage.FRAGMENT)
        .sampled_image(1, fg.image("pbr"), lr.Stage.FRAGMENT)
        .sampled_depth(2, fg.image("gbufferDepth"), lr.Stage.FRAGMENT)
        .push_constant_size(fog.nbytes, lr.Stage.FRAGMENT)
        .color_attachment(fg.image("fogged"), lr.Format.R16G16B16A16_SFLOAT)
        .execute(draw_fog)
    )
    composite = lr.CompositePass(viewer, gpu, input="fogged")

    blocks = [ibl, geometry, ao, pbr, composite]
    for block in blocks:
        print(block.describe())

    # The contracts the blocks were wired up by.
    def names(uses):
        return {use.name for use in uses}

    assert {"gbufferAlbedo", "gbufferNormal", "gbufferMaterial", "gbufferEmissive", "gbufferDepth"} <= names(
        geometry.outputs
    ), geometry
    assert names(ao.outputs) == {"hbao_ao"} and "gbufferDepth" in names(ao.inputs), ao
    assert names(pbr.outputs) == {"pbr"} and {"hbao_ao", "gbufferDepth", "ibl_irradiance"} <= names(pbr.inputs), pbr
    assert names(composite.inputs) >= {"fogged", "gbufferDepth", "ibl_env"}, composite
    assert {"ibl_env", "ibl_irradiance", "ibl_prefiltered"} <= names(ibl.outputs), ibl

    state: dict[str, Any] = {"frames": 0, "fog": True}

    def gui():
        with lr.gui.window("Building blocks", size=(340, 190), position=(20, 20)):
            _, state["fog"] = lr.gui.checkbox("Fog (Python pass)", state["fog"])
            _, fog_density["value"] = lr.gui.slider_float("Fog density", fog_density["value"], 0.0, 2.0 / size)
            changed, radius = lr.gui.slider_float("AO radius", ao.sphere_radius, 0.0, size * 0.1)
            if changed:
                ao.sphere_radius = radius
            lr.gui.text(f"{gpu.mesh_count} meshes, {gpu.num_lights} lights")
            lr.gui.text(f"{lr.gui.framerate():.0f} fps")

    def update(dt, extent):
        camera.update(dt)
        fog[4] = fog_density["value"] if state["fog"] else 0.0
        state["frames"] += 1
        if args.frames and state["frames"] >= args.frames:
            viewer.close()

    viewer.on_update(update)
    viewer.on_gui(gui)
    viewer.run()
    assert fg.compile_count >= 1


if __name__ == "__main__":
    main()
