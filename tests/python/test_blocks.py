"""The engine's building blocks (SceneGpu, Ibl, GeometryPass, AmbientOcclusionPass, PbrPass,
CompositePass) driven from Python. Needs a GPU and a display (opens short-lived windows).

Run via ctest (`python.blocks`), or directly:
    PYTHONPATH=build/python python tests/python/test_blocks.py
"""

import dataclasses
import pathlib
import sys

import numpy as np

import lr

BIRD = pathlib.Path(lr.ASSET_DIR) / "samples" / "models" / "bird_orange.glb"
F = lr.Format

# Samples the centre of the lit image and of the depth buffer into a storage buffer.
PROBE = """
#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0) uniform sampler2D lit;
layout(set = 0, binding = 1) uniform sampler2D depth;
layout(set = 0, binding = 2) buffer Probe { vec4 color; vec4 depthValue; } probe;
void main()
{
    probe.color = texture(lit, vec2(0.5));
    probe.depthValue = vec4(texture(depth, vec2(0.5)).r);
}
"""


@dataclasses.dataclass
class Chain:
    viewer: lr.Viewer
    camera: lr.OrbitCamera
    ibl: lr.Ibl
    gpu: lr.SceneGpu
    geometry: lr.GeometryPass
    ao: lr.AmbientOcclusionPass
    pbr: lr.PbrPass
    composite: lr.CompositePass


def build(extra=None) -> Chain:
    """A viewer with the full engine chain over the bird. `extra(viewer)` adds passes before the composite."""
    scene = lr.load_scene(BIRD)
    scene.add_light("directional", intensity=2.0, rotation=(-0.3826834, 0.0, 0.0, 0.9238795))
    scene.add_light("image")
    viewer = lr.Viewer(title="lr building blocks test", width=320, height=240)
    camera = lr.OrbitCamera(viewer)
    camera.target, camera.radius, camera.elevation = (0.0, 0.5, 0.0), 2.5, 0.2
    ibl = lr.Ibl(viewer, env_res=64, irr_res=16, pf_res=64, pf_mips=4)
    gpu = lr.SceneGpu(viewer, scene, camera)
    geometry = lr.GeometryPass(viewer, gpu)
    ao = lr.AmbientOcclusionPass(viewer, gpu, sphere_radius=0.05)
    pbr = lr.PbrPass(viewer, gpu, ibl)
    if extra:
        extra(viewer)
    composite = lr.CompositePass(viewer, gpu)
    return Chain(viewer, camera, ibl, gpu, geometry, ao, pbr, composite)


def run_frames(viewer, frames, each=None):
    count = [0]

    def update(dt, extent):
        count[0] += 1
        if each:
            each(count[0])
        if count[0] >= frames:
            viewer.close()

    viewer.on_update(update)
    viewer.run()


def uses(block_uses):
    return {use.name: (use.kind, use.format) for use in block_uses}


def test_contracts_come_from_the_declared_passes():
    c = build()
    geometry = uses(c.geometry.outputs)
    assert geometry == {
        "gbufferAlbedo": ("image", F.R16G16B16A16_SFLOAT),
        "gbufferNormal": ("image", F.R16G16_SFLOAT),
        "gbufferMaterial": ("image", F.R16G16B16A16_UNORM),
        "gbufferEmissive": ("image", F.R16G16B16A16_SFLOAT),
        "gbufferDepth": ("image", F.D32_SFLOAT),
    }, geometry
    gpu_buffers = set(c.gpu.buffers.values())
    geometry_buffers = {u.name for u in c.geometry.inputs if u.kind == "buffer"}
    assert geometry_buffers <= gpu_buffers | {"skinInfluenceEntries", "skinInfluenceOffsets", "skinPositionIndices"}
    assert c.gpu.camera_buffer in geometry_buffers

    # Private data (HBAO params, LTC tables, the HDRI) and intermediates (HBAO's raw AO) don't appear.
    assert uses(c.ao.inputs).keys() == {"gbufferDepth", c.gpu.camera_buffer}, c.ao
    assert uses(c.ao.outputs) == {"hbao_ao": ("image", F.R32_SFLOAT)}, c.ao
    assert uses(c.pbr.outputs) == {"pbr": ("image", F.R16G16B16A16_SFLOAT)}, c.pbr
    assert {"hbao_ao", "ibl_irradiance", "ibl_prefiltered", "ibl_brdf_lut", "gbufferAlbedo", "gbufferNormal",
            "gbufferMaterial", "gbufferEmissive", "gbufferDepth", c.gpu.light_buffer} <= uses(c.pbr.inputs).keys()
    assert not {"ltc1", "ltc2"} & uses(c.pbr.inputs).keys()
    assert uses(c.ibl.inputs) == {}, c.ibl
    assert uses(c.ibl.outputs).keys() == {"ibl_env", "ibl_irradiance", "ibl_prefiltered", "ibl_brdf_lut"}
    assert uses(c.composite.inputs).keys() == {"pbr", "gbufferDepth", "ibl_env", c.gpu.camera_buffer}
    assert list(uses(c.composite.outputs)) == ["swapchain"]

    assert c.geometry.pass_names == ["geometry"] and len(c.geometry.passes) == 1
    assert "hbao_ao" in c.ao.describe() and repr(c.pbr).startswith("PbrPass")
    run_frames(c.viewer, 3)


def test_scene_renders_and_custom_passes_read_engine_outputs():
    def probe(viewer):
        res, fg = viewer.resources, viewer.frame_graph
        res.register_static_buffer("probe", 32, lr.BufferUsage.STORAGE)
        (
            fg.add_pass("probe")
            .type(lr.PassType.COMPUTE)
            .compute_shader(lr.compile_glsl_source(PROBE, lr.ShaderStage.COMPUTE, "probe.comp"))
            .sampled_image(0, fg.image("pbr"), lr.Stage.COMPUTE)
            .sampled_depth(1, fg.image("gbufferDepth"), lr.Stage.COMPUTE)
            .storage_buffer_write(2, fg.buffer("probe"), lr.Stage.COMPUTE)
            .execute(lambda ctx: ctx.cmd.dispatch(1))
        )

    c = build(probe)
    run_frames(c.viewer, 5)
    color_and_depth = c.viewer.resources.read_buffer("probe").view(np.float32)
    color, depth = color_and_depth[:4], color_and_depth[4]
    assert 0.0 < depth < 1.0, f"the bird should cover the centre (depth {depth})"
    assert np.all(np.isfinite(color)) and color[:3].max() > 0.01, f"the bird should be lit: {color}"


def test_camera_buffer_follows_the_camera():
    c = build()
    seen = {}

    def each(frame):
        if frame == 3:
            c.camera.azimuth = 1.0  # SceneGpu re-uploads on its late update, this same frame
        if frame == 5:
            seen["view"] = c.camera.view_matrix()

    run_frames(c.viewer, 5, each)
    ubo = c.viewer.resources.read_buffer(c.gpu.camera_buffer).view(np.float32)
    assert ubo.size == 84, ubo.size  # 5 mat4 + vec4
    view = ubo[:16].reshape(4, 4).T  # column-major on the GPU
    assert np.allclose(view, seen["view"], atol=1e-5), (view, seen["view"])
    assert np.allclose(ubo[80:83], c.camera.position, atol=1e-4)


def test_ao_parameters_change_while_running():
    c = build()
    before = c.viewer.resources.read_buffer("hbao_params").copy()
    c.ao.sphere_radius, c.ao.num_steps = 0.2, 8
    assert c.ao.sphere_radius == np.float32(0.2) and c.ao.num_steps == 8
    after = c.viewer.resources.read_buffer("hbao_params")
    assert not np.array_equal(before, after), "the params buffer should change"
    run_frames(c.viewer, 2)


def test_misuse_is_reported():
    viewer = lr.Viewer(title="lr building blocks test", width=320, height=240)
    camera = lr.OrbitCamera(viewer)
    try:
        lr.SceneGpu(viewer, lr.Scene(), camera)
    except ValueError as e:
        assert "no meshes" in str(e), e
    else:
        raise AssertionError("expected ValueError for an empty scene")

    scene = lr.load_scene(BIRD)
    lr.SceneGpu(viewer, scene, camera)
    try:
        lr.SceneGpu(viewer, scene, camera)
    except RuntimeError as e:
        assert "already" in str(e), e
    else:
        raise AssertionError("expected RuntimeError for a second SceneGpu on one Viewer")

    try:
        scene.add_light("laser")
    except ValueError as e:
        assert "'laser'" in str(e), e
    else:
        raise AssertionError("expected ValueError for an unknown light type")


def test_a_viewer_that_never_ran_is_released_by_the_next():
    # SceneGpu and the passes are held by the Viewer until run() returns; without run() (say an
    # exception first), creating the next Viewer releases and collects the old one.
    c = build()
    del c
    c = build()
    run_frames(c.viewer, 2)

    # One Viewer at a time: one that is still referenced is an error rather than an abort.
    try:
        lr.Viewer(title="lr building blocks test", width=320, height=240)
    except RuntimeError as e:
        assert "one Viewer at a time" in str(e), e
    else:
        raise AssertionError("expected RuntimeError while another Viewer is referenced")


def main():
    tests = [
        test_contracts_come_from_the_declared_passes,
        test_scene_renders_and_custom_passes_read_engine_outputs,
        test_camera_buffer_follows_the_camera,
        test_ao_parameters_change_while_running,
        test_misuse_is_reported,
        test_a_viewer_that_never_ran_is_released_by_the_next,
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
