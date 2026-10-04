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
from lr import engine

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


def probe_pass(viewer):
    """Adds a compute pass writing the centre pixel of "pbr" and "gbufferDepth" to the "probe" buffer."""
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


@dataclasses.dataclass
class Chain:
    gpu_scene: engine.Scene
    viewer: lr.Viewer
    camera: lr.OrbitCamera
    ibl: engine.Ibl
    gpu: engine.SceneGpu
    geometry: engine.GeometryPass
    ao: engine.AmbientOcclusionPass
    pbr: engine.PbrPass
    composite: engine.CompositePass


def build(extra=None) -> Chain:
    """A viewer with the full engine chain over the bird. `extra(viewer)` adds passes before the composite."""
    scene = engine.load_scene(BIRD)
    scene.add_light("directional", intensity=2.0, rotation=(-0.3826834, 0.0, 0.0, 0.9238795))
    scene.add_light("image")
    viewer = lr.Viewer(title="lr building blocks test", width=320, height=240)
    camera = lr.OrbitCamera(viewer)
    camera.target, camera.radius, camera.elevation = (0.0, 0.5, 0.0), 2.5, 0.2
    ibl = engine.Ibl(viewer, env_res=64, irr_res=16, pf_res=64, pf_mips=4)
    gpu = engine.SceneGpu(viewer, scene, camera)
    geometry = engine.GeometryPass(viewer, gpu)
    ao = engine.AmbientOcclusionPass(viewer, gpu, sphere_radius=0.05)
    pbr = engine.PbrPass(viewer, gpu, ibl)
    if extra:
        extra(viewer)
    composite = engine.CompositePass(viewer, gpu)
    return Chain(scene, viewer, camera, ibl, gpu, geometry, ao, pbr, composite)


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
    c = build(probe_pass)
    run_frames(c.viewer, 5)
    color_and_depth = c.viewer.resources.read_buffer("probe").view(np.float32)
    color, depth = color_and_depth[:4], color_and_depth[4]
    assert 0.0 < depth < 1.0, f"the bird should cover the centre (depth {depth})"
    assert np.all(np.isfinite(color)) and color[:3].max() > 0.01, f"the bird should be lit: {color}"


def test_lights_added_and_removed_while_running_light_the_scene():
    # PbrPass must see the new light count (it used to keep the count it was built with), and the
    # geometry must stay consistent as light quads come and go.
    c = build(probe_pass)
    scene = c.gpu_scene
    samples, lights = {}, {}

    def probe():
        return c.viewer.resources.read_buffer("probe").view(np.float32)[:3].copy()

    def each(frame):
        if frame == 3:
            samples["before"] = probe()
            lights["red"] = scene.add_light("point", color=(1.0, 0.0, 0.0), intensity=40.0, position=(0.0, 0.6, 1.0))
        elif frame == 6:
            samples["added"] = probe()
            scene.remove(lights["red"])
        elif frame == 9:
            samples["removed"] = probe()

    run_frames(c.viewer, 9, each)
    before, added, removed = samples["before"], samples["added"], samples["removed"]
    assert added[0] > before[0] + 0.05 and added[0] - before[0] > 2 * abs(added[2] - before[2]), (before, added)
    assert np.allclose(removed, before, atol=1e-3), (before, removed)
    assert c.gpu.num_lights == 2


PROBE_POINTS = """
#version 450
layout(local_size_x = 2) in;
layout(set = 0, binding = 0) uniform sampler2D lit;
layout(set = 0, binding = 1) buffer Probe { vec4 color[2]; } probe;
layout(push_constant) uniform Points { vec2 uv[2]; } points;
void main() { probe.color[gl_LocalInvocationID.x] = texture(lit, points.uv[gl_LocalInvocationID.x]); }
"""


def test_area_lights_face_forward_or_both_ways():
    # An area light between the camera (+Z side) and the bird. Aimed at the bird (its forward, -Z, points
    # there), a one-sided light lights the bird and shows the camera its back, so its quad is culled.
    # Two-sided, the camera sees the quad too. Turned around, a one-sided light no longer lights the bird
    # and the camera sees its emitting face. (Both used to be reversed.)
    light_position = np.array([0.0, 1.0, 1.2])
    scene = engine.load_scene(BIRD)  # no other lights; the environment is black
    light = scene.add_light("area", intensity=10.0, size=(0.6, 0.6), position=tuple(light_position),
                            two_sided=False)
    viewer = lr.Viewer(title="lr building blocks test", width=320, height=240)
    camera = lr.OrbitCamera(viewer)
    camera.target, camera.radius, camera.elevation, camera.azimuth = (0.0, 0.5, 0.0), 4.0, 0.15, 0.0
    gpu = engine.SceneGpu(viewer, scene, camera)
    engine.GeometryPass(viewer, gpu)
    engine.AmbientOcclusionPass(viewer, gpu, sphere_radius=0.05)
    engine.PbrPass(viewer, gpu, engine.Ibl(viewer, env_res=64, irr_res=16, pf_res=64, pf_mips=4))
    fg, res = viewer.frame_graph, viewer.resources
    res.register_static_buffer("probe", 32, lr.BufferUsage.STORAGE)
    uv = np.zeros(4, np.float32)  # bird centre, light centre

    def probe(ctx):
        ctx.push_constants(lr.Stage.COMPUTE, uv)
        ctx.cmd.dispatch(1)

    (
        fg.add_pass("probe")
        .type(lr.PassType.COMPUTE)
        .compute_shader(lr.compile_glsl_source(PROBE_POINTS, lr.ShaderStage.COMPUTE, "probe_points.comp"))
        .sampled_image(0, fg.image("pbr"), lr.Stage.COMPUTE)
        .storage_buffer_write(1, fg.buffer("probe"), lr.Stage.COMPUTE)
        .push_constant_size(uv.nbytes, lr.Stage.COMPUTE)
        .execute(probe)
    )
    engine.CompositePass(viewer, gpu)
    samples = {}

    def sample():
        bird, quad = res.read_buffer("probe").view(np.float32).reshape(2, 4)[:, :3]
        return float(bird.max()), float(quad.min())

    def each(frame):
        view, proj = camera.view_matrix(), camera.projection_matrix(320 / 240)
        for i, point in enumerate([np.array([0.0, 0.45, 0.0]), light_position]):
            clip = proj @ view @ np.array([*point, 1.0])
            uv[2 * i: 2 * i + 2] = clip[:2] / clip[3] * 0.5 + 0.5
        if frame == 4:
            samples["one-sided, aimed at the bird"] = sample()
            light.set_light(two_sided=True)
        elif frame == 7:
            samples["two-sided, aimed at the bird"] = sample()
            light.set_light(two_sided=False)
            light.rotation = (0.0, 1.0, 0.0, 0.0)  # turned to face the camera
        elif frame == 10:
            samples["one-sided, aimed away"] = sample()

    run_frames(viewer, 10, each)
    lit_bird, hidden_quad = samples["one-sided, aimed at the bird"]
    two_sided_bird, two_sided_quad = samples["two-sided, aimed at the bird"]
    unlit_bird, facing_quad = samples["one-sided, aimed away"]
    assert lit_bird > unlit_bird + 0.05, samples  # one-sided lights shine forward only
    assert hidden_quad < 5.0 and two_sided_quad > 9.0 and facing_quad > 9.0, samples  # quad pixel: 10 when seen
    assert abs(two_sided_bird - lit_bird) < 1e-3, samples


def test_lights_added_before_the_first_frame():
    c = build()
    c.gpu_scene.add_light("spot", position=(0.0, 1.0, 1.0))  # picked up in frame 1, before anything has run
    run_frames(c.viewer, 3)
    assert c.gpu.num_lights == 3


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
        engine.SceneGpu(viewer, engine.Scene(), camera)
    except ValueError as e:
        assert "no meshes" in str(e), e
    else:
        raise AssertionError("expected ValueError for an empty scene")

    scene = engine.load_scene(BIRD)
    engine.SceneGpu(viewer, scene, camera)
    try:
        engine.SceneGpu(viewer, scene, camera)
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


def test_a_scene_can_be_shown_again():
    # SceneGpu draws each light as a quad it owns; the scene itself must come out unchanged.
    scene = engine.load_scene(BIRD)
    light = scene.add_light("area", size=(0.5, 0.5), position=(0.0, 1.5, 1.0))
    objects_before = [obj.id for obj in scene.objects]
    counts = []
    for _ in range(2):
        viewer = lr.Viewer(title="lr building blocks test", width=320, height=240)
        camera = lr.OrbitCamera(viewer)
        gpu = engine.SceneGpu(viewer, scene, camera)
        engine.GeometryPass(viewer, gpu)
        counts.append((gpu.mesh_count, gpu.num_lights))
        run_frames(viewer, 2)
        assert light.mesh is None, "the light's quad must not become a scene mesh"
        del viewer, camera, gpu
    assert counts == [(1, 1), (1, 1)], counts
    assert [obj.id for obj in scene.objects] == objects_before


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
        test_lights_added_and_removed_while_running_light_the_scene,
        test_lights_added_before_the_first_frame,
        test_area_lights_face_forward_or_both_ways,
        test_camera_buffer_follows_the_camera,
        test_ao_parameters_change_while_running,
        test_misuse_is_reported,
        test_a_scene_can_be_shown_again,
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
