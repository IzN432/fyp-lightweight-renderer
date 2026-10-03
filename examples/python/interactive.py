"""Interactive viewer — the engine's orbit camera driven by the mouse, and an ImGui panel of live controls.

lr.OrbitCamera is the C++ renderer's own camera (SphericalCameraController), so the controls are the same:

  middle-drag: orbit    Shift + middle-drag: pan    scroll: zoom    R: reset

The panel edits the material colour, light direction, spin speed and a wireframe overlay. Colour and light
go into a dynamic uniform buffer that the shader reads; the wireframe toggle just skips that pass's draw.

Run from the repo root after building:
    PYTHONPATH=build/python python examples/python/interactive.py
    PYTHONPATH=build/python python examples/python/interactive.py --scripted   # self-test with injected input
"""

import argparse
import math
import pathlib
from typing import Any

import numpy as np

import lr
from lr import transforms as tf
from meshes import make_torus

SHADERS = pathlib.Path(__file__).parent / "shaders"


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--scripted", action="store_true",
                        help="drive the camera with injected mouse input for 90 frames, check it moved, then exit")
    args = parser.parse_args()

    viewer = lr.Viewer(title="lr - interactive (Python)", width=1280, height=720)
    fg, res = viewer.frame_graph, viewer.resources
    swapchain, depth = fg.image("swapchain"), fg.image("depth")

    vertices, indices = make_torus()
    res.upload_buffer("torus_vertices", vertices, lr.BufferUsage.VERTEX)
    res.upload_buffer("torus_indices", indices, lr.BufferUsage.INDEX)
    res.register_dynamic_buffer("camera", 2 * 64, lr.BufferUsage.UNIFORM)
    res.register_dynamic_buffer("settings", 2 * 16, lr.BufferUsage.UNIFORM)

    camera = lr.OrbitCamera(viewer)
    camera.radius, camera.elevation = 4.0, 0.45
    settings: dict[str, Any] = {
        "albedo": (0.85, 0.55, 0.25),
        "light_yaw": 0.6,
        "light_pitch": 0.9,
        "spin": 0.4,
        "wireframe": True,
        "angle": 0.0,
    }

    def model_matrix():
        return tf.to_gpu(tf.rotation((0, 1, 0), settings["angle"]))

    def draw_solid(ctx):
        ctx.push_constants(lr.Stage.VERTEX, model_matrix())
        ctx.cmd.draw_indexed(len(indices))

    def draw_wire(ctx):
        if settings["wireframe"]:
            ctx.push_constants(lr.Stage.VERTEX, model_matrix())
            ctx.cmd.draw_indexed(len(indices))

    layout = (
        [lr.VertexBinding(0, stride=vertices.strides[0])],
        [lr.VertexAttribute(0, lr.Format.R32G32B32_SFLOAT, 0), lr.VertexAttribute(1, lr.Format.R32G32B32_SFLOAT, 12)],
    )

    def torus_pass(name, frag):
        return (
            fg.add_pass(name)
            .type(lr.PassType.GEOMETRY)
            .vert_shader(SHADERS / "torus.vert")
            .frag_shader(SHADERS / frag)
            .vertex_layout(*layout)
            .vertex_buffer(0, fg.buffer("torus_vertices"))
            .index_buffer(fg.buffer("torus_indices"))
            .uniform_buffer(0, fg.buffer("camera"), lr.Stage.VERTEX)
            .push_constant_size(64, lr.Stage.VERTEX)
        )

    (
        torus_pass("solid", "lit.frag")
        .uniform_buffer(1, fg.buffer("settings"), lr.Stage.FRAGMENT)
        .color_attachment(swapchain, viewer.swapchain_format, clear_color=(0.07, 0.07, 0.09, 1.0))
        .depth_attachment(depth)
        .execute(draw_solid)
    )
    (
        torus_pass("wire", "wire.frag")
        .polygon_mode(lr.PolygonMode.LINE)
        .depth(test=True, write=False, compare=lr.CompareOp.LESS_OR_EQUAL)
        .depth_bias(constant=-1.0, slope=-1.0)
        .color_attachment(swapchain, viewer.swapchain_format, load_op=lr.LoadOp.LOAD)
        .depth_attachment(depth, load_op=lr.LoadOp.LOAD)
        .execute(draw_wire)
    )

    def gui():
        with lr.gui.window("Controls", size=(340, 330), position=(20, 20)):
            lr.gui.text(f"{lr.gui.framerate():.0f} fps, {fg.compile_count} compile(s)")
            lr.gui.text("middle-drag orbit, Shift+middle pan, scroll zoom, R reset")
            lr.gui.separator()
            if lr.gui.collapsing_header("Material"):
                _, settings["albedo"] = lr.gui.color_edit3("Albedo", settings["albedo"])
                _, settings["wireframe"] = lr.gui.checkbox("Wireframe overlay", settings["wireframe"])
            if lr.gui.collapsing_header("Light"):
                _, settings["light_yaw"] = lr.gui.slider_float("Yaw", settings["light_yaw"], -math.pi, math.pi)
                _, settings["light_pitch"] = lr.gui.slider_float("Pitch", settings["light_pitch"], -1.5, 1.5)
            if lr.gui.collapsing_header("Motion"):
                _, settings["spin"] = lr.gui.slider_float("Spin speed", settings["spin"], 0.0, 3.0)
                if lr.gui.button("Frame torus"):
                    camera.target, camera.radius, camera.azimuth, camera.elevation = (0, 0, 0), 4.0, 0.0, 0.45

    script: dict[str, Any] = {"frame": 0, "start": None}

    def inject_script():
        # A slow middle-drag to the right (orbit) over frames 10-40, then a scroll (zoom in) at frame 50.
        from lr._lr import _testing

        frame = script["frame"]
        if frame == 5:
            script["start"] = (camera.azimuth, camera.radius)
        x, y = viewer.input.mouse_position
        if frame == 10:
            _testing.inject_mouse_button(viewer, lr.MouseButton.MIDDLE, True)
        if 10 <= frame < 40:
            _testing.inject_mouse_move(viewer, x + 6.0, y)
        if frame == 40:
            _testing.inject_mouse_button(viewer, lr.MouseButton.MIDDLE, False)
        if frame == 50:
            _testing.inject_scroll(viewer, 3.0)
        script["frame"] += 1
        if script["frame"] >= 90:
            viewer.close()

    def update(dt, extent):
        if args.scripted:
            inject_script()
        camera.update(dt)
        settings["angle"] += settings["spin"] * dt
        res.update_buffer("camera", camera.matrices(extent))

        yaw, pitch = settings["light_yaw"], settings["light_pitch"]
        light = [math.cos(pitch) * math.sin(yaw), math.sin(pitch), math.cos(pitch) * math.cos(yaw), 0.0]
        res.update_buffer("settings", np.array([*settings["albedo"], 1.0, *light], dtype=np.float32))

    viewer.on_update(update)
    # The camera ignores the mouse while it's over UI, so the self-test runs without the panel: the real
    # cursor resting on it would otherwise swallow the injected drag.
    if not args.scripted:
        viewer.on_gui(gui)
    viewer.run()

    if args.scripted:
        start_azimuth, start_radius = script["start"]
        # SphericalCameraController: 0.01 rad per pixel of middle-drag (30 frames x 6 px), and the radius
        # shrinks by 1/1.1 per scroll notch.
        expected_radius = start_radius / 1.1**3
        assert camera.azimuth < start_azimuth - 1.2, f"middle-drag should orbit: {start_azimuth} -> {camera.azimuth}"
        assert abs(camera.radius - expected_radius) < 1e-4, f"radius {camera.radius}, expected {expected_radius}"
        print(f"camera azimuth {start_azimuth:.2f} -> {camera.azimuth:.2f}, "
              f"radius {start_radius:.2f} -> {camera.radius:.3f} (expected {expected_radius:.3f})")


if __name__ == "__main__":
    main()
