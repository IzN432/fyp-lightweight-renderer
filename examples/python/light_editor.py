"""Light editor: add, remove and change lights while the engine's deferred renderer is running.

Everything goes through the scene; the engine does the rest:
  scene.add_light(...)      a SceneGpu showing the scene builds the light's quad, re-packs the geometry and
                            re-uploads the light buffer on the next frame (GeometryPass and PbrPass follow)
  scene.remove(obj)         the light stops lighting and drawing from the next frame
  obj.set_light(...)        type, colour, intensity, area size, two-sidedness, cone angles
  obj.position / rotation   move and aim it (lights shine along their -Z axis)

Use the "Lights" panel: pick a light, add point/spot/area lights around the model, remove the selected one,
edit it, or let it orbit. Middle-drag orbits the camera, Shift + middle-drag pans, the wheel zooms.

    PYTHONPATH=build/python python examples/python/light_editor.py [model.glb]
    PYTHONPATH=build/python python examples/python/light_editor.py --scripted   # self-checking run (ctest)
"""

import argparse
import math
import pathlib
from typing import Any

import numpy as np

import lr
from lr import engine

ASSETS = pathlib.Path(lr.ASSET_DIR) / "samples"
DEFAULT_MODEL = ASSETS / "models" / "bird_orange.glb"
HDRI = ASSETS / "environments" / "cedar_bridge_sunset_2_4k.hdr"
TYPES = ["point", "spot", "area", "directional"]
LIGHT_TYPE_CODE = {"point": 0, "spot": 1, "area": 2, "directional": 3, "image": 4}


def look_rotation(direction) -> tuple[float, float, float, float]:
    """Quaternion (x, y, z, w) turning the light's -Z axis towards `direction`."""
    forward = np.asarray(direction, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    up = (0.0, 1.0, 0.0) if abs(forward[1]) < 0.99 else (1.0, 0.0, 0.0)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    m = np.column_stack([right, true_up, -forward])  # columns: where X, Y, Z go
    w = math.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2.0
    x = math.copysign(math.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) / 2.0, m[2, 1] - m[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - m[0, 0] + m[1, 1] - m[2, 2])) / 2.0, m[0, 2] - m[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - m[0, 0] - m[1, 1] + m[2, 2])) / 2.0, m[1, 0] - m[0, 1])
    return (x, y, z, w)


def mesh_bounds(scene: engine.Scene) -> tuple[np.ndarray, np.ndarray]:
    lows, highs = [], []
    for obj in scene.objects:
        if obj.mesh is not None and obj.mesh.vertex_count:
            world = obj.mesh.positions @ obj.world_matrix[:3, :3].T + obj.world_matrix[:3, 3]
            lows.append(world.min(axis=0))
            highs.append(world.max(axis=0))
    return np.min(lows, axis=0), np.max(highs, axis=0)


class LightEditor:
    def __init__(self, scene: engine.Scene, center: np.ndarray, size: float):
        self.scene, self.center, self.size = scene, center, size
        self.selected = 0
        self.orbit = False
        self.added = 0

    def lights(self) -> list[engine.SceneObject]:
        return [obj for obj in self.scene.objects if obj.light is not None]

    def place(self, obj: engine.SceneObject, angle: float, height: float = 0.6):
        radius = self.size * 0.9
        position = self.center + np.array([math.sin(angle) * radius, height * self.size, math.cos(angle) * radius])
        obj.position = tuple(position.tolist())
        obj.rotation = look_rotation(self.center - position)

    def add(self, light_type: str, **parameters) -> engine.SceneObject:
        self.added += 1
        colors = [(1.0, 0.45, 0.2), (0.3, 0.6, 1.0), (0.5, 1.0, 0.4), (1.0, 0.9, 0.6)]
        parameters.setdefault("color", colors[self.added % len(colors)])
        parameters.setdefault("intensity", 6.0 if light_type != "area" else 4.0)
        parameters.setdefault("size", (0.3 * self.size, 0.2 * self.size))
        obj = self.scene.add_light(light_type, name=f"{light_type.title()} {self.added}", **parameters)
        self.place(obj, angle=self.added * 2.1)
        return obj

    def gui(self, gpu: engine.SceneGpu):
        lights = self.lights()
        with lr.gui.window("Lights", size=(400, 470), position=(20, 20)):
            lr.gui.text(f"{gpu.num_lights} / {gpu.max_lights} lights on the GPU")
            for light_type in ("point", "spot", "area"):
                if lr.gui.button(f"Add {light_type}") and len(lights) < gpu.max_lights:
                    self.add(light_type)
                    self.selected = len(self.lights()) - 1
                lr.gui.same_line()
            lr.gui.spacing()
            if not lights:
                return
            self.selected = min(self.selected, len(lights) - 1)
            _, self.selected = lr.gui.combo("Light", self.selected, [obj.name for obj in lights])
            obj = lights[self.selected]
            if lr.gui.button("Remove"):
                self.scene.remove(obj)
                return
            lr.gui.separator()
            self.edit(obj)

    def edit(self, obj: engine.SceneObject):
        light = obj.light
        assert light is not None
        if light.type == "image":
            lr.gui.text("Environment light (from the Ibl)")
            changed, intensity = lr.gui.slider_float("Intensity", light.intensity, 0.0, 3.0)
            if changed:
                obj.set_light(intensity=intensity)
            return

        changed, index = lr.gui.combo("Type", TYPES.index(light.type), TYPES)
        if changed:
            obj.set_light(type=TYPES[index])
        changed, color = lr.gui.color_edit3("Color", light.color)
        if changed:
            obj.set_light(color=color)
        changed, intensity = lr.gui.slider_float("Intensity", light.intensity, 0.0, 1000.0)
        if changed:
            obj.set_light(intensity=intensity)
        if light.type == "spot" and light.inner_cone_degrees is not None and light.outer_cone_degrees is not None:
            changed_inner, inner = lr.gui.slider_float("Inner cone", light.inner_cone_degrees, 1.0, 89.0)
            changed_outer, outer = lr.gui.slider_float("Outer cone", light.outer_cone_degrees, 1.0, 89.0)
            if changed_inner or changed_outer:
                obj.set_light(inner_cone_degrees=min(inner, outer), outer_cone_degrees=outer)
        if light.type == "area" and light.area_size is not None:
            width, height = light.area_size
            changed_w, width = lr.gui.slider_float("Width", width, 0.01, self.size)
            changed_h, height = lr.gui.slider_float("Height", height, 0.01, self.size)
            if changed_w or changed_h:
                obj.set_light(size=(width, height))
            changed, two_sided = lr.gui.checkbox("Two-sided", bool(light.two_sided))
            if changed:
                obj.set_light(two_sided=two_sided)

        lr.gui.separator()
        position = list(obj.position)
        moved = False
        for axis, name in enumerate("XYZ"):
            changed, position[axis] = lr.gui.slider_float(f"Position {name}", position[axis], -2 * self.size,
                                                          2 * self.size)
            moved |= changed
        if moved:
            obj.position = position
            obj.rotation = look_rotation(self.center - np.asarray(position))
        _, self.orbit = lr.gui.checkbox("Orbit around the model", self.orbit)

    def update(self, dt: float):
        lights = self.lights()
        if self.orbit and lights:
            obj = lights[min(self.selected, len(lights) - 1)]
            offset = np.asarray(obj.position) - self.center
            angle = math.atan2(offset[0], offset[2]) + dt * 1.2
            self.place(obj, angle, height=offset[1] / self.size)


def gpu_lights(viewer: lr.Viewer, gpu: engine.SceneGpu) -> np.ndarray:
    """The light buffer as the GPU sees it: one row of 20 floats per light (see LightGpuData)."""
    data = viewer.resources.read_buffer(gpu.light_buffer).view(np.float32).reshape(-1, 20)
    return data[: gpu.num_lights]


def scripted_check(frame: int, editor: LightEditor, viewer: lr.Viewer, gpu: engine.SceneGpu, state: dict[str, Any]):
    """Adds, edits and removes lights on a schedule, checking the GPU light buffer after each step."""

    def types():
        return [int(v) for v in gpu_lights(viewer, gpu)[:, 3].view(np.uint32)]

    if frame == 3:
        assert gpu.num_lights == 2 and types() == [3, 4], types()
        state["point"] = editor.add("point", intensity=5.0, color=(1.0, 0.0, 0.0))
    elif frame == 6:
        assert gpu.num_lights == 3 and types() == [3, 4, 0], types()
        row = gpu_lights(viewer, gpu)[2]
        assert np.allclose(row[8:11], (1, 0, 0)) and row[11] == 5.0, row
        assert np.allclose(row[0:3], state["point"].position, atol=1e-5), (row, state["point"].position)
        state["point"].set_light(intensity=9.0, color=(0.0, 1.0, 0.0))
    elif frame == 9:
        row = gpu_lights(viewer, gpu)[2]
        assert np.allclose(row[8:11], (0, 1, 0)) and row[11] == 9.0, row
        state["point"].set_light(type="area", size=(0.4, 0.2), two_sided=False)
        state["spot"] = editor.add("spot", outer_cone_degrees=25.0)
    elif frame == 12:
        assert types() == [3, 4, 2, 1], types()
        rows = gpu_lights(viewer, gpu)
        assert np.allclose(rows[2][14:16], (0.4, 0.2)) and rows[2][16:17].view(np.uint32)[0] == 0, rows[2]
        assert state["point"].light.two_sided is False
        assert math.isclose(rows[3][13], math.radians(25.0), rel_tol=1e-5), rows[3]
        state["point"].position = (0.0, 2.0, 0.0)
        editor.scene.remove(state["point"])
    elif frame == 15:
        assert gpu.num_lights == 3 and types() == [3, 4, 1], types()
        assert state["point"].id not in [obj.id for obj in editor.scene.objects]
        editor.scene.remove(state["spot"])
    elif frame == 18:
        assert gpu.num_lights == 2 and types() == [3, 4], types()
        print("scripted light edits: all checks passed")
        viewer.close()


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("model", nargs="?", default=str(DEFAULT_MODEL), help="OBJ, glTF or GLB file")
    parser.add_argument("--scripted", action="store_true", help="run a self-checking sequence of edits, then exit")
    args = parser.parse_args()

    scene = engine.load_scene(args.model)
    low, high = mesh_bounds(scene)
    center, size = (low + high) / 2, float(np.linalg.norm(high - low))
    scene.add_light("directional", intensity=0.6, rotation=look_rotation((-0.3, -1.0, -0.4)), name="Sun")
    scene.add_light("image", intensity=0.3, name="Environment")

    viewer = lr.Viewer(title="lr - light editor (Python)", width=1280, height=720)
    camera = lr.OrbitCamera(viewer)
    camera.target, camera.radius, camera.elevation = tuple(center.tolist()), size * 1.6, 0.35
    ibl = engine.Ibl(viewer, hdri=HDRI, env_res=512, pf_res=256, pf_mips=6)
    gpu = engine.SceneGpu(viewer, scene, camera)
    engine.GeometryPass(viewer, gpu)
    engine.AmbientOcclusionPass(viewer, gpu, sphere_radius=size * 0.02)
    engine.PbrPass(viewer, gpu, ibl)
    engine.CompositePass(viewer, gpu)

    editor = LightEditor(scene, center, size)
    if not args.scripted:
        editor.add("spot", intensity=10.0)
        editor.add("area", intensity=6.0)
        editor.selected = 2

    state: dict[str, Any] = {"frame": 0}

    def update(dt, extent):
        camera.update(dt)
        editor.update(dt)
        state["frame"] += 1
        if args.scripted:
            scripted_check(state["frame"], editor, viewer, gpu, state)

    viewer.on_update(update)
    viewer.on_gui(lambda: editor.gui(gpu))
    viewer.run()


if __name__ == "__main__":
    main()
