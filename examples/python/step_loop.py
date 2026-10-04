"""Your own frame loop: a particle fountain simulated in numpy, rendered one viewer.step() at a time.

With run(), the viewer owns the loop and calls you back. With step(), your code owns it:

    while viewer.step():          # handles window events, renders one frame; False once closed
        simulate(dt)              # plain Python between frames: numpy, I/O, anything
        upload(...)               # whatever you write here is what the next frame draws

Callbacks (on_update, on_gui) still run inside each step, so the ImGui panel and the orbit camera work as
usual. Space pauses the simulation; the panel tunes it. Middle-drag orbits, the wheel zooms.

    PYTHONPATH=build/python python examples/python/step_loop.py [--frames N]
"""

import argparse
import pathlib
import time

import numpy as np

import lr

SHADERS = pathlib.Path(__file__).parent / "shaders"
COUNT = 6000


class Fountain:
    """Particles launched upwards from the origin, falling under gravity and bouncing on the floor (y = 0)."""

    def __init__(self, count: int, seed: int = 1):
        self.rng = np.random.default_rng(seed)
        self.position = np.zeros((count, 3), np.float32)
        self.velocity = np.zeros((count, 3), np.float32)
        self.age = np.zeros(count, np.float32)
        self.lifetime = self.rng.uniform(2.0, 4.0, count).astype(np.float32)
        self.gravity = 9.81
        self.spread = 0.6
        self.bounce = 0.55
        self.respawn(np.ones(count, bool))
        self.age[:] = self.rng.uniform(0.0, 4.0, count)  # stagger, so it doesn't start as one burst

    def respawn(self, which: np.ndarray):
        n = int(which.sum())
        angle = self.rng.uniform(0.0, 2.0 * np.pi, n)
        speed = self.rng.uniform(0.2, 1.0, n) * self.spread
        self.position[which] = 0.0
        self.velocity[which] = np.stack([np.cos(angle) * speed, self.rng.uniform(4.5, 6.0, n),
                                         np.sin(angle) * speed], axis=1)
        self.age[which] = 0.0

    def simulate(self, dt: float):
        self.velocity[:, 1] -= self.gravity * dt
        self.position += self.velocity * dt
        below = self.position[:, 1] < 0.0
        self.position[below, 1] *= -1.0
        self.velocity[below, 1] *= -self.bounce
        self.velocity[below, 0::2] *= 0.8  # friction
        self.age += dt
        self.respawn(self.age > self.lifetime)

    def gpu_data(self) -> np.ndarray:
        """(count, 8) float32: vec4 position, vec4 colour (hot when young, cooling to blue)."""
        t = np.clip(self.age / self.lifetime, 0.0, 1.0)[:, None]
        hot, cold = np.array([1.0, 0.75, 0.25], np.float32), np.array([0.2, 0.4, 1.0], np.float32)
        color = hot * (1.0 - t) + cold * t
        data = np.zeros((len(self.position), 8), np.float32)
        data[:, :3], data[:, 4:7] = self.position, color
        return data


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--frames", type=int, default=0, help="stop after this many frames (0 = until closed)")
    args = parser.parse_args()

    fountain = Fountain(COUNT)
    viewer = lr.Viewer(title="lr - your own loop: viewer.step() (Python)", width=1280, height=720)
    fg, res = viewer.frame_graph, viewer.resources
    res.register_dynamic_buffer("camera", 128, lr.BufferUsage.UNIFORM)
    res.register_dynamic_buffer("particles", fountain.gpu_data().nbytes, lr.BufferUsage.STORAGE)
    radius = np.array([0.035], np.float32)

    def draw(ctx):
        ctx.push_constants(lr.Stage.VERTEX, radius)
        ctx.cmd.draw(6, instance_count=COUNT)

    (
        fg.add_pass("particles")
        .type(lr.PassType.GEOMETRY)
        .vert_shader(SHADERS / "particles.vert")
        .frag_shader(SHADERS / "particles.frag")
        .vertex_layout([], [])
        .uniform_buffer(0, fg.buffer("camera"), lr.Stage.VERTEX)
        .storage_buffer_read(1, fg.buffer("particles"), lr.Stage.VERTEX)
        .push_constant_size(radius.nbytes, lr.Stage.VERTEX)
        .cull(lr.CullMode.NONE)
        .color_attachment(fg.image(lr.SWAPCHAIN), viewer.swapchain_format, clear_color=(0.05, 0.05, 0.08, 1.0))
        .depth_attachment(fg.image("depth"))
        .execute(draw)
    )

    camera = lr.OrbitCamera(viewer)
    camera.target, camera.radius, camera.elevation = (0.0, 1.2, 0.0), 6.0, 0.25
    viewer.on_update(lambda dt, extent: camera.update(dt))  # callbacks still run inside each step

    paused = False
    timings = {"simulate": 0.0, "frame": 0.0}

    def gui():
        with lr.gui.window("Your own loop", size=(390, 170), position=(20, 20)):
            lr.gui.text(f"{COUNT} particles, simulated in numpy between steps")
            _, fountain.gravity = lr.gui.slider_float("Gravity", fountain.gravity, 0.0, 20.0)
            _, fountain.spread = lr.gui.slider_float("Spread", fountain.spread, 0.0, 3.0)
            _, fountain.bounce = lr.gui.slider_float("Bounce", fountain.bounce, 0.0, 1.0)
            lr.gui.text("Paused (Space)" if paused else "Running (Space pauses)")
            lr.gui.text(f"simulate {timings['simulate'] * 1e3:.2f} ms, frame {timings['frame'] * 1e3:.2f} ms")

    viewer.on_gui(gui)

    frames, space_was_down = 0, False
    last = time.perf_counter()
    while True:
        # --- between frames: your code ---------------------------------------------------------
        now = time.perf_counter()
        dt, last = min(now - last, 1.0 / 30.0), now  # clamp, so a stall (e.g. a window drag) doesn't explode
        space = viewer.input.is_key_down(lr.Key.SPACE)
        if space and not space_was_down:
            paused = not paused
        space_was_down = space
        if not paused:
            start = time.perf_counter()
            fountain.simulate(dt)
            timings["simulate"] = time.perf_counter() - start
        res.update_buffer("particles", fountain.gpu_data())
        res.update_buffer("camera", camera.matrices(res.extent))

        # --- one frame: events, callbacks, rendering ---------------------------------------------
        start = time.perf_counter()
        if not viewer.step():
            break
        timings["frame"] = time.perf_counter() - start
        frames += 1
        if args.frames and frames >= args.frames:
            viewer.close()

    # The loop is yours, so so is what happens after it.
    print(f"{frames} frames; particles in the air: {(fountain.position[:, 1] > 0.05).sum()} / {COUNT}")


if __name__ == "__main__":
    main()
