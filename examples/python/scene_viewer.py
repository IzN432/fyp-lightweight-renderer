"""Scene viewer — a renderer written in Python, drawing a glTF/OBJ scene loaded with the engine's loaders.

  lr.load_scene(path)       objects, hierarchy, world matrices, meshes and materials (numpy), on the CPU
  -> one vertex buffer      every mesh's positions + "normal" + "uv", concatenated
  -> one index buffer       each mesh's faces, sorted by material, so each (object, material) is one range
  -> a texture array        each material's baseColorTexture (white if it has none)
  -> one pass               a draw per range: draw_indexed with a vertex offset, plus the object's world matrix,
                            the material's baseDiffuse factor and its texture index as push constants
                            (albedo = texture * baseDiffuse, as the engine's geometry.frag computes it)

The camera is the engine's orbit camera (middle-drag orbit, Shift+middle-drag pan, scroll zoom, R reset),
framed to the scene's bounds. Skinned meshes are drawn in their rest pose.

Run from the repo root after building:
    PYTHONPATH=build/python python examples/python/scene_viewer.py [path/to/model.glb] [--frames N]
"""

import argparse
import pathlib
from typing import Any

import numpy as np

import lr

SHADERS = pathlib.Path(__file__).parent / "shaders"
DEFAULT_MODEL = pathlib.Path(lr.ASSET_DIR) / "samples" / "models" / "bird_orange.glb"


def build_geometry(scene: lr.Scene):
    """Concatenate every mesh into one vertex + one index buffer; return them with per-range draw records."""
    vertices, indices, draws = [], [], []
    materials: dict[int, int] = {}  # material handle -> texture array index
    vertex_offset = index_offset = 0
    for obj in scene.objects:
        mesh = obj.mesh
        if mesh is None or mesh.face_count == 0:
            continue
        names = mesh.attribute_names
        normal = mesh.attribute("normal") if "normal" in names else np.zeros((mesh.vertex_count, 3), np.float32)
        uv = mesh.attribute("uv") if "uv" in names else np.zeros((mesh.vertex_count, 2), np.float32)
        vertices.append(np.concatenate([mesh.positions, normal, uv], axis=1).astype(np.float32))

        faces = mesh.indices
        face_materials = mesh.face_materials
        if len(face_materials) != len(faces):
            face_materials = np.zeros(len(faces), np.uint32)
        order = np.argsort(face_materials, kind="stable")
        faces, face_materials = faces[order], face_materials[order]
        indices.append(faces.reshape(-1))
        for handle in np.unique(face_materials):
            first, last = np.searchsorted(face_materials, handle), np.searchsorted(face_materials, handle, "right")
            draws.append({
                "first_index": index_offset + 3 * int(first),
                "index_count": 3 * int(last - first),
                "vertex_offset": vertex_offset,
                "object": obj,
                "material": materials.setdefault(int(handle), len(materials)),
            })
        vertex_offset += mesh.vertex_count
        index_offset += faces.size

    if not draws:
        raise SystemExit("the scene has no meshes")
    return np.concatenate(vertices), np.concatenate(indices).astype(np.uint32), draws, materials


WHITE = np.full((1, 1, 4), 255, dtype=np.uint8)


def base_color(material: lr.Material) -> tuple[np.ndarray, np.ndarray]:
    """(texture, factor): albedo = texture * factor, as the engine's geometry.frag computes it."""
    texture = material.texture("baseColorTexture") if "baseColorTexture" in material.texture_names else WHITE
    factor = np.array(material.parameters.get("baseDiffuse", (1.0, 1.0, 1.0, 1.0)), dtype=np.float32)
    return texture, factor


def world_bounds(draws) -> tuple[np.ndarray, np.ndarray]:
    lows, highs = [], []
    for obj in {id(d["object"]): d["object"] for d in draws}.values():
        positions = obj.mesh.positions
        world = positions @ obj.world_matrix[:3, :3].T + obj.world_matrix[:3, 3]
        lows.append(world.min(axis=0))
        highs.append(world.max(axis=0))
    return np.min(lows, axis=0), np.max(highs, axis=0)


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("model", nargs="?", default=str(DEFAULT_MODEL), help="OBJ, glTF or GLB file")
    parser.add_argument("--frames", type=int, default=0, help="close after this many frames (0 = run until closed)")
    args = parser.parse_args()

    scene = lr.load_scene(args.model)
    vertices, indices, draws, materials = build_geometry(scene)

    viewer = lr.Viewer(title=f"lr - {pathlib.Path(args.model).name} (Python)", width=1280, height=720)
    fg, res = viewer.frame_graph, viewer.resources
    res.upload_buffer("scene_vertices", vertices, lr.BufferUsage.VERTEX)
    res.upload_buffer("scene_indices", indices, lr.BufferUsage.INDEX)
    res.register_dynamic_buffer("camera", 2 * 64, lr.BufferUsage.UNIFORM)
    factors = np.zeros((len(materials), 4), dtype=np.float32)
    for handle, index in materials.items():
        texture, factors[index] = base_color(scene.material(handle))
        res.upload_array_image("base_colors", index, np.ascontiguousarray(texture), lr.Format.R8G8B8A8_SRGB,
                               generate_mipmaps=True)

    camera = lr.OrbitCamera(viewer)
    low, high = world_bounds(draws)
    camera.target = tuple(((low + high) / 2).tolist())
    camera.radius = float(np.linalg.norm(high - low)) * 1.2
    camera.elevation = 0.3

    frag = (SHADERS / "scene.frag").read_text().replace(
        "#version 450", f"#version 450\n#define MATERIAL_COUNT {len(materials)}", 1)

    push = np.zeros(21, dtype=np.float32)  # mat4 model, vec4 base colour factor, uint material

    def draw_scene(ctx):
        for d in draws:
            push[:16] = lr.transforms.to_gpu(d["object"].world_matrix)
            push[16:20] = factors[d["material"]]
            push.view(np.uint32)[20] = d["material"]
            ctx.push_constants(lr.Stage.VERTEX | lr.Stage.FRAGMENT, push)
            ctx.cmd.draw_indexed(d["index_count"], first_index=d["first_index"], vertex_offset=d["vertex_offset"])

    (
        fg.add_pass("scene")
        .type(lr.PassType.GEOMETRY)
        .vert_shader(SHADERS / "scene.vert")
        .frag_shader(lr.compile_glsl_source(frag, lr.ShaderStage.FRAGMENT, str(SHADERS / "scene.frag")))
        .vertex_layout(
            [lr.VertexBinding(0, stride=vertices.strides[0])],
            [
                lr.VertexAttribute(0, lr.Format.R32G32B32_SFLOAT, 0),
                lr.VertexAttribute(1, lr.Format.R32G32B32_SFLOAT, 12),
                lr.VertexAttribute(2, lr.Format.R32G32_SFLOAT, 24),
            ],
        )
        .vertex_buffer(0, fg.buffer("scene_vertices"))
        .index_buffer(fg.buffer("scene_indices"))
        .uniform_buffer(0, fg.buffer("camera"), lr.Stage.VERTEX)
        .sampled_image_array(1, fg.image("base_colors"), len(materials), lr.Stage.FRAGMENT)
        .push_constant_size(push.nbytes, lr.Stage.VERTEX | lr.Stage.FRAGMENT)
        .cull(lr.CullMode.NONE)  # imported meshes aren't always closed or consistently wound
        .color_attachment(fg.image("swapchain"), viewer.swapchain_format, clear_color=(0.08, 0.08, 0.1, 1.0))
        .depth_attachment(fg.image("depth"))
        .execute(draw_scene)
    )

    mesh_objects = len({id(d["object"]) for d in draws})
    triangles = len(indices) // 3

    def gui():
        with lr.gui.window("Scene", size=(330, 150), position=(20, 20)):
            lr.gui.text(pathlib.Path(args.model).name)
            lr.gui.text(f"{len(scene.objects)} objects, {mesh_objects} with meshes")
            lr.gui.text(f"{triangles} triangles, {len(materials)} materials, {len(draws)} draws")
            lr.gui.text(f"{lr.gui.framerate():.0f} fps")

    state: dict[str, Any] = {"frames": 0}

    def update(dt, extent):
        camera.update(dt)
        res.update_buffer("camera", camera.matrices(extent))
        state["frames"] += 1
        if args.frames and state["frames"] >= args.frames:
            viewer.close()

    viewer.on_update(update)
    viewer.on_gui(gui)
    viewer.run()


if __name__ == "__main__":
    main()
