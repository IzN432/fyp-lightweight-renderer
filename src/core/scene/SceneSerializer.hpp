#pragma once

#include "core/scene/SceneObjectId.hpp"

#include <filesystem>
#include <memory>
#include <vector>

namespace lr
{

class SceneAssets;
class Scene;
class MeshStore;
class MaterialStore;
class AnimationLibrary;

// Versioned, CPU-only scene persistence. A single .lrscene file contains its JSON manifest followed
// by the bulk mesh/texture payload. Objects, meshes and materials are named by the UUIDs they
// already carry, so a loaded object is the same object it was when saved and every stored reference
// still resolves. Mesh and material handles are per-session and are resolved through their stores.
//
// RULE FOR COMPONENT DESERIALIZATION — a component's deserialization restores that component's own
// state and nothing else. In particular it must not call SceneObject::getComponent (or
// hasComponent, or reach another object through the Scene) to read a sibling component, because the
// components of an object are created in whatever order the manifest lists them: a sibling may not
// exist yet, and getComponent would throw, or worse, succeed only by accident of ordering.
//
// Anything that has to read a sibling or another object belongs in Component::onLoaded, which this
// serializer runs over every loaded object once the entire scene — all objects, all components, all
// parent links — is in place. SphericalCameraController is the worked example: its codec calls
// restoreOrbitState, which touches only its own fields, and its onLoaded then places the camera
// through the object's TransformComponent.
//
// Keeping to this means component deserialization has no order relationship to any other
// component, so codecs can be registered and run in any order.
class SceneSerializer
{
public:
    static void                         save(const SceneAssets &assets, const std::filesystem::path &path);
    static void                         save(const Scene &scene, const MeshStore &meshes,
                                             const MaterialStore &materials, const AnimationLibrary &animations,
                                             const std::filesystem::path &path);
    static std::unique_ptr<SceneAssets> load(const std::filesystem::path &path);
    // Appends a saved scene to caller-owned stores, preserving existing objects and assets, and
    // returns the objects it created. Throws if the file names an object, mesh or material whose
    // identity the caller's scene or stores already hold.
    static std::vector<SceneObjectId> load(const std::filesystem::path &path, Scene &scene,
                                           MeshStore &meshes, MaterialStore &materials,
                                           AnimationLibrary &animations,
                                           bool remapCameraObject = false);
};

} // namespace lr
