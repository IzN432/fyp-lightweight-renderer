#pragma once

namespace lr
{

class Mesh;
class SceneObject;

// The mesh a session's editing tools are currently bound to. Passed to every tool as one value so
// that widening what a tool may need from its target does not change every tool's signature.
struct EditableMeshContext
{
    SceneObject &object;
    Mesh        &mesh;
};

} // namespace lr
