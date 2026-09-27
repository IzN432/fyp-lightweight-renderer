#pragma once

namespace lr
{

// Surface properties used when two colliders generate a contact. Keeping these on the
// collider allows different parts of one rigid body to use different materials.
struct PhysicsMaterial
{
    float restitution = 0.5f;
    float friction    = 0.3f;
};

} // namespace lr
