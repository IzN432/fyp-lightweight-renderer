#pragma once

#include "core/scene/Component.hpp"

#include <glm/glm.hpp>

#include <memory>

namespace lr
{

enum class RigidBodyType
{
    Static,
    Dynamic,
};

// Dynamic state for one non-deformable body. Spatial state deliberately remains in the
// owning object's TransformComponent so position and orientation have one source of truth.
class RigidBodyComponent : public Component
{
public:
    explicit RigidBodyComponent(float mass = 1.0f, RigidBodyType type = RigidBodyType::Dynamic);

    RigidBodyType type() const { return m_type; }
    bool          isStatic() const { return m_type == RigidBodyType::Static; }
    bool          isDynamic() const { return m_type == RigidBodyType::Dynamic; }

    float mass() const { return m_mass; }
    float inverseMass() const { return m_inverseMass; }
    const glm::vec3 &inertiaDiagonal() const { return m_inertiaDiagonal; }
    const glm::vec3 &inverseInertiaDiagonal() const { return m_inverseInertiaDiagonal; }
    float linearDrag() const { return m_linearDrag; }
    float angularDrag() const { return m_angularDrag; }

    const glm::vec3 &linearVelocity() const { return m_linearVelocity; }
    const glm::vec3 &angularVelocity() const { return m_angularVelocity; }
    const glm::vec3 &accumulatedForce() const { return m_accumulatedForce; }
    const glm::vec3 &accumulatedTorque() const { return m_accumulatedTorque; }

    void setType(RigidBodyType type);
    void setMass(float mass);
    void setInertiaDiagonal(const glm::vec3 &inertiaDiagonal);
    void setLinearDrag(float drag);
    void setAngularDrag(float drag);
    void setLinearVelocity(const glm::vec3 &velocity);
    void setAngularVelocity(const glm::vec3 &velocity);

    void addForce(const glm::vec3 &force);
    void addTorque(const glm::vec3 &torque);
    void clearAccumulators();

    // Body setup and velocity, matching what the scene format stores for a rigid body. The force
    // and torque accumulators are left out on purpose: they are cleared every solver step, so
    // copying them would paste a value that has already expired.
    struct Values
    {
        RigidBodyType type = RigidBodyType::Dynamic;
        float         mass = 1.0f;
        glm::vec3     inertiaDiagonal{1.0f / 6.0f};
        float         linearDrag  = 0.05f;
        float         angularDrag = 0.05f;
        glm::vec3     linearVelocity{0.0f};
        glm::vec3     angularVelocity{0.0f};

        bool operator==(const Values &) const = default;
    };

    // What a copy carries plus the force and torque accumulators. The inspector drives those, so
    // undo has to put them back, even though copying them onto another object would paste a value
    // that has already expired (see Values).
    struct EditedValues
    {
        Values    values;
        glm::vec3 accumulatedForce{0.0f};
        glm::vec3 accumulatedTorque{0.0f};

        bool operator==(const EditedValues &) const = default;
    };

    std::unique_ptr<ComponentValues> copyValues() const override;
    void                             pasteValues(const ComponentValues &values) override;
    ComponentValuesAdder             valuesAdder() const override;

    std::unique_ptr<ComponentValues> undoValues() const override;
    void                             restoreUndoValues(const ComponentValues &values) override;

    // See ColliderComponent::allowsRemoval: the physics backend is rebuilt from the scene, and its
    // step tolerates a body whose component has gone since.
    bool allowsRemoval() const override { return true; }

    void onGUIImpl() override;

private:
    void updateInverseMass();
    void updateInverseInertia();

    RigidBodyType m_type        = RigidBodyType::Dynamic;
    float         m_mass        = 1.0f;
    float         m_inverseMass = 1.0f;

    // Principal moments of inertia in body-local space. The default is a solid unit cube;
    // callers may replace it with moments calculated from the body's actual mass distribution.
    glm::vec3 m_inertiaDiagonal{1.0f / 6.0f};
    glm::vec3 m_inverseInertiaDiagonal{6.0f};
    float     m_linearDrag  = 0.05f;
    float     m_angularDrag = 0.05f;

    glm::vec3 m_linearVelocity{0.0f};
    glm::vec3 m_angularVelocity{0.0f};
    glm::vec3 m_accumulatedForce{0.0f};
    glm::vec3 m_accumulatedTorque{0.0f};
};

} // namespace lr
