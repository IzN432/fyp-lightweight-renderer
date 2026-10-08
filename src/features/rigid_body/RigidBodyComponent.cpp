#include "features/rigid_body/RigidBodyComponent.hpp"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace lr
{
namespace
{

constexpr float kMinimumMass = 0.001f;
constexpr float kMinimumInertia = 0.000001f;
constexpr const char *kBodyTypeNames[] = {"Static", "Dynamic"};

void validateMass(float mass)
{
    if (!std::isfinite(mass) || mass <= 0.0f)
    {
        throw std::invalid_argument("Rigid body mass must be finite and greater than zero");
    }
}

void validateInertia(const glm::vec3 &inertia)
{
    if (!std::isfinite(inertia.x) || !std::isfinite(inertia.y) || !std::isfinite(inertia.z) ||
        glm::any(glm::lessThanEqual(inertia, glm::vec3(0.0f))))
    {
        throw std::invalid_argument("Rigid body inertia moments must be finite and greater than zero");
    }
}

void validateDrag(float drag)
{
    if (!std::isfinite(drag) || drag < 0.0f)
    {
        throw std::invalid_argument("Rigid body drag must be finite and non-negative");
    }
}

} // namespace

RigidBodyComponent::RigidBodyComponent(float mass, RigidBodyType type)
    : Component("RigidBodyComponent"), m_type(type), m_mass(mass), m_inertiaDiagonal(mass / 6.0f)
{
    validateMass(mass);
    updateInverseMass();
    updateInverseInertia();
}

void RigidBodyComponent::updateInverseMass()
{
    m_inverseMass = isStatic() ? 0.0f : 1.0f / m_mass;
}

void RigidBodyComponent::updateInverseInertia()
{
    m_inverseInertiaDiagonal = isStatic()
                                   ? glm::vec3(0.0f)
                                   : glm::vec3(1.0f) / m_inertiaDiagonal;
}

void RigidBodyComponent::setType(RigidBodyType type)
{
    m_type = type;
    updateInverseMass();
    updateInverseInertia();
    markDirty();
}

void RigidBodyComponent::setMass(float mass)
{
    validateMass(mass);
    const float massRatio = mass / m_mass;
    m_mass = mass;
    m_inertiaDiagonal *= massRatio;
    updateInverseMass();
    updateInverseInertia();
    markDirty();
}

void RigidBodyComponent::setInertiaDiagonal(const glm::vec3 &inertiaDiagonal)
{
    validateInertia(inertiaDiagonal);
    m_inertiaDiagonal = inertiaDiagonal;
    updateInverseInertia();
    markDirty();
}

void RigidBodyComponent::setLinearDrag(float drag)
{
    validateDrag(drag);
    m_linearDrag = drag;
    markDirty();
}

void RigidBodyComponent::setAngularDrag(float drag)
{
    validateDrag(drag);
    m_angularDrag = drag;
    markDirty();
}

void RigidBodyComponent::setLinearVelocity(const glm::vec3 &velocity)
{
    m_linearVelocity = velocity;
    markDirty();
}

void RigidBodyComponent::setAngularVelocity(const glm::vec3 &velocity)
{
    m_angularVelocity = velocity;
    markDirty();
}

void RigidBodyComponent::addForce(const glm::vec3 &force)
{
    m_accumulatedForce += force;
    markDirty();
}

void RigidBodyComponent::addTorque(const glm::vec3 &torque)
{
    m_accumulatedTorque += torque;
    markDirty();
}

void RigidBodyComponent::clearAccumulators()
{
    m_accumulatedForce  = glm::vec3(0.0f);
    m_accumulatedTorque = glm::vec3(0.0f);
    markDirty();
}

std::unique_ptr<ComponentValues> RigidBodyComponent::copyValues() const
{
    return std::make_unique<ComponentValueSnapshot<Values>>(Values{m_type, m_mass, m_inertiaDiagonal,
                                                                   m_linearDrag, m_angularDrag,
                                                                   m_linearVelocity, m_angularVelocity});
}

void RigidBodyComponent::pasteValues(const ComponentValues &values)
{
    // Routed through the setters so a pasted body passes the same clamps and derived-quantity
    // updates — inverse mass, inverse inertia — as one edited by hand.
    const Values &pasted = componentValuesAs<Values>(values);
    setType(pasted.type);
    setMass(pasted.mass);
    setInertiaDiagonal(pasted.inertiaDiagonal);
    setLinearDrag(pasted.linearDrag);
    setAngularDrag(pasted.angularDrag);
    setLinearVelocity(pasted.linearVelocity);
    setAngularVelocity(pasted.angularVelocity);
}

void RigidBodyComponent::onGUIImpl()
{
    bool changed = false;

    int bodyType = static_cast<int>(m_type);
    if (ImGui::Combo("Body Type", &bodyType, kBodyTypeNames, 2))
    {
        setType(static_cast<RigidBodyType>(bodyType));
    }

    float mass = m_mass;
    if (ImGui::DragFloat("Mass", &mass, 0.05f, kMinimumMass, 100000.0f))
    {
        setMass(std::max(mass, kMinimumMass));
    }
    ImGui::Text("Inverse Mass: %.6f", m_inverseMass);

    glm::vec3 inertiaDiagonal = m_inertiaDiagonal;
    if (ImGui::DragFloat3("Inertia Diagonal", &inertiaDiagonal.x, 0.01f, kMinimumInertia, 100000.0f))
    {
        setInertiaDiagonal(glm::max(inertiaDiagonal, glm::vec3(kMinimumInertia)));
    }
    ImGui::Text("Inverse Inertia: %.4f, %.4f, %.4f",
                m_inverseInertiaDiagonal.x, m_inverseInertiaDiagonal.y, m_inverseInertiaDiagonal.z);

    float linearDrag = m_linearDrag;
    if (ImGui::DragFloat("Linear Drag", &linearDrag, 0.01f, 0.0f, 100.0f, "%.3f /s"))
    {
        setLinearDrag(std::max(linearDrag, 0.0f));
    }
    float angularDrag = m_angularDrag;
    if (ImGui::DragFloat("Angular Drag", &angularDrag, 0.01f, 0.0f, 100.0f, "%.3f /s"))
    {
        setAngularDrag(std::max(angularDrag, 0.0f));
    }

    changed |= ImGui::DragFloat3("Linear Velocity", &m_linearVelocity.x, 0.05f);
    changed |= ImGui::DragFloat3("Angular Velocity", &m_angularVelocity.x, 0.05f);
    changed |= ImGui::DragFloat3("Accumulated Force", &m_accumulatedForce.x, 0.05f);
    changed |= ImGui::DragFloat3("Accumulated Torque", &m_accumulatedTorque.x, 0.05f);

    if (ImGui::Button("Clear Force and Torque"))
    {
        clearAccumulators();
    }
    else if (changed)
    {
        markDirty();
    }
}

} // namespace lr
