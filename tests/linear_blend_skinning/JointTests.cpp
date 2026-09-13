#include "features/linear_blend_skinning/Joint.hpp"
#include "features/linear_blend_skinning/Skin.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/epsilon.hpp>

#include <cassert>
#include <stdexcept>
#include <type_traits>
#include <vector>

int main()
{
    static_assert(std::is_same_v<decltype(lr::SkeletonNode::localTransform), lr::Transform>);
    static_assert(std::is_same_v<decltype(lr::Joint::node), lr::SkeletonNodeIndex>);

    lr::SkeletonNode root;
    root.localTransform.setPosition(glm::vec3(1.0f, 2.0f, 3.0f));

    lr::SkeletonNode child;
    child.parent = 0;
    child.localTransform.setPosition(glm::vec3(0.0f, 4.0f, 0.0f));

    const glm::mat4 rootWorld  = root.localTransform.localMatrix();
    const glm::mat4 childWorld = child.localTransform.worldMatrix(rootWorld);

    assert(glm::all(glm::epsilonEqual(glm::vec3(childWorld[3]), glm::vec3(1.0f, 6.0f, 3.0f), 0.0001f)));

    const lr::Joint joint{.node = 1, .inverseBindMatrix = glm::mat4(1.0f)};
    assert(joint.node == 1);

    lr::Skin skin({root, child}, {joint});
    assert(skin.nodes().size() == 2);
    assert(skin.joints().size() == 1);

    lr::Transform posedTransform;
    posedTransform.setPosition(glm::vec3(5.0f, 0.0f, 0.0f));
    skin.setNodeTransform(1, posedTransform);
    assert(skin.node(1).localTransform.position().x == 5.0f);

    skin.evaluate(glm::mat4(1.0f));
    assert(glm::all(glm::epsilonEqual(glm::vec3(skin.nodeWorldMatrices()[1][3]),
                                     glm::vec3(6.0f, 2.0f, 3.0f), 0.0001f)));
    assert(glm::all(glm::epsilonEqual(glm::vec3(skin.jointMatrices()[0][3]),
                                     glm::vec3(6.0f, 2.0f, 3.0f), 0.0001f)));

    skin.resetPose();
    skin.evaluate(glm::mat4(1.0f));
    assert(glm::all(glm::epsilonEqual(glm::vec3(skin.nodeWorldMatrices()[1][3]),
                                     glm::vec3(1.0f, 6.0f, 3.0f), 0.0001f)));

    lr::SkeletonNode childBeforeParent = child;
    childBeforeParent.parent = 1;
    lr::Skin outOfOrder({childBeforeParent, root}, {{.node = 0}});
    outOfOrder.evaluate(glm::mat4(1.0f));
    assert(glm::all(glm::epsilonEqual(glm::vec3(outOfOrder.nodeWorldMatrices()[0][3]),
                                     glm::vec3(1.0f, 6.0f, 3.0f), 0.0001f)));

    bool invalidJointRejected = false;
    try
    {
        lr::Skin invalid({root}, {{.node = 1}});
    }
    catch (const std::invalid_argument &)
    {
        invalidJointRejected = true;
    }
    assert(invalidJointRejected);

    bool invalidParentRejected = false;
    try
    {
        lr::SkeletonNode invalidParent;
        invalidParent.parent = 1;
        lr::Skin invalid({invalidParent}, {});
    }
    catch (const std::invalid_argument &)
    {
        invalidParentRejected = true;
    }
    assert(invalidParentRejected);

    bool cycleRejected = false;
    try
    {
        lr::SkeletonNode first;
        first.parent = 1;
        lr::SkeletonNode second;
        second.parent = 0;
        lr::Skin invalid({first, second}, {});
    }
    catch (const std::invalid_argument &)
    {
        cycleRejected = true;
    }
    assert(cycleRejected);
}
