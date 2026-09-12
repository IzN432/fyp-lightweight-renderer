#include "features/linear_blend_skinning/Joint.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/epsilon.hpp>

#include <cassert>
#include <type_traits>

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
}
