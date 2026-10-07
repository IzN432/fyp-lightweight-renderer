#include "core/scene/serialization/ComponentCodec.hpp"
#include "core/scene/serialization/JsonUtils.hpp"

#include "core/scene/Scene.hpp"
#include "features/linear_blend_skinning/SkinComponent.hpp"

#include <memory>
#include <stdexcept>

namespace lr
{
namespace
{
using namespace scene_serialization;

class SkinComponentCodec final : public ComponentCodec
{
public:
    std::string_view key() const override { return "skin"; }
    std::type_index componentType() const override { return typeid(SkinComponent); }
    json encode(const SceneObject &object, const ComponentSaveContext &) const override
    {
        json joints = json::array();
        for (const Joint &joint : object.getComponent<SkinComponent>().skin().joints())
            joints.push_back({{"object", toString(joint.sceneObject)},
                              {"inverse_bind_matrix", mat4(joint.inverseBindMatrix)}});
        return {{"joints", std::move(joints)}};
    }
    void decode(const json &value, SceneObject &object, ComponentLoadContext &context,
                const std::string &where) const override
    {
        std::vector<Joint> joints;
        const json &serializedJoints = required(value, "joints", where);
        if (!serializedJoints.is_array())
            throw std::runtime_error("SceneSerializer: skin joints must be an array at " + where);
        for (size_t i = 0; i < serializedJoints.size(); ++i)
        {
            const json &joint = serializedJoints[i];
            const std::string jointWhere = where + ".joints[" + std::to_string(i) + "]";
            const SceneObjectId jointObject = readId(required(joint, "object", jointWhere), jointWhere + ".object");
            if (!context.scene.contains(jointObject))
                throw std::runtime_error("SceneSerializer: missing skin joint object " + toString(jointObject) + " at " + jointWhere);
            joints.push_back({jointObject, readMat4(required(joint, "inverse_bind_matrix", jointWhere),
                                                    jointWhere + ".inverse_bind_matrix")});
        }
        object.addComponent<SkinComponent>(Skin(context.scene, std::move(joints)));
    }
};
} // namespace

void addSkinComponentCodecs(ComponentCodecRegistry &registry)
{
    registry.add(std::make_unique<SkinComponentCodec>());
}

} // namespace lr
