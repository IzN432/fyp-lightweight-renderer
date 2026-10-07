#include "JsonUtils.hpp"

#include <stdexcept>

namespace lr::scene_serialization
{

const json &required(const json &object, const char *key, const std::string &where)
{
    if (!object.is_object() || !object.contains(key))
        throw std::runtime_error("SceneSerializer: missing " + where + "." + key);
    return object.at(key);
}

json vec2(const glm::vec2 &value) { return {value.x, value.y}; }
json vec3(const glm::vec3 &value) { return {value.x, value.y, value.z}; }
json quat(const glm::quat &value) { return {value.x, value.y, value.z, value.w}; }

json mat4(const glm::mat4 &value)
{
    json result = json::array();
    for (int column = 0; column < 4; ++column)
        for (int row = 0; row < 4; ++row) result.push_back(value[column][row]);
    return result;
}

glm::mat4 readMat4(const json &value, const std::string &where)
{
    if (!value.is_array() || value.size() != 16)
        throw std::runtime_error("SceneSerializer: " + where +
                                 " must be a 16-number column-major matrix");
    glm::mat4 result(1.0f);
    for (int column = 0; column < 4; ++column)
        for (int row = 0; row < 4; ++row)
        {
            const json &entry = value[column * 4 + row];
            if (!entry.is_number())
                throw std::runtime_error("SceneSerializer: " + where + " must contain only numbers");
            result[column][row] = entry.get<float>();
        }
    return result;
}

Uuid readId(const json &value, const std::string &where)
{
    if (!value.is_string())
        throw std::runtime_error("SceneSerializer: " + where + " must be a UUID string");
    try
    {
        return parseUuid(value.get<std::string>());
    }
    catch (const std::runtime_error &error)
    {
        throw std::runtime_error("SceneSerializer: " + where + ": " + error.what());
    }
}

} // namespace lr::scene_serialization
