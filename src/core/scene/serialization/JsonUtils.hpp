#pragma once

#include "core/scene/SceneObjectId.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <nlohmann/json.hpp>

#include <string>

namespace lr::scene_serialization
{

using json = nlohmann::json;

const json &required(const json &object, const char *key, const std::string &where);
json vec2(const glm::vec2 &value);
json vec3(const glm::vec3 &value);
json quat(const glm::quat &value);
json mat4(const glm::mat4 &value);
glm::mat4 readMat4(const json &value, const std::string &where);
Uuid readId(const json &value, const std::string &where);

template <glm::length_t N, typename T, glm::qualifier Q = glm::defaultp>
glm::vec<N, T, Q> readVector(const json &value, const std::string &where)
{
    if (!value.is_array() || value.size() != N)
        throw std::runtime_error("SceneSerializer: " + where + " must be an array of " +
                                 std::to_string(N) + " numbers");
    glm::vec<N, T, Q> result{};
    for (glm::length_t i = 0; i < N; ++i)
    {
        if (!value[i].is_number())
            throw std::runtime_error("SceneSerializer: " + where + " must contain only numbers");
        result[i] = value[i].get<T>();
    }
    return result;
}

} // namespace lr::scene_serialization
