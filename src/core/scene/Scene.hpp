#pragma once

#include "SceneObject.hpp"

#include <memory>
#include <vector>

namespace lr
{

class Scene
{
public:
    Scene() = default;

    void onGUI();

    SceneObject& createSceneObject() 
    {
        m_sceneObjects.push_back(std::make_unique<SceneObject>());
        return *m_sceneObjects.back();
    }

    const std::vector<std::unique_ptr<SceneObject>>& sceneObjects() const { return m_sceneObjects; }
private:
    std::vector<std::unique_ptr<SceneObject>> m_sceneObjects;
};

}
