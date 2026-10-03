#pragma once

#include <functional>

namespace lr
{

class SceneObject;

enum class TransformTool
{
    None,
    Translate,
    Rotate,
    Scale,
};

class TransformEditService
{
public:
    virtual ~TransformEditService() = default;
    virtual void beginTransformEdit(SceneObject &target, TransformTool tool,
                                    std::function<void()> cancel) = 0;
    virtual void endTransformEdit() = 0;
};

struct EditorContext
{
    TransformEditService &transformEdits;
};

} // namespace lr
