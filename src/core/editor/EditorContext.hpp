#pragma once

#include "core/scene/ComponentClipboard.hpp"

#include <functional>

namespace lr
{

class SceneObject;
class CommandManager;

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
    CommandManager       &commands;
    // Owned here rather than referenced, because nothing outside the inspector's copy/paste menu
    // touches it and it should live exactly as long as the editor session does.
    ComponentClipboard componentClipboard;
};

} // namespace lr
