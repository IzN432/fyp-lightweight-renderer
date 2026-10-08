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

// Receives a component the inspector has just added to an object, so the editor can bring its own
// state up to date. The inspector can add the component by itself, but not know that an object which
// has just become renderable has to be registered with the GPU side — that lives in the editor.
class ComponentPasteService
{
public:
    virtual ~ComponentPasteService() = default;
    virtual void onComponentPasted(SceneObject &object) = 0;
};

struct EditorContext
{
    TransformEditService &transformEdits;
    CommandManager       &commands;
    ComponentPasteService &componentPastes;
    // Owned here rather than referenced, because nothing outside the inspector's copy/paste menu
    // touches it and it should live exactly as long as the editor session does.
    ComponentClipboard componentClipboard;
};

} // namespace lr
