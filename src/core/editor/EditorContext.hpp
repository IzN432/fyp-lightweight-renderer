#pragma once

#include "core/scene/ComponentCatalog.hpp"
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

// Receives a component the Inspector has just added to an object, whether pasted or picked from the
// catalog, so the editor can bring its own state up to date. The Inspector can add a component by
// itself, but not know that an object which has just become renderable has to be registered with
// the GPU side — that lives in the editor.
class ComponentAddService
{
public:
    virtual ~ComponentAddService()                     = default;
    virtual void onComponentAdded(SceneObject &object) = 0;
};

struct EditorContext
{
    TransformEditService &transformEdits;
    CommandManager       &commands;
    ComponentAddService  &componentAdds;
    // The component types the Inspector's "Add Component" offers.
    const ComponentCatalog &componentCatalog;
    // Owned here rather than referenced, because nothing outside the inspector's copy/paste menu
    // touches it and it should live exactly as long as the editor session does.
    ComponentClipboard componentClipboard;
};

} // namespace lr
