#pragma once

namespace lr
{

class EditorShortcuts;
class EditorStateController;
struct EditableMeshContext;

// The host capabilities a tool is allowed to use while registering itself. Exists so that widening
// what the host offers — an input router, a render bridge — does not change EditorTool's interface
// or every tool's registration signature.
class EditorServices
{
public:
    EditorServices(EditorStateController &states, EditorShortcuts &shortcuts) : m_states(states), m_shortcuts(shortcuts)
    {}

    EditorStateController &states() const { return m_states; }
    EditorShortcuts       &shortcuts() const { return m_shortcuts; }

private:
    EditorStateController &m_states;
    EditorShortcuts       &m_shortcuts;
};

// A feature's entire contribution to the editor: the states it adds, the panel it draws, and how it
// follows the session's editable mesh. The host drives registered tools generically, so adding a
// feature means writing an EditorTool rather than extending the session that hosts it.
class EditorTool
{
public:
    virtual ~EditorTool() = default;

    // Called once, before any target notification. The tool registers its states and shortcuts here.
    virtual void registerWith(EditorServices &services) = 0;

    // The editable mesh was replaced, or went away. A tool must assume anything it derived from the
    // previous mesh — factorizations, analysis results, cached indices — is invalid.
    virtual void onTargetChanged(const EditableMeshContext &target) = 0;
    virtual void onTargetCleared()                                  = 0;

    // Contents of this tool's section of the host's feature panel. The host draws the surrounding
    // header, so a tool does not get to decide where in the panel it sits.
    virtual void        drawPanel()         = 0;
    virtual const char *displayName() const = 0;
};

} // namespace lr
