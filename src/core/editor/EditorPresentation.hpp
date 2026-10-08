#pragma once

namespace lr
{

// Rendering and selection capabilities requested by an editor state. SceneManager consumes these
// flags without knowing whether the state is View, Edit, ARAP, analysis, or a future feature.
struct EditorPresentation
{
    bool skinningEnabled       = true;
    bool vertexPointsVisible   = false;
    bool heatmapVisible        = false;
    bool vertexSelectionActive = false;
    bool objectSelectionActive = false;

    friend bool operator==(const EditorPresentation &, const EditorPresentation &) = default;
};

} // namespace lr
