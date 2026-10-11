#pragma once

#include "core/framegraph/ImageReadback.hpp"

#include <glm/vec2.hpp>

#include <optional>
#include <string>

namespace lr
{

class Viewer;

// Reads picking ids out of a picking pass's image at the mouse cursor, and filters raw pointer
// events down to the clicks that should actually pick. Both halves are shared by every picking
// layer (scene objects, skeleton bones), which otherwise each repeat the cursor-to-pixel mapping
// and the press/release gesture bookkeeping.
class ViewportPicker
{
public:
    ViewportPicker(const VulkanContext &ctx, Allocator &alloc);

    static constexpr uint32_t kNoData = ImageReadback::kNoData;

    // A drag that ends over geometry is not a click, so a press must release within this distance
    // (in screen pixels) to pick.
    static constexpr double kClickRadius = 4.0;

    // Reads the picking id under the mouse from imageName. Returns kNoData when nothing could be
    // read -- no frame rendered yet, a degenerate extent, or the cursor outside the display -- which
    // is a different answer from "nothing is there" (0) and callers must not treat as a miss.
    uint32_t readIdAtCursor(Viewer &viewer, const std::string &imageName);

    // Feeds a pointer button event through the click gesture. Returns true on the release that
    // completes a click; the press is recorded, and a release too far from it is dropped.
    bool clickCompleted(Viewer &viewer, int action);

    // Whether this layer should consume the event it just fed to clickCompleted: a press, which it
    // records, and the release that ends one, whether or not that release became a click. Call it
    // before clickCompleted, which clears the pending press.
    bool consumes(int action) const;

    void reset() { m_press.reset(); }

private:
    ImageReadback             m_readback;
    std::optional<glm::dvec2> m_press;
};

} // namespace lr
