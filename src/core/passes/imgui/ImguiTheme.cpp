#include "ImguiTheme.hpp"

#include <imgui.h>

namespace lr
{
namespace
{

// Palette: high-contrast near-black greys with a red accent.
constexpr ImVec4 kBg        = {0.055f, 0.055f, 0.060f, 1.00f};
constexpr ImVec4 kBgDark    = {0.020f, 0.020f, 0.024f, 1.00f};
constexpr ImVec4 kBgLight   = {0.130f, 0.130f, 0.140f, 1.00f};
constexpr ImVec4 kBgHovered = {0.205f, 0.205f, 0.220f, 1.00f};
constexpr ImVec4 kBorder    = {0.300f, 0.300f, 0.320f, 1.00f};
constexpr ImVec4 kText      = {0.950f, 0.950f, 0.955f, 1.00f};
constexpr ImVec4 kTextDim   = {0.460f, 0.460f, 0.480f, 1.00f};

// Red accent — kTitle is the deep red reserved for title bars / menu bar.
constexpr ImVec4 kTitle     = {0.420f, 0.075f, 0.090f, 1.00f};
constexpr ImVec4 kTitleDim  = {0.190f, 0.055f, 0.065f, 1.00f};
constexpr ImVec4 kAccent    = {0.760f, 0.160f, 0.180f, 1.00f};
constexpr ImVec4 kAccentHi  = {0.920f, 0.260f, 0.280f, 1.00f};
constexpr ImVec4 kAccentLo  = {0.520f, 0.100f, 0.120f, 1.00f};

ImVec4 withAlpha(const ImVec4 &c, float a)
{
    return {c.x, c.y, c.z, a};
}

} // namespace

void applyImguiTheme()
{
    ImGuiStyle &style = ImGui::GetStyle();
    ImGui::StyleColorsDark(&style);

    // --- Geometry -----------------------------------------------------------
    style.WindowPadding     = {10.0f, 10.0f};
    style.FramePadding      = {8.0f, 4.0f};
    style.CellPadding       = {6.0f, 4.0f};
    style.ItemSpacing       = {8.0f, 6.0f};
    style.ItemInnerSpacing  = {6.0f, 4.0f};
    style.IndentSpacing     = 20.0f;
    style.ScrollbarSize     = 12.0f;
    style.GrabMinSize       = 10.0f;

    style.WindowBorderSize  = 1.0f;
    style.ChildBorderSize   = 1.0f;
    style.PopupBorderSize   = 1.0f;
    style.FrameBorderSize   = 1.0f;
    style.TabBarBorderSize  = 0.0f;

    style.WindowRounding    = 6.0f;
    style.ChildRounding     = 6.0f;
    style.FrameRounding     = 4.0f;
    style.PopupRounding     = 6.0f;
    style.ScrollbarRounding = 6.0f;
    style.GrabRounding      = 4.0f;
    style.TabRounding       = 4.0f;

    style.WindowTitleAlign  = {0.0f, 0.5f};
    style.SeparatorTextBorderSize = 1.0f;
    style.SeparatorTextPadding    = {16.0f, 4.0f};

    style.AntiAliasedLines      = true;
    style.AntiAliasedLinesUseTex = true;
    style.AntiAliasedFill       = true;

    // --- Colors -------------------------------------------------------------
    ImVec4 *c = style.Colors;

    c[ImGuiCol_Text]                  = kText;
    c[ImGuiCol_TextDisabled]          = kTextDim;

    c[ImGuiCol_WindowBg]              = kBg;
    c[ImGuiCol_ChildBg]               = kBgDark;
    c[ImGuiCol_PopupBg]               = withAlpha(kBgDark, 0.98f);
    c[ImGuiCol_Border]                = kBorder;
    c[ImGuiCol_BorderShadow]          = {0.0f, 0.0f, 0.0f, 0.0f};

    c[ImGuiCol_FrameBg]               = kBgLight;
    c[ImGuiCol_FrameBgHovered]        = kBgHovered;
    c[ImGuiCol_FrameBgActive]         = withAlpha(kAccent, 0.35f);

    c[ImGuiCol_TitleBg]               = kTitleDim;
    c[ImGuiCol_TitleBgActive]         = kTitle;
    c[ImGuiCol_TitleBgCollapsed]      = withAlpha(kTitleDim, 0.85f);
    c[ImGuiCol_MenuBarBg]             = kTitleDim;

    c[ImGuiCol_ScrollbarBg]           = {0.0f, 0.0f, 0.0f, 0.0f};
    c[ImGuiCol_ScrollbarGrab]         = kBgHovered;
    c[ImGuiCol_ScrollbarGrabHovered]  = withAlpha(kAccent, 0.60f);
    c[ImGuiCol_ScrollbarGrabActive]   = kAccent;

    c[ImGuiCol_CheckMark]             = kAccentHi;
    c[ImGuiCol_SliderGrab]            = kAccent;
    c[ImGuiCol_SliderGrabActive]      = kAccentHi;

    c[ImGuiCol_Button]                = kBgLight;
    c[ImGuiCol_ButtonHovered]         = kBgHovered;
    c[ImGuiCol_ButtonActive]          = kAccentLo;

    c[ImGuiCol_Header]                = withAlpha(kBgLight, 0.80f);
    c[ImGuiCol_HeaderHovered]         = kBgHovered;
    c[ImGuiCol_HeaderActive]          = withAlpha(kAccent, 0.45f);

    c[ImGuiCol_Separator]             = kBorder;
    c[ImGuiCol_SeparatorHovered]      = withAlpha(kAccent, 0.70f);
    c[ImGuiCol_SeparatorActive]       = kAccent;

    c[ImGuiCol_ResizeGrip]            = withAlpha(kTextDim, 0.25f);
    c[ImGuiCol_ResizeGripHovered]     = withAlpha(kAccent, 0.70f);
    c[ImGuiCol_ResizeGripActive]      = kAccent;

    c[ImGuiCol_Tab]                   = kTitleDim;
    c[ImGuiCol_TabHovered]            = kAccentLo;
    c[ImGuiCol_TabSelected]           = kBgLight;
    c[ImGuiCol_TabSelectedOverline]   = kAccent;
    c[ImGuiCol_TabDimmed]             = withAlpha(kTitleDim, 0.70f);
    c[ImGuiCol_TabDimmedSelected]     = withAlpha(kBgLight, 0.80f);
    c[ImGuiCol_TabDimmedSelectedOverline] = withAlpha(kAccent, 0.40f);

    c[ImGuiCol_PlotLines]             = {0.95f, 0.75f, 0.35f, 1.00f};
    c[ImGuiCol_PlotLinesHovered]      = {1.00f, 0.95f, 0.60f, 1.00f};
    c[ImGuiCol_PlotHistogram]         = kAccent;
    c[ImGuiCol_PlotHistogramHovered]  = kAccentHi;

    c[ImGuiCol_TableHeaderBg]         = kBgLight;
    c[ImGuiCol_TableBorderStrong]     = kBorder;
    c[ImGuiCol_TableBorderLight]      = withAlpha(kBorder, 0.50f);
    c[ImGuiCol_TableRowBg]            = {0.0f, 0.0f, 0.0f, 0.0f};
    c[ImGuiCol_TableRowBgAlt]         = withAlpha(kText, 0.05f);

    c[ImGuiCol_TextSelectedBg]        = withAlpha(kAccent, 0.35f);
    c[ImGuiCol_DragDropTarget]        = kAccentHi;
    c[ImGuiCol_NavCursor]             = kAccentHi;
    c[ImGuiCol_NavWindowingHighlight] = withAlpha(kText, 0.70f);
    c[ImGuiCol_NavWindowingDimBg]     = {0.10f, 0.10f, 0.12f, 0.40f};
    c[ImGuiCol_ModalWindowDimBg]      = {0.05f, 0.05f, 0.06f, 0.60f};
}

} // namespace lr
