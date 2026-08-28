#pragma once

#include "stream_types.hpp"

#include <cstdint>
#include <vector>

namespace omnivggt::observer {

struct ExportPoint {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    std::uint8_t r = 0;
    std::uint8_t g = 0;
    std::uint8_t b = 0;
    bool changed = false;
};

// Build the bounded plane-residual depth representation used by the live
// OpenGL viewer.  The returned vector is display-only; it never mutates the
// authoritative streaming CanvasState.
std::vector<float> visual_depth_canvas(const CanvasState& state);

// Return a GUI-only CanvasState with a bounded plane-residual depth and
// conservative nearest-neighbour repair inside the camera support mask.
CanvasState visual_canvas_state(const CanvasState& state);

// Shared with the interactive viewer and replay PLY writer.  This mirrors
// Python's export_canvas_pointcloud; it is deliberately export-only and never
// mutates the authoritative streaming CanvasState.
std::vector<ExportPoint> export_clean_canvas_points(
    const CanvasState& state,
    FrameSeq changed_frame = 0);

}  // namespace omnivggt::observer
