#pragma once

#include "stream_types.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace omnivggt::observer {

// The live protocol has no camera capture timestamp or hardware-trigger ID.
// GroupWorldFusion therefore stabilizes one complete GUI flush cycle, not
// hardware exposure simultaneity.  The three views must already be in model
// slot order when they reach this class. The real GUI derives XY from a
// physical slot row/column and cannot represent multiple Z values at exactly
// one XY. The 2x2 layer block below is therefore an explicit compromise:
// floor and object surfaces can be separated by up to one GUI grid pixel.
struct GroupWorldView {
    cv::Mat world_points;       // HxW CV_32FC3, directly from Prediction.world_points
    cv::Mat world_confidence;   // HxW CV_32FC1, directly from the point head
    cv::Mat rgb;                // HxW CV_32FC3, the corresponding model input pixels
    cv::Vec3f translation = cv::Vec3f(0.0f, 0.0f, 0.0f);
    cv::Vec4f quaternion = cv::Vec4f(0.0f, 0.0f, 0.0f, 1.0f);  // xyzw
    bool has_pose = false;
};

struct FusedSlot {
    std::uint32_t slot_id = 0;
    float depth = 0.0f;       // display Z; floor is always zero
    float confidence = 0.0f;
    std::uint32_t rgba = 0;
    bool floor = false;
};

struct GroupWorldFusionResult {
    bool accepted = false;
    std::string rejection_reason;
    std::vector<FusedSlot> slots;              // desired valid slots only
    std::vector<std::uint32_t> occupied_slots; // floor + current object slots
    float scene_scale = 0.0f;
    float floor_band = 0.0f;
};

class GroupWorldFusion {
public:
    GroupWorldFusion() = default;

    GroupWorldFusionResult fuse(
        const std::array<GroupWorldView, 3>& views,
        const CanvasState& state);

    void reset();
    bool reference_initialized() const noexcept { return reference_initialized_; }

private:
    bool reference_initialized_ = false;
    std::array<cv::Vec3f, 3> reference_centers_{};
    cv::Vec3f plane_origin_ = cv::Vec3f(0.0f, 0.0f, 0.0f);
    cv::Vec3f plane_normal_ = cv::Vec3f(0.0f, 0.0f, 1.0f);
    cv::Vec3f axis_x_ = cv::Vec3f(1.0f, 0.0f, 0.0f);
    cv::Vec3f axis_y_ = cv::Vec3f(0.0f, 1.0f, 0.0f);
    float u_min_ = 0.0f;
    float u_max_ = 0.0f;
    float v_min_ = 0.0f;
    float v_max_ = 0.0f;
    float center_u_ = 0.0f;
    float center_v_ = 0.0f;
    float display_scale_ = 1.0f;
    float scene_scale_ = 1.0f;
    float floor_band_ = 0.01f;
    float max_object_height_ = 1.0f;
    float global_color_gain_ = 1.0f;
    // Indexed by model view, then RGB channel. View 1 is the fixed anchor.
    std::array<std::array<float, 3>, 3> color_gain_{{
        {{1.0f, 1.0f, 1.0f}},
        {{1.0f, 1.0f, 1.0f}},
        {{1.0f, 1.0f, 1.0f}}}};

    int canvas_width_ = 0;
    int canvas_height_ = 0;
    int logical_width_ = 0;
    int logical_height_ = 0;
    std::vector<FusedSlot> floor_cells_;
    std::vector<std::uint8_t> floor_cell_valid_;
};

}  // namespace omnivggt::observer
