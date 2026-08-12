#pragma once

#include "frame_source.hpp"

#include <opencv2/core.hpp>
#include <torch/script.h>
#include <torch/torch.h>

#include <filesystem>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace omnivggt::observer {

struct InferenceOptions {
    std::string model;
    // Independent three-image batch graph.  It must be exported with
    // [B=3,S=1] and is never used as a native S=3 sequence graph.
    std::string group_model;
    int group_width = 406;
    int group_height = 252;
    int group_stride = 1;
    bool group_mode = false;
    bool group_observation_mode = false;
    // Python uses one image for the anchor frame and a two-image
    // anchor/current window for every subsequent frame.  Keep both exported
    // TorchScript signatures available instead of silently reusing the
    // single-frame graph for the stream.
    std::string pair_model;
    // Optional directory containing one TorchScript pair graph per actual
    // Python ROI bucket, named like omnivggt_s2_672x700_*.pt.  A traced graph
    // records the output height/width it was exported with, so reusing a
    // 700x700 graph for a 700x672 input silently returns the wrong geometry.
    std::filesystem::path pair_model_dir;
    // Optional fast path for a single fixed-shape pair graph.  The ROI is
    // resized without changing its aspect ratio and replicated at the model
    // border; the padded pixels are excluded again before projection to the
    // canvas.  This avoids reloading a multi-gigabyte TorchScript graph for
    // every dynamic ROI shape while retaining the Python crop geometry.
    bool pair_letterbox = false;
    std::string device = "cuda";
    // Match the Python live backend on CUDA.  The float32 TorchScript graph
    // exceeds the practical memory budget of the target laptop GPU and can
    // fall back to paging, turning a ~1 s inference into minutes.
    std::string dtype = "bfloat16";
    int width = 518;
    int height = 518;
    // ``width``/``height`` are the maximum model bucket.  The aligned-canvas
    // state can be larger because Python pads the matching image before it
    // starts adding views outside the first frame.
    int canvas_width = 0;
    int canvas_height = 0;
    int first_model_width = 0;
    int first_model_height = 0;
    double image_l1_thr = 12.0 / 255.0;
    double no_change_ratio = 0.001;
    double scene_jump_ratio = 0.35;
    double min_conf = 0.1;
    int dilate_ksize = 3;
    bool save_debug = false;
    std::filesystem::path debug_dir;
};

struct InferenceMetrics {
    FrameSeq frame_seq = 0;
    std::string image;
    double read_ms = 0.0;
    double align2d_ms = 0.0;
    double diff_ms = 0.0;
    double model_ms = 0.0;
    double depth_align_ms = 0.0;
    double patch_ms = 0.0;
    double total_ms = 0.0;
    double changed_ratio = 0.0;
    double photometric_changed_ratio = 0.0;
    double support_changed_ratio = 0.0;
    std::uint32_t changed_point_count = 0;
    std::uint32_t valid_point_count = 0;
    int roi_width = 0;
    int roi_height = 0;
    int model_input_width = 0;
    int model_input_height = 0;
    int homography_inliers = 0;
    double homography_error_px = -1.0;
    bool skipped_model = false;
    std::string fallback;
    int group_size = 1;
    int group_stride = 1;
    int group_anchor_index = 0;
    int forward_calls = 0;
    int forward_batch_size = 1;
    int forward_sequence_size = 1;
    int group_fused_sources = 1;
    int group_rejected_sources = 0;
    double group_max_depth_residual = -1.0;

    std::string csv_line() const;
};

struct CandidateCommit {
    FrameRecord frame;
    CandidatePatch patch;
    InferenceMetrics metrics;
    bool has_patch = false;
};

struct PreparedInput {
    RawFrame raw;
    FrameSeq frame_seq = 0;
    double read_ms = 0.0;
    std::vector<std::string> image_names;
    std::filesystem::path path;
    cv::Mat rgb_u8;
    cv::Mat rgb_f;
    cv::Mat match_rgb_u8;
    cv::Mat match_rgb_f;
    cv::Mat support;
    bool has_group = false;
    std::vector<cv::Mat> group_warped_rgb_f;
    std::vector<cv::Mat> group_valid_warp;
    cv::Mat group_fused_rgb_f;
    cv::Mat group_union_valid;
    bool has_observation_group = false;
};

class InferenceEngine {
public:
    explicit InferenceEngine(InferenceOptions options);

    CandidateCommit process_prepared(const PreparedInput& prepared, const CanvasState& state);
    CandidateCommit process(const RawFrame& raw, const CanvasState& state);

private:
    struct FrameImage {
        std::filesystem::path path;
        cv::Mat rgb_u8;
        cv::Mat rgb_f;
        cv::Mat match_rgb_u8;
        cv::Mat match_rgb_f;
        cv::Mat support;
    };

    struct Prediction {
        cv::Mat depth;
        cv::Mat confidence;
        // Keep the complete OmniVGGT point-head result available even though
        // the aligned-canvas fusion consumes the depth/confidence maps.  The
        // Python live replay uses the same depth canvas for its final PLY,
        // while retaining these arrays prevents a silent depth-only path.
        cv::Mat world_points;
        cv::Mat world_points_confidence;
        float fov_h = 1.2f;
        float fov_w = 1.2f;
    };

    struct PreparedGroup {
        std::vector<cv::Mat> warped_rgb_f;
        std::vector<cv::Mat> valid_warp;
        cv::Mat fused_rgb_f;
        cv::Mat union_valid;
    };

    InferenceOptions options_;
    torch::Device device_ = torch::Device(torch::kCPU);
    torch::ScalarType dtype_ = torch::kFloat32;
    torch::jit::script::Module module_;
    torch::jit::script::Module group_module_;
    torch::jit::script::Module pair_module_;
    // Dynamic pair artifacts are large (about 2.4 GB each in the BF16
    // observer build). Keep a small GPU LRU so switching between the common
    // ROI buckets does not deserialize and upload the graph on every frame.
    std::map<std::pair<int, int>, torch::jit::script::Module> pair_module_cache_;
    std::vector<std::pair<int, int>> pair_module_lru_;
    std::vector<std::pair<int, int>> dynamic_pair_shapes_;
    bool has_pair_module_ = false;
    bool has_group_module_ = false;
    bool pair_module_loaded_ = false;
    std::pair<int, int> pair_module_shape_ = {-1, -1};
    std::filesystem::path pair_model_dir_;
    bool single_model_released_ = false;
    // Python keeps the aligned float canvas and the original float anchor in
    // memory.  CanvasState serializes colors as RGBA bytes for the protocol;
    // retain these live float buffers inside the external inference engine so
    // matching and seam color transfer do not quantize every frame.
    mutable cv::Mat anchor_rgb_float_;
    mutable cv::Mat live_rgb_float_;
    // Once a grouped sliding window repairs an old-support hole, later
    // windows must not overwrite that protected strip with a new ROI colour
    // layer.  This mask is in-process only; the versioned Canvas remains the
    // authoritative persisted state.
    mutable cv::Mat group_gap_protected_;
    // A fixed-anchor SIFT/RANSAC can fail on a late frame even when the
    // inter-frame motion is small. Keep the last accepted canvas transform so
    // one numerical feature-detection miss does not discard the whole frame.
    mutable cv::Mat last_homography_;

    FrameImage load_frame(const std::filesystem::path& path) const;
    Prediction run_model(const std::vector<cv::Mat>& rgb_u8, int output_frame_index);
    std::vector<Prediction> run_group_model(const std::vector<cv::Mat>& rgb_u8);
    Prediction fuse_group_predictions(
        const std::vector<Prediction>& predictions,
        int anchor_index,
        InferenceMetrics& metrics) const;
    void activate_dynamic_pair_module(
        const std::filesystem::path& bucket_path,
        const std::pair<int, int>& shape);
    std::pair<int, int> dynamic_pair_shape_for_target(
        const std::pair<int, int>& target) const;
    void release_single_model_after_first_frame();
    cv::Mat estimate_pair_homography(const FrameImage& source, const FrameImage& target) const;

    static cv::Mat gray_u8(const cv::Mat& rgb_u8);
    static cv::Mat translation_h(double dx, double dy);
    static cv::Mat warp_like(const cv::Mat& source, const cv::Mat& homography, cv::Size size, int interpolation);
    static cv::Mat dilate_mask(const cv::Mat& mask, int kernel_size);
    static cv::Mat state_rgb_float(const CanvasState& state);
    static cv::Mat anchor_rgb_float(const CanvasState& state);
    static cv::Mat state_mask(const std::vector<std::uint8_t>& values, int width, int height);
    static std::uint32_t color_at(const cv::Mat& rgb_u8, int x, int y);
    static std::vector<std::uint32_t> pack_image(const cv::Mat& rgb_u8);

    cv::Mat estimate_homography(
        const FrameImage& current,
        const CanvasState& state,
        InferenceMetrics& metrics) const;
    cv::Mat compute_change_mask(
        const FrameImage& current,
        const cv::Mat& homography,
        const CanvasState& state,
        InferenceMetrics& metrics,
        cv::Mat& warped_rgb_f,
        cv::Mat& valid_warp,
        cv::Mat& support_change,
        cv::Mat& photometric_change) const;
    cv::Mat align_depth_to_canvas(
        const cv::Mat& depth,
        const cv::Mat& confidence,
        const cv::Mat& valid_warp,
        const CanvasState& state,
        const cv::Mat& anchor_mask) const;
    void save_debug_images(
        const FrameImage& frame,
        const cv::Mat& warped_rgb_f,
        const cv::Mat& change_mask,
        const cv::Mat& valid_warp,
        const cv::Mat& anchor_ring,
        const cv::Mat& update_mask,
        const cv::Mat& color_bridge_mask,
        const cv::Mat& fused_rgb) const;
    CandidateCommit process_impl(
        const RawFrame& raw,
        const CanvasState& state,
        const FrameImage& frame,
        const PreparedGroup* group,
        double read_ms);
    CandidateCommit process_group(const RawFrame& raw, const CanvasState& state);
};

class FramePreprocessor {
public:
    explicit FramePreprocessor(InferenceOptions options);

    PreparedInput prepare(const RawFrame& raw) const;

private:
    InferenceOptions options_;

    struct FrameImage {
        std::filesystem::path path;
        cv::Mat rgb_u8;
        cv::Mat rgb_f;
        cv::Mat match_rgb_u8;
        cv::Mat match_rgb_f;
        cv::Mat support;
    };

    FrameImage load_frame(const std::filesystem::path& path) const;
    cv::Mat estimate_pair_homography(const FrameImage& source, const FrameImage& target) const;
    static cv::Mat gray_u8(const cv::Mat& rgb_u8);
    static cv::Mat warp_like(const cv::Mat& source, const cv::Mat& homography, cv::Size size, int interpolation);
};

}  // namespace omnivggt::observer
