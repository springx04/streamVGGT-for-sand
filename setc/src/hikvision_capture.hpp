#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace omnivggt::hikvision {

// MVS exposes device-specific GenICam nodes rather than one universal
// "get intrinsic matrix" call.  Fill these names only when the connected
// device exposes calibrated camera nodes.  For ordinary 2-D industrial
// cameras, use Hikvision/MVS or OpenCV to calibrate once and pass the
// resulting YAML/XML file through HikvisionCameraOptions::calibration_file.
struct IntrinsicNodeNames {
    std::string fx;
    std::string fy;
    std::string cx;
    std::string cy;
    std::string k1;
    std::string k2;
    std::string p1;
    std::string p2;
    std::string k3;

    bool has_matrix_nodes() const noexcept;
    bool has_any_distortion_node() const noexcept;
    bool has_complete_distortion_nodes() const noexcept;
};

struct CameraIntrinsics {
    int image_width = 0;
    int image_height = 0;
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    std::vector<double> distortion;
    std::string source;

    bool valid() const noexcept;
    cv::Mat camera_matrix() const;
    cv::Mat distortion_coefficients() const;
};

struct HikvisionCameraOptions {
    std::size_t device_index = 0U;
    int timeout_ms = 1000;
    bool software_trigger = false;
    std::string camera_name;
    std::filesystem::path calibration_file;
    IntrinsicNodeNames intrinsic_nodes;
};

struct CapturedFrame {
    std::uint64_t capture_index = 0;
    std::uint64_t device_frame_number = 0;
    std::uint64_t host_timestamp_ns = 0;
    std::string camera_id;
    cv::Mat bgr;
    CameraIntrinsics intrinsics;
};

struct CaptureOutputOptions {
    std::filesystem::path output_dir;
    int png_compression = 3;
};

class HikvisionCamera {
public:
    explicit HikvisionCamera(HikvisionCameraOptions options = {});
    ~HikvisionCamera();

    HikvisionCamera(const HikvisionCamera&) = delete;
    HikvisionCamera& operator=(const HikvisionCamera&) = delete;

    void open();
    void start();
    void stop();
    void close() noexcept;

    bool is_open() const noexcept;
    bool is_grabbing() const noexcept;
    const std::string& camera_id() const noexcept;

    // Returns an empty optional for a normal SDK timeout.  Other SDK errors
    // throw std::runtime_error so a disconnected or malformed stream is not
    // silently recorded as a valid frame.
    std::optional<CapturedFrame> grab();

    // Public for applications that want to inspect the K matrix before the
    // first frame.  The dimensions must be the actual image dimensions that
    // will be written to disk.
    CameraIntrinsics read_intrinsics(int image_width, int image_height) const;

private:
    HikvisionCameraOptions options_;
    void* handle_ = nullptr;
    bool device_open_ = false;
    bool grabbing_ = false;
    std::uint64_t capture_index_ = 0;
    std::string camera_id_;
    std::optional<CameraIntrinsics> calibration_intrinsics_;

    static CameraIntrinsics load_calibration_file(const std::filesystem::path& path);
    CameraIntrinsics read_intrinsics_from_nodes(int image_width, int image_height) const;
    void validate_intrinsics_for_image(CameraIntrinsics& intrinsics, int image_width, int image_height) const;
};

// Writes images/frame_XXXXXXXX.png, cameras/frame_XXXXXXXX.json, and appends
// one row to frames.csv.  The JSON contains the same image stem and an
// explicit 3x3 "intrinsic" array, so it can be consumed without MVS later.
void save_captured_frame(const CapturedFrame& frame, const CaptureOutputOptions& options);

using CaptureCallback = std::function<bool(const CapturedFrame&)>;

// max_frames == 0 means keep acquiring until the callback returns false or
// the caller terminates the process.
void capture_to_directory(
    HikvisionCamera& camera,
    const CaptureOutputOptions& output,
    std::uint64_t max_frames = 0U,
    const CaptureCallback& callback = {});

}  // namespace omnivggt::hikvision
