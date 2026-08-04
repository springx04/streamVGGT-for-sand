#include "protocol.hpp"
#include "pointcloud_export.hpp"
#include "tcp_transport.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

using namespace omnivggt::observer;

namespace {

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

Vec3 operator+(const Vec3 lhs, const Vec3 rhs) { return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z}; }
Vec3 operator-(const Vec3 lhs, const Vec3 rhs) { return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z}; }
Vec3 operator*(const Vec3 lhs, const float scalar) { return {lhs.x * scalar, lhs.y * scalar, lhs.z * scalar}; }
float dot(const Vec3 lhs, const Vec3 rhs) { return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z; }
Vec3 cross(const Vec3 lhs, const Vec3 rhs) {
    return {lhs.y * rhs.z - lhs.z * rhs.y, lhs.z * rhs.x - lhs.x * rhs.z, lhs.x * rhs.y - lhs.y * rhs.x};
}
float length(const Vec3 value) { return std::sqrt(std::max(0.0f, dot(value, value))); }
Vec3 normalized(const Vec3 value) {
    const float norm = length(value);
    return norm > 1e-6f ? value * (1.0f / norm) : Vec3{0.0f, 0.0f, 1.0f};
}

// Keep the camera model identical to the Python viewer.  The Python window
// projects the cleaned canonical canvas with an orthographic orbit camera;
// using the same state here is both faster during drag and avoids turning a
// nearly planar depth field into a tall perspective object.
struct OrbitCamera {
    float yaw = -0.62f;
    float pitch = 0.34f;
    float zoom_factor = 1.0f;
    float pan_x = 0.0f;
    float pan_y = 0.0f;

    void reset() {
        yaw = -0.62f;
        pitch = 0.34f;
        zoom_factor = 1.0f;
        pan_x = 0.0f;
        pan_y = 0.0f;
    }

    void rotate(const float dx, const float dy) {
        yaw += dx;
        pitch = std::clamp(pitch + dy, -1.45f, 1.45f);
    }

    void zoom(const float amount) {
        zoom_factor = std::clamp(zoom_factor * amount, 0.25f, 4.0f);
    }

    void pan(const float dx, const float dy) {
        pan_x = std::clamp(pan_x + dx, -500.0f, 500.0f);
        pan_y = std::clamp(pan_y + dy, -500.0f, 500.0f);
    }
};

constexpr int kPanelWidth = 760;
constexpr int kPanelHeight = 540;
constexpr int kWindowWidth = kPanelWidth * 2;
constexpr int kWindowHeight = 620;

struct CloudPoint {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    cv::Vec3b bgr{0, 0, 0};
    bool changed = false;
};

double median_value(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    const std::size_t middle = values.size() / 2U;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle), values.end());
    const double upper = values[middle];
    if ((values.size() % 2U) != 0U) {
        return upper;
    }
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle - 1U), values.end());
    return 0.5 * (upper + values[middle - 1U]);
}

std::array<double, 6> fit_surface(const std::vector<CloudPoint>& points, const std::vector<std::uint8_t>& keep) {
    cv::Mat normal = cv::Mat::zeros(6, 6, CV_64F);
    cv::Mat rhs = cv::Mat::zeros(6, 1, CV_64F);
    for (std::size_t index = 0; index < points.size(); ++index) {
        if (keep[index] == 0U) {
            continue;
        }
        const CloudPoint& point = points[index];
        const double basis[6] = {
            point.x, point.y, point.x * point.x, point.y * point.y, point.x * point.y, 1.0};
        for (int row = 0; row < 6; ++row) {
            rhs.at<double>(row, 0) += basis[row] * static_cast<double>(point.z);
            for (int column = 0; column < 6; ++column) {
                normal.at<double>(row, column) += basis[row] * basis[column];
            }
        }
    }
    cv::Mat solution;
    if (!cv::solve(normal, rhs, solution, cv::DECOMP_SVD) || solution.rows != 6) {
        return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    }
    std::array<double, 6> result{};
    for (int index = 0; index < 6; ++index) {
        result[static_cast<std::size_t>(index)] = solution.at<double>(index, 0);
    }
    return result;
}

double surface_value(const std::array<double, 6>& coeff, const CloudPoint& point) {
    return coeff[0] * point.x + coeff[1] * point.y
        + coeff[2] * point.x * point.x + coeff[3] * point.y * point.y
        + coeff[4] * point.x * point.y + coeff[5];
}

class LocalState {
public:
    void set_snapshot(const Snapshot& snapshot) {
        state_ = snapshot.state;
        frame_ = state_.last_frame;
        ghost_.clear();
        changed_frame_ = 0;
    }

    void apply(const PointCloudDelta& delta) {
        apply_delta_forward(state_, delta);
        frame_ = delta.frame_seq;
        ghost_ = delta.changes;
        changed_frame_ = delta.frame_seq;
    }

    const CanvasState& state() const noexcept { return state_; }
    CanvasState& state() noexcept { return state_; }
    FrameSeq frame() const noexcept { return frame_; }
    const std::vector<SlotDelta>& ghost() const noexcept { return ghost_; }
    FrameSeq changed_frame() const noexcept { return changed_frame_; }
    bool initialized() const noexcept { return state_.shape_valid() && state_.initialized; }

private:
    CanvasState state_;
    FrameSeq frame_ = 0;
    FrameSeq changed_frame_ = 0;
    std::vector<SlotDelta> ghost_;
};

class ViewerModel {
public:
    struct SeekRequest {
        std::uint64_t generation = 0;
        FrameSeq target_frame = 0;
    };

    void on_packet(const Packet& packet) {
        std::lock_guard<std::mutex> lock(mutex_);
        try {
            switch (packet.type) {
            case MessageType::Hello: {
                const HelloMessage hello = decode_hello(packet);
                width_ = hello.width;
                height_ = hello.height;
                live_head_frame_ = std::max(live_head_frame_, hello.live_head_frame);
                received_head_frame_ = std::max(received_head_frame_, hello.live_head_frame);
                live_version_ = hello.commit_version;
                run_name_ = hello.run_name;
                break;
            }
            case MessageType::Snapshot: {
                live_.set_snapshot(decode_snapshot(packet));
                // A snapshot is authoritative.  It is also the only packet
                // that may replace the complete live state, so it ends a
                // pending inference display even when the stream was
                // resynchronised while a frame was running.
                inference_pending_ = false;
                pending_frame_ = 0;
                live_version_ = live_.state().version;
                live_head_frame_ = std::max(live_head_frame_, live_.frame());
                received_head_frame_ = std::max(received_head_frame_, live_head_frame_);
                if (follow_live_) {
                    playback_ = live_;
                    view_cursor_frame_ = live_head_frame_;
                    view_version_ = playback_.state().version;
                }
                break;
            }
            case MessageType::FrameStatus: {
                const FrameRecord record = decode_frame_status(packet);
                received_head_frame_ = std::max(received_head_frame_, record.frame_seq);
                last_status_ = std::string(frame_status_name(record.status));
                // The server publishes Received before inference starts and
                // publishes Committed immediately before Delta.  Neither
                // packet contains a new visual state.  Advancing the live
                // cursor here made the viewer render the previous committed
                // canvas as if it were the not-yet-inferred frame.
                switch (record.status) {
                case FrameStatus::Received:
                case FrameStatus::Aligning:
                case FrameStatus::DiffReady:
                case FrameStatus::Inferencing:
                case FrameStatus::Committed:
                    inference_pending_ = true;
                    pending_frame_ = record.frame_seq;
                    break;
                case FrameStatus::NoChange:
                case FrameStatus::Coalesced:
                case FrameStatus::Failed:
                    inference_pending_ = false;
                    pending_frame_ = 0;
                    live_head_frame_ = std::max(live_head_frame_, record.frame_seq);
                    if (follow_live_) {
                        view_cursor_frame_ = live_head_frame_;
                    }
                    break;
                }
                break;
            }
            case MessageType::Delta: {
                const PointCloudDelta delta = decode_delta(packet);
                live_.apply(delta);
                live_version_ = live_.state().version;
                live_head_frame_ = std::max(live_head_frame_, delta.frame_seq);
                received_head_frame_ = std::max(received_head_frame_, delta.frame_seq);
                inference_pending_ = false;
                pending_frame_ = 0;
                if (follow_live_) {
                    playback_ = live_;
                    view_cursor_frame_ = live_head_frame_;
                    view_version_ = playback_.state().version;
                }
                break;
            }
            case MessageType::LiveHead: {
                const auto head = decode_live_head(packet);
                received_head_frame_ = std::max(received_head_frame_, head.first);
                live_version_ = std::max(live_version_, head.second);
                // LiveHead is sent both when a raw frame is received and
                // after its committed delta.  Do not expose the former as a
                // replay frame until the corresponding terminal status or
                // delta arrives.
                if (!inference_pending_) {
                    live_head_frame_ = std::max(live_head_frame_, head.first);
                    if (follow_live_) {
                        view_cursor_frame_ = live_head_frame_;
                    }
                }
                break;
            }
            case MessageType::ReplayBegin: {
                const ReplayBeginMessage begin = decode_replay_begin(packet);
                if (begin.generation != seek_generation_) {
                    break;
                }
                playback_.set_snapshot(begin.snapshot);
                view_version_ = playback_.state().version;
                playback_target_frame_ = begin.target_frame;
                playback_target_version_ = begin.target_version;
                playback_loading_ = true;
                break;
            }
            case MessageType::ReplayDelta: {
                std::uint64_t generation = 0;
                const PointCloudDelta delta = decode_delta(packet, &generation);
                if (generation != seek_generation_) {
                    break;
                }
                playback_.apply(delta);
                view_version_ = playback_.state().version;
                break;
            }
            case MessageType::ReplayEnd: {
                const ReplayEndMessage end = decode_replay_end(packet);
                if (end.generation != seek_generation_) {
                    break;
                }
                view_cursor_frame_ = end.target_frame;
                view_version_ = end.target_version;
                playback_loading_ = false;
                break;
            }
            case MessageType::ResyncRequired:
                last_status_ = "ResyncRequired v" + std::to_string(decode_resync_required(packet));
                break;
            case MessageType::Error:
                last_status_ = decode_error(packet);
                playback_loading_ = false;
                break;
            default:
                break;
            }
        } catch (const std::exception& error) {
            last_status_ = std::string("protocol error: ") + error.what();
        }
    }

    std::optional<SeekRequest> seek(const FrameSeq target) {
        std::lock_guard<std::mutex> lock(mutex_);
        follow_live_ = false;
        playback_running_ = false;
        view_cursor_frame_ = target;
        ++seek_generation_;
        return SeekRequest{seek_generation_, target};
    }

    void follow_live() {
        std::lock_guard<std::mutex> lock(mutex_);
        follow_live_ = true;
        playback_running_ = false;
        playback_ = live_;
        view_cursor_frame_ = live_head_frame_;
        view_version_ = playback_.state().version;
    }

    std::optional<SeekRequest> toggle_playback() {
        std::lock_guard<std::mutex> lock(mutex_);
        playback_running_ = !playback_running_;
        follow_live_ = false;
        if (playback_running_) {
            ++seek_generation_;
            return SeekRequest{seek_generation_, std::min(view_cursor_frame_ + 1U, live_head_frame_)};
        }
        return std::nullopt;
    }

    std::optional<SeekRequest> tick_playback() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!playback_running_ || playback_loading_ || live_head_frame_ == 0U) {
            return std::nullopt;
        }
        if (view_cursor_frame_ >= live_head_frame_) {
            playback_running_ = false;
            return std::nullopt;
        }
        ++view_cursor_frame_;
        ++seek_generation_;
        playback_loading_ = true;
        return SeekRequest{seek_generation_, view_cursor_frame_};
    }

    void adjust_fps(const float delta) {
        std::lock_guard<std::mutex> lock(mutex_);
        playback_fps_ = std::clamp(playback_fps_ + delta, 1.0f, 60.0f);
    }

    float playback_fps() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return playback_fps_;
    }

    FrameSeq live_head() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return live_head_frame_;
    }

    FrameSeq view_cursor() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return follow_live_ ? live_head_frame_ : view_cursor_frame_;
    }

    std::string status() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return last_status_;
    }

    bool follow_live_mode() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return follow_live_;
    }

    bool playback_running() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return playback_running_;
    }

    void set_display_max_points(const std::size_t maximum) {
        std::lock_guard<std::mutex> lock(mutex_);
        display_max_points_ = maximum;
        invalidate_cache();
    }

    void render(cv::Mat& output, const OrbitCamera& camera) const {
        std::lock_guard<std::mutex> lock(mutex_);
        output = cv::Mat(kWindowHeight, kWindowWidth, CV_8UC3, cv::Scalar(24, 24, 24));
        const LocalState& active = follow_live_ ? live_ : playback_;
        if (!active.initialized()) {
            cv::putText(output, "Waiting for stream snapshot...", {35, 70}, cv::FONT_HERSHEY_SIMPLEX, 0.9,
                cv::Scalar(220, 220, 220), 2, cv::LINE_AA);
            return;
        }

        rebuild_cache_if_needed(active);
        const cv::Mat cloud = render_cloud_panel(camera);
        cached_left_.copyTo(output(cv::Rect(0, 0, kPanelWidth, kPanelHeight)));
        cloud.copyTo(output(cv::Rect(kPanelWidth, 0, kPanelWidth, kPanelHeight)));

        const FrameSeq maximum = std::max<FrameSeq>(live_head_frame_, 1U);
        const FrameSeq cursor = follow_live_ ? live_head_frame_ : view_cursor_frame_;
        std::ostringstream header;
        header << "frame " << cursor << "/" << maximum
               << "  points " << cached_points_.size()
               << "  clean " << cached_points_.size()
               << "  delta " << cached_delta_count_ << " px";
        cv::rectangle(output, {0, 0}, {output.cols - 1, 36}, cv::Scalar(18, 18, 18), -1);
        cv::putText(output, header.str(), {14, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.58,
            cv::Scalar(240, 240, 240), 1, cv::LINE_AA);

        // Python's synchronous replay keeps the last committed frame on
        // screen while the next stream.push() is running.  The C++ server is
        // asynchronous, so retain the same visual contract explicitly: the
        // live state remains the previous committed state until Delta arrives.
        if (follow_live_ && inference_pending_) {
            std::ostringstream pending;
            pending << "inference frame " << pending_frame_ << "...";
            cv::putText(output, pending.str(), {output.cols - 220, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(60, 220, 255), 1, cv::LINE_AA);
        }

        const std::string footer =
            "mouse left-drag: rotate | right-drag: pan | wheel: zoom | r: reset view | "
            "red outline: committed change | drag frame / a,d: replay | q/Esc: close";
        cv::putText(output, footer, {14, output.rows - 12}, cv::FONT_HERSHEY_SIMPLEX, 0.48,
            cv::Scalar(175, 205, 220), 1, cv::LINE_AA);
        if (playback_loading_) {
            cv::putText(output, "seeking...", {output.cols - 135, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(60, 220, 255), 1, cv::LINE_AA);
        }
    }

    void handle_pan(const float dx, const float dy, OrbitCamera& camera) const {
        camera.pan(dx, dy);
    }

private:
    static cv::Mat render_canvas_panel(const CanvasState& state, const FrameSeq changed_frame) {
        if (!state.shape_valid()) {
            return cv::Mat(kPanelHeight, kPanelWidth, CV_8UC3, cv::Scalar(24, 24, 24));
        }
        cv::Mat canvas(state.height, state.width, CV_8UC3, cv::Scalar(24, 24, 24));
        cv::Mat changed(state.height, state.width, CV_8UC1, cv::Scalar(0));
        const bool show_changed = changed_frame != 0U;
        for (int y = 0; y < state.height; ++y) {
            for (int x = 0; x < state.width; ++x) {
                const std::size_t index = static_cast<std::size_t>(y) * static_cast<std::size_t>(state.width)
                    + static_cast<std::size_t>(x);
                if (state.valid[index] == 0U) {
                    continue;
                }
                const auto color = unpack_rgba(state.rgba[index]);
                canvas.at<cv::Vec3b>(y, x) = cv::Vec3b(color[2], color[1], color[0]);
                if (show_changed && state.last_update_frame[index] == static_cast<std::uint32_t>(changed_frame)) {
                    changed.at<std::uint8_t>(y, x) = 255U;
                }
            }
        }

        cv::Mat panel;
        cv::resize(canvas, panel, cv::Size(kPanelWidth, kPanelHeight), 0.0, 0.0, cv::INTER_NEAREST);
        if (cv::countNonZero(changed) > 0) {
            cv::Mat eroded;
            cv::erode(changed, eroded, cv::Mat::ones(5, 5, CV_8U));
            cv::Mat inverse_eroded;
            cv::bitwise_not(eroded, inverse_eroded);
            cv::Mat edge;
            cv::bitwise_and(changed, inverse_eroded, edge);
            cv::Mat edge_panel;
            cv::resize(edge, edge_panel, cv::Size(kPanelWidth, kPanelHeight), 0.0, 0.0, cv::INTER_NEAREST);
            panel.setTo(cv::Scalar(0, 0, 255), edge_panel);
        }
        cv::putText(panel, "aligned canvas", {14, 29}, cv::FONT_HERSHEY_SIMPLEX, 0.72,
            cv::Scalar(245, 245, 245), 2, cv::LINE_AA);
        cv::putText(panel, "red outline: committed delta", {14, 56}, cv::FONT_HERSHEY_SIMPLEX, 0.48,
            cv::Scalar(220, 230, 240), 1, cv::LINE_AA);
        return panel;
    }

    void rebuild_cache_if_needed(const LocalState& active) const {
        const CanvasState& state = active.state();
        const FrameSeq changed_frame = active.changed_frame();
        if (cache_frame_ == active.frame()
            && cache_version_ == state.version
            && cache_changed_frame_ == changed_frame
            && cache_follow_live_ == follow_live_
            && cached_display_max_points_ == display_max_points_) {
            return;
        }

        const std::vector<ExportPoint> cleaned = export_clean_canvas_points(state, changed_frame);
        std::vector<CloudPoint> raw;
        raw.reserve(cleaned.size());
        for (const ExportPoint& point : cleaned) {
            raw.push_back(CloudPoint{
                point.x,
                point.y,
                point.z,
                cv::Vec3b(point.b, point.g, point.r),
                point.changed});
        }

#if 0
        if (raw.size() >= 6U) {
            std::vector<std::uint8_t> keep(raw.size(), 1U);
            std::array<double, 6> coefficients{};
            for (int iteration = 0; iteration < 3; ++iteration) {
                coefficients = fit_surface(raw, keep);
                std::vector<double> residuals;
                residuals.reserve(raw.size());
                for (std::size_t index = 0; index < raw.size(); ++index) {
                    if (keep[index] != 0U) {
                        residuals.push_back(static_cast<double>(raw[index].z) - surface_value(coefficients, raw[index]));
                    }
                }
                const double center = median_value(residuals);
                std::vector<double> deviations;
                deviations.reserve(residuals.size());
                for (const double value : residuals) {
                    deviations.push_back(std::abs(value - center));
                }
                const double mad = std::max(median_value(deviations), 1e-6);
                const double threshold = 4.0 * 1.4826 * mad;
                std::size_t kept = 0;
                for (std::size_t index = 0; index < raw.size(); ++index) {
                    const double residual = static_cast<double>(raw[index].z) - surface_value(coefficients, raw[index]);
                    keep[index] = std::abs(residual - center) <= threshold ? 1U : 0U;
                    kept += keep[index] != 0U ? 1U : 0U;
                }
                if (kept < 128U) {
                    std::fill(keep.begin(), keep.end(), static_cast<std::uint8_t>(1U));
                    break;
                }
            }

            coefficients = fit_surface(raw, keep);
            std::vector<double> residuals;
            residuals.reserve(raw.size());
            std::vector<double> depth_values;
            depth_values.reserve(raw.size());
            for (const CloudPoint& point : raw) {
                residuals.push_back(static_cast<double>(point.z) - surface_value(coefficients, point));
                depth_values.push_back(point.z);
            }
            const double center = median_value(residuals);
            std::vector<double> deviations;
            deviations.reserve(residuals.size());
            for (const double value : residuals) {
                deviations.push_back(std::abs(value - center));
            }
            const double mad = std::max(median_value(deviations), 1e-6);
            const double base_depth = std::max(std::abs(median_value(depth_values)), 1e-6);
            const double limit = std::min(3.0 * 1.4826 * mad, 0.03 * base_depth);
            std::vector<double> clipped;
            clipped.reserve(residuals.size());
            for (const double value : residuals) {
                clipped.push_back(std::clamp(value, center - limit, center + limit));
            }
            const double clipped_center = median_value(clipped);
            for (std::size_t index = 0; index < raw.size(); ++index) {
                raw[index].z = static_cast<float>((clipped[index] - clipped_center) / base_depth);
            }
        } else {
            for (CloudPoint& point : raw) {
                point.z = 0.0f;
            }
        }
#endif

        cached_points_.clear();
        if (display_max_points_ == 0U || raw.size() <= display_max_points_) {
            cached_points_ = std::move(raw);
        } else if (display_max_points_ == 1U) {
            cached_points_.push_back(raw.front());
        } else {
            cached_points_.reserve(display_max_points_);
            for (std::size_t index = 0; index < display_max_points_; ++index) {
                const std::size_t source = (index * (raw.size() - 1U)) / (display_max_points_ - 1U);
                cached_points_.push_back(raw[source]);
            }
        }
        cached_delta_count_ = std::count_if(cached_points_.begin(), cached_points_.end(),
            [](const CloudPoint& point) { return point.changed; });
        cached_left_ = render_canvas_panel(state, changed_frame);
        cache_frame_ = active.frame();
        cache_version_ = state.version;
        cache_changed_frame_ = changed_frame;
        cache_follow_live_ = follow_live_;
        cached_display_max_points_ = display_max_points_;
    }

    cv::Mat render_cloud_panel(const OrbitCamera& camera) const {
        cv::Mat image(kPanelHeight, kPanelWidth, CV_8UC3, cv::Scalar(24, 24, 24));
        cv::putText(image, "replayed point cloud", {14, 29}, cv::FONT_HERSHEY_SIMPLEX, 0.72,
            cv::Scalar(245, 245, 245), 2, cv::LINE_AA);
        if (cached_points_.empty()) {
            cv::putText(image, "waiting for valid fused points...", {30, kPanelHeight / 2},
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(190, 200, 210), 1, cv::LINE_AA);
            return image;
        }

        cv::Mat changed_pixels(kPanelHeight, kPanelWidth, CV_8UC1, cv::Scalar(0));
        std::size_t rendered = 0;
        const float cos_yaw = std::cos(camera.yaw);
        const float sin_yaw = std::sin(camera.yaw);
        const float cos_pitch = std::cos(camera.pitch);
        const float sin_pitch = std::sin(camera.pitch);
        for (const CloudPoint& point : cached_points_) {
            const float x1 = point.x * cos_yaw - point.z * sin_yaw;
            const float z1 = point.x * sin_yaw + point.z * cos_yaw;
            const float y2 = point.y * cos_pitch - z1 * sin_pitch;
            const int px = static_cast<int>(std::lround(x1 * 540.0f * camera.zoom_factor + 380.0f + camera.pan_x));
            const int py = static_cast<int>(std::lround(-y2 * 540.0f * camera.zoom_factor + 285.0f + camera.pan_y));
            if (px < 0 || px >= kPanelWidth || py < 40 || py >= kPanelHeight - 25) {
                continue;
            }
            ++rendered;
            // Match the Python viewer's 3x3 footprint.  It is a display-only
            // raster footprint and does not alter the stored point cloud.
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int xx = px + dx;
                    const int yy = py + dy;
                    if (xx >= 0 && xx < kPanelWidth && yy >= 40 && yy < kPanelHeight - 25) {
                        image.at<cv::Vec3b>(yy, xx) = point.bgr;
                    }
                }
            }
            if (point.changed) {
                changed_pixels.at<std::uint8_t>(py, px) = 255U;
            }
        }

        if (cv::countNonZero(changed_pixels) > 0) {
            cv::Mat dilated;
            cv::dilate(changed_pixels, dilated, cv::Mat::ones(3, 3, CV_8U));
            cv::Mat inverse_changed;
            cv::bitwise_not(changed_pixels, inverse_changed);
            cv::Mat ring;
            cv::bitwise_and(dilated, inverse_changed, ring);
            image.setTo(cv::Scalar(0, 0, 255), ring);
        }
        cv::line(image, {kPanelWidth / 2, kPanelHeight / 2}, {kPanelWidth / 2 + 65, kPanelHeight / 2},
            cv::Scalar(80, 90, 100), 1, cv::LINE_AA);
        cv::line(image, {kPanelWidth / 2, kPanelHeight / 2}, {kPanelWidth / 2, kPanelHeight / 2 - 65},
            cv::Scalar(80, 90, 100), 1, cv::LINE_AA);
        std::ostringstream point_count;
        point_count << rendered << " rendered / " << cached_points_.size() << " valid";
        cv::putText(image, point_count.str(), {14, kPanelHeight - 14}, cv::FONT_HERSHEY_SIMPLEX, 0.48,
            cv::Scalar(180, 205, 215), 1, cv::LINE_AA);
        return image;
    }

    void invalidate_cache() const {
        cache_frame_ = std::numeric_limits<FrameSeq>::max();
        cache_version_ = std::numeric_limits<CommitVersion>::max();
        cache_changed_frame_ = std::numeric_limits<FrameSeq>::max();
        cached_left_.release();
        cached_points_.clear();
    }

    mutable std::mutex mutex_;
    int width_ = 0;
    int height_ = 0;
    std::string run_name_ = "-";
    std::string last_status_;
    LocalState live_;
    LocalState playback_;
    FrameSeq live_head_frame_ = 0;
    FrameSeq received_head_frame_ = 0;
    FrameSeq pending_frame_ = 0;
    FrameSeq view_cursor_frame_ = 0;
    CommitVersion live_version_ = 0;
    CommitVersion view_version_ = 0;
    FrameSeq playback_target_frame_ = 0;
    CommitVersion playback_target_version_ = 0;
    std::uint64_t seek_generation_ = 0;
    bool follow_live_ = true;
    bool inference_pending_ = false;
    bool playback_running_ = false;
    bool playback_loading_ = false;
    float playback_fps_ = 10.0f;
    std::size_t display_max_points_ = 0U;
    mutable std::size_t cached_display_max_points_ = std::numeric_limits<std::size_t>::max();
    mutable FrameSeq cache_frame_ = std::numeric_limits<FrameSeq>::max();
    mutable CommitVersion cache_version_ = std::numeric_limits<CommitVersion>::max();
    mutable FrameSeq cache_changed_frame_ = std::numeric_limits<FrameSeq>::max();
    mutable bool cache_follow_live_ = false;
    mutable cv::Mat cached_left_;
    mutable std::vector<CloudPoint> cached_points_;
    mutable std::size_t cached_delta_count_ = 0U;
};

class ViewerClient {
public:
    ViewerClient(TcpSocket socket, std::function<void(const Packet&)> on_packet)
        : socket_(std::move(socket)), on_packet_(std::move(on_packet)) {}

    ~ViewerClient() {
        stop();
        if (reader_thread_.joinable()) {
            reader_thread_.join();
        }
    }

    void start() {
        send(make_subscribe());
        reader_thread_ = std::thread([this] {
            try {
                Packet packet;
                while (!stopping_.load() && socket_.receive_packet(packet)) {
                    on_packet_(packet);
                }
            } catch (const std::exception& error) {
                std::cerr << "viewer connection: " << error.what() << "\n";
            }
            stopping_.store(true);
        });
    }

    void send(const Packet& packet) {
        std::lock_guard<std::mutex> lock(send_mutex_);
        if (!stopping_.load()) {
            socket_.send_packet(packet);
        }
    }

    void stop() {
        if (!stopping_.exchange(true)) {
            socket_.shutdown_both();
        }
    }

    bool stopped() const noexcept { return stopping_.load(); }

private:
    TcpSocket socket_;
    std::function<void(const Packet&)> on_packet_;
    std::mutex send_mutex_;
    std::atomic<bool> stopping_{false};
    std::thread reader_thread_;
};

struct ViewerApp {
    ViewerModel model;
    std::unique_ptr<ViewerClient> client;
    OrbitCamera camera;
    bool dragging = false;
    bool panning = false;
    int last_x = 0;
    int last_y = 0;
    bool updating_trackbar = false;
    int trackbar_max = 1;
    int trackbar_position = 0;
    bool quit = false;

    void send_seek(const std::optional<ViewerModel::SeekRequest>& request) {
        if (request.has_value() && client) {
            client->send(make_replay_request(ReplayRequestMessage{request->generation, request->target_frame}));
        }
    }

    void on_mouse(const int event, const int x, const int y, const int flags) {
        const bool in_cloud = x >= kPanelWidth && x < kWindowWidth && y >= 0 && y < kPanelHeight;
        if (event == cv::EVENT_MOUSEWHEEL && in_cloud) {
            camera.zoom(flags > 0 ? 1.15f : (1.0f / 1.15f));
            return;
        }
        if (event == cv::EVENT_LBUTTONDOWN && in_cloud) {
            dragging = true;
            panning = false;
            last_x = x;
            last_y = y;
            return;
        }
        if (event == cv::EVENT_RBUTTONDOWN && in_cloud) {
            dragging = true;
            panning = true;
            last_x = x;
            last_y = y;
            return;
        }
        if (event == cv::EVENT_MOUSEMOVE && dragging) {
            const float dx = static_cast<float>(x - last_x);
            const float dy = static_cast<float>(y - last_y);
            if (panning) {
                model.handle_pan(dx, dy, camera);
            } else {
                camera.rotate(dx * 0.012f, dy * 0.012f);
            }
            last_x = x;
            last_y = y;
            return;
        }
        if (event == cv::EVENT_LBUTTONUP || event == cv::EVENT_RBUTTONUP || event == cv::EVENT_MBUTTONUP) {
            dragging = false;
            panning = false;
        }
    }

    void on_trackbar(const int position) {
        if (updating_trackbar) {
            return;
        }
        trackbar_position = std::clamp(position, 0, trackbar_max);
        const FrameSeq head = model.live_head();
        send_seek(model.seek(std::min<FrameSeq>(static_cast<FrameSeq>(trackbar_position), head)));
    }

    void sync_trackbar(const std::string& window_name) {
        const int maximum = static_cast<int>(std::min<FrameSeq>(model.live_head(),
            static_cast<FrameSeq>(std::numeric_limits<int>::max())));
        const int bounded_maximum = std::max(maximum, 1);
        const int position = static_cast<int>(std::min<FrameSeq>(model.view_cursor(),
            static_cast<FrameSeq>(bounded_maximum)));
        updating_trackbar = true;
        if (bounded_maximum != trackbar_max) {
            cv::setTrackbarMax("frame", window_name, bounded_maximum);
            trackbar_max = bounded_maximum;
        }
        if (position != trackbar_position) {
            cv::setTrackbarPos("frame", window_name, position);
            trackbar_position = position;
        }
        updating_trackbar = false;
    }
};

void mouse_callback(int event, int x, int y, int flags, void* userdata) {
    if (userdata != nullptr) {
        static_cast<ViewerApp*>(userdata)->on_mouse(event, x, y, flags);
    }
}

struct ViewerArgs {
    std::string host = "127.0.0.1";
    std::uint16_t port = 37651;
    std::size_t display_max_points = 0U;
};

ViewerArgs parse_args(const int argc, char** argv) {
    ViewerArgs args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--host") {
            if (++i >= argc) {
                throw std::runtime_error("missing value for --host");
            }
            args.host = argv[i];
        } else if (key == "--port") {
            if (++i >= argc) {
                throw std::runtime_error("missing value for --port");
            }
            args.port = static_cast<std::uint16_t>(std::stoul(argv[i]));
        } else if (key == "--display-max-points" || key == "--display_max_points") {
            if (++i >= argc) {
                throw std::runtime_error("missing value for " + key);
            }
            args.display_max_points = static_cast<std::size_t>(std::stoull(argv[i]));
        } else if (key == "--help" || key == "-h") {
            std::cout << "Usage: omnivggt_live_viewer [--host 127.0.0.1] [--port 37651]"
                         " [--display-max-points 0]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const ViewerArgs args = parse_args(argc, argv);
        SocketRuntime socket_runtime;
        TcpSocket socket = TcpSocket::connect_to(args.host, args.port);
        ViewerApp app;
        app.model.set_display_max_points(args.display_max_points);
        app.client = std::make_unique<ViewerClient>(
            std::move(socket), [&app](const Packet& packet) { app.model.on_packet(packet); });
        app.client->start();

        const std::string window_name = "OmniVGGT C++ Live Replay";
        cv::namedWindow(window_name, cv::WINDOW_NORMAL);
        cv::resizeWindow(window_name, kWindowWidth, kWindowHeight);
        cv::setMouseCallback(window_name, mouse_callback, &app);
        cv::createTrackbar("frame", window_name, 0, 1,
            [](int position, void* userdata) {
                if (userdata != nullptr) {
                    static_cast<ViewerApp*>(userdata)->on_trackbar(position);
                }
            }, &app);
        auto next_playback = std::chrono::steady_clock::now();
        while (!app.quit && !app.client->stopped()) {
            const auto now = std::chrono::steady_clock::now();
            if (now >= next_playback) {
                app.send_seek(app.model.tick_playback());
                const int interval_ms = static_cast<int>(1000.0f / std::max(app.model.playback_fps(), 1.0f));
                next_playback = now + std::chrono::milliseconds(std::max(interval_ms, 1));
            }
            cv::Mat frame;
            app.model.render(frame, app.camera);
            cv::imshow(window_name, frame);
            app.sync_trackbar(window_name);
            const int key = cv::waitKey(16);
            if (key < 0) {
                continue;
            }
            switch (key & 0xff) {
            case 27:
            case 'q':
                app.quit = true;
                break;
            case 'r':
                app.camera.reset();
                break;
            case 'l':
                app.model.follow_live();
                break;
            case ' ':
                app.send_seek(app.model.toggle_playback());
                break;
            case '+':
            case '=':
                app.model.adjust_fps(1.0f);
                break;
            case '-':
            case '_':
                app.model.adjust_fps(-1.0f);
                break;
            case 81:  // Left on common OpenCV/Windows builds.
            case 'a': {
                const FrameSeq cursor = app.model.view_cursor();
                app.send_seek(app.model.seek(cursor == 0 ? 0 : cursor - 1));
                break;
            }
            case 83:  // Right on common OpenCV/Windows builds.
            case 'd': {
                const FrameSeq cursor = app.model.view_cursor();
                const FrameSeq head = app.model.live_head();
                app.send_seek(app.model.seek(std::min(cursor + 1U, head)));
                break;
            }
            default:
                break;
            }
        }
        app.client->stop();
        cv::destroyAllWindows();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << "\n";
        return 1;
    }
}
