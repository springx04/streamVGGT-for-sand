#include "frame_source.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <opencv2/imgproc.hpp>

namespace omnivggt::observer {

namespace {

// SubmitFrame carries a global sequence only; the GUI-side camera index is
// intentionally not part of that protocol. Identify the three fixed camera
// views from a small normalized thumbnail instead of assuming arrival order
// or sequence modulo three is a camera identity.
constexpr double kSameCameraDistance = 0.30;
constexpr double kNewCameraDistance = 0.50;
constexpr std::uint64_t kMaxLiveGroupSourceSpan = 15U;

}  // namespace

LiveFrameSource::LiveFrameSource(
    const std::size_t group_size,
    const std::size_t group_stride,
    const std::size_t anchor_index,
    GroupHandler on_group)
    : group_size_(group_size),
      group_stride_(group_stride),
      anchor_index_(anchor_index),
      on_group_(std::move(on_group)) {
    if (group_size_ != 3U || group_stride_ == 0U || group_stride_ > group_size_ || anchor_index_ != 1U) {
        throw std::invalid_argument("live input requires group_size=3, stride in [1,3], anchor=1");
    }
}

cv::Mat LiveFrameSource::make_camera_signature(const cv::Mat& image) {
    if (image.empty()) {
        return {};
    }
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_RGB2GRAY);
    } else if (image.channels() == 1) {
        gray = image;
    } else {
        throw std::invalid_argument("live camera frame must be one- or three-channel");
    }
    cv::Mat thumbnail;
    cv::resize(gray, thumbnail, cv::Size(32, 24), 0.0, 0.0, cv::INTER_AREA);
    thumbnail.convertTo(thumbnail, CV_32FC1, 1.0 / 255.0);
    cv::Scalar mean;
    cv::Scalar deviation;
    cv::meanStdDev(thumbnail, mean, deviation);
    const double scale = std::max(deviation[0], 0.03);
    thumbnail = (thumbnail - mean[0]) / scale;
    return thumbnail;
}

double LiveFrameSource::signature_distance(
    const cv::Mat& lhs,
    const cv::Mat& rhs) {
    if (lhs.empty() || rhs.empty() || lhs.size() != rhs.size() || lhs.type() != rhs.type()) {
        return std::numeric_limits<double>::infinity();
    }
    return cv::norm(lhs, rhs, cv::NORM_L1)
        / static_cast<double>(std::max<std::size_t>(1U, lhs.total()));
}

bool LiveFrameSource::submit_frame(
    const std::uint64_t source_seq,
    const cv::Mat& image) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_ || image.empty()) {
        return false;
    }

    const cv::Mat signature = make_camera_signature(image);
    std::size_t best_slot = 0U;
    double best_distance = std::numeric_limits<double>::infinity();
    int empty_slot = -1;
    for (std::size_t index = 0; index < camera_slots_.size(); ++index) {
        CameraSlot& slot = camera_slots_[index];
        if (!slot.initialized) {
            if (empty_slot < 0) {
                empty_slot = static_cast<int>(index);
            }
            continue;
        }
        const double distance = signature_distance(signature, slot.signature);
        if (distance < best_distance) {
            best_distance = distance;
            best_slot = index;
        }
    }

    // During startup, only create a new lane when the image is sufficiently
    // different from every known lane. Repeated frames from one camera then
    // update one slot until the other two actual cameras arrive, even if the
    // GUI thread scheduling delivers several frames from the same camera in a
    // row. Once all lanes exist, nearest-view assignment keeps identities
    // stable while allowing ordinary scene motion and exposure drift.
    std::size_t slot_index = best_slot;
    if (empty_slot >= 0
        && (best_distance == std::numeric_limits<double>::infinity()
            || best_distance >= kNewCameraDistance)) {
        slot_index = static_cast<std::size_t>(empty_slot);
    } else if (best_distance > kSameCameraDistance && empty_slot >= 0
        && best_distance >= kNewCameraDistance * 0.85) {
        slot_index = static_cast<std::size_t>(empty_slot);
    }

    CameraSlot& slot = camera_slots_[slot_index];
    slot.source_seq = source_seq;
    slot.image = std::make_shared<cv::Mat>(image.clone());
    if (!slot.initialized) {
        slot.signature = signature.clone();
        slot.initialized = true;
    } else {
        // Slowly follow illumination/object changes without allowing one
        // transient frame to redefine a camera identity.
        cv::addWeighted(slot.signature, 0.85, signature, 0.15, 0.0, slot.signature);
    }
    slot.dirty = true;

    for (const CameraSlot& current : camera_slots_) {
        if (!current.initialized || !current.dirty) {
            return true;
        }
    }

    std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t newest = 0U;
    for (const CameraSlot& current : camera_slots_) {
        oldest = std::min(oldest, current.source_seq);
        newest = std::max(newest, current.source_seq);
    }
    if (newest - oldest > kMaxLiveGroupSourceSpan) {
        // Keep the dirty flags. As the slowest camera sends a newer frame the
        // span will contract; emitting now would combine visibly different
        // moments and cause the reconstructed cloud to jump.
        return true;
    }

    RawFrame group;
    group.frame_seq = next_group_seq_++;
    group.group_anchor_index = static_cast<int>(anchor_index_);
    group.group_images.reserve(group_size_);
    group.group_source_seqs.reserve(group_size_);
    group.group_paths.reserve(group_size_);
    std::ostringstream key;
    for (std::size_t index = 0; index < group_size_; ++index) {
        const CameraSlot& current = camera_slots_[index];
        if (index != 0U) {
            key << '|';
        }
        group.group_source_seqs.push_back(current.source_seq);
        group.group_images.push_back(current.image);
        const std::string name = "camera_slot" + std::to_string(index)
            + "_" + std::to_string(current.source_seq);
        group.group_paths.emplace_back(name);
        key << current.source_seq;
    }
    group.group_key = key.str();
    group.path = group.group_paths[anchor_index_];
    for (CameraSlot& current : camera_slots_) {
        current.dirty = false;
    }
    return on_group_ ? on_group_(std::move(group)) : false;
}

void LiveFrameSource::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    for (CameraSlot& slot : camera_slots_) {
        slot = CameraSlot{};
    }
}

}  // namespace omnivggt::observer
