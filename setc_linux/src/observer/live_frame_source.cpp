#include "frame_source.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace omnivggt::observer {

namespace {

// HikvisionCameraClient flushes its three cached frames from a QMap every
// 33 ms tick. The SubmitFrame packets in one flush are adjacent on the TCP
// sender, but the receiver timestamps a packet only after its full image has
// arrived.  On the production camera stream the observed intra-flush gaps are
// below 20 ms while the next GUI flush remains above it; keep this fixed
// threshold between those two measured regimes.
constexpr std::chrono::milliseconds kBurstBoundaryGap{20};
constexpr std::array<std::size_t, 3> kModelSlotForArrival{1U, 0U, 2U};

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
    if (group_size_ != 3U || group_stride_ != 3U || anchor_index_ != 1U) {
        throw std::invalid_argument("live input requires group_size=3, stride=3, anchor=1");
    }
}

bool LiveFrameSource::submit_frame(const std::uint64_t source_seq, const cv::Mat& image) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto arrival = std::chrono::steady_clock::now();
    if (closed_ || image.empty()) {
        return false;
    }

    if (!have_previous_) {
        have_previous_ = true;
        previous_seq_ = source_seq;
        previous_arrival_ = arrival;
        return true;
    }

    const bool sequence_contiguous = previous_seq_ != std::numeric_limits<std::uint64_t>::max()
        && source_seq == previous_seq_ + 1U;
    const bool boundary = arrival - previous_arrival_ > kBurstBoundaryGap;
    if (!sequence_contiguous) {
        // A rollback/jump may be a reconnect or a sender drop. The current
        // frame is not allowed to bridge it; wait for a later explicit cycle
        // boundary before starting a new complete burst.
        synced_ = false;
        burst_position_ = 0;
        burst_frames_ = {};
        previous_seq_ = source_seq;
        previous_arrival_ = arrival;
        return true;
    }

    if (!synced_) {
        if (!boundary) {
            // Connection may have started at position 1 or 2 of a cycle. Do
            // not guess those lanes and do not emit a partial group.
            previous_seq_ = source_seq;
            previous_arrival_ = arrival;
            return true;
        }
        synced_ = true;
        burst_position_ = 0;
        burst_frames_ = {};
    } else if (burst_position_ != 0 && boundary) {
        // A disconnect or a delayed packet can leave a synced partial burst.
        // A later inter-cycle gap is sufficient to discard that partial burst
        // without re-detecting the boundary after every complete group.
        burst_position_ = 0;
        burst_frames_ = {};
    }

    burst_frames_[static_cast<std::size_t>(burst_position_)] = BurstFrame{
        source_seq,
        std::make_shared<cv::Mat>(image.clone())};
    ++burst_position_;
    previous_seq_ = source_seq;
    previous_arrival_ = arrival;
    if (burst_position_ < 3) {
        return true;
    }

    RawFrame group;
    group.frame_seq = next_group_seq_++;
    group.group_anchor_index = static_cast<int>(anchor_index_);
    group.group_images.reserve(3U);
    group.group_source_seqs.reserve(3U);
    group.group_paths.reserve(3U);
    group.group_model_slot_for_arrival = kModelSlotForArrival;
    std::ostringstream key;
    for (std::size_t position = 0; position < 3U; ++position) {
        if (position != 0U) {
            key << '|';
        }
        const BurstFrame& frame = burst_frames_[position];
        group.group_source_seqs.push_back(frame.source_seq);
        group.group_images.push_back(frame.image);
        group.group_paths.emplace_back(
            "camera_arrival" + std::to_string(position)
            + "_" + std::to_string(frame.source_seq));
        key << frame.source_seq;
    }
    group.group_key = key.str();
    std::size_t anchor_arrival = 0U;
    for (std::size_t position = 0; position < 3U; ++position) {
        if (group.group_model_slot_for_arrival[position] == anchor_index_) {
            anchor_arrival = position;
            break;
        }
    }
    group.path = group.group_paths[anchor_arrival];
    // The next source frame is the next burst position after this complete
    // group. No modulo of source_seq is used; only arrival position defines
    // the verified {1,0,2} model permutation.
    burst_position_ = 0;
    burst_frames_ = {};
    return on_group_(std::move(group));
}

void LiveFrameSource::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    synced_ = false;
    have_previous_ = false;
    burst_position_ = 0;
    burst_frames_ = {};
}

}  // namespace omnivggt::observer
