#include "frame_source.hpp"

#include <algorithm>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace omnivggt::observer {

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

bool LiveFrameSource::submit_frame(const std::uint64_t source_seq, const cv::Mat& image) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) {
        return false;
    }
    auto owned_image = std::make_shared<cv::Mat>(image.clone());
    frames_.emplace_back(source_seq, std::move(owned_image));
    if (frames_.size() < group_size_) {
        return true;
    }

    RawFrame group;
    group.frame_seq = next_group_seq_++;
    group.group_anchor_index = static_cast<int>(anchor_index_);
    group.group_images.reserve(group_size_);
    group.group_source_seqs.reserve(group_size_);
    std::ostringstream key;
    for (std::size_t index = 0; index < group_size_; ++index) {
        if (index != 0U) {
            key << '|';
        }
        group.group_source_seqs.push_back(frames_[index].first);
        group.group_images.push_back(frames_[index].second);
        key << frames_[index].first;
    }
    group.group_key = key.str();
    group.path = std::filesystem::path("camera_" + std::to_string(group.group_source_seqs[anchor_index_]));
    group.group_paths.reserve(group_size_);
    for (const std::uint64_t sequence : group.group_source_seqs) {
        group.group_paths.emplace_back("camera_" + std::to_string(sequence));
    }
    frames_.erase(
        frames_.begin(),
        frames_.begin() + static_cast<std::ptrdiff_t>(std::min(group_stride_, group_size_)));
    return on_group_(std::move(group));
}

void LiveFrameSource::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    frames_.clear();
}

}  // namespace omnivggt::observer
