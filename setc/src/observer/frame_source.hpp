#pragma once

#include "bounded_queue.hpp"
#include "stream_types.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <functional>
#include <memory>
#include <opencv2/core.hpp>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace omnivggt::observer {

struct RawFrame {
    FrameSeq frame_seq = 0;
    std::filesystem::path path;
    // A logical input can contain a sliding window of source images.  The
    // legacy single-image path keeps this vector empty (or size one when it
    // is produced by the grouped source); consumers should use group_paths
    // when present and path as the canonical/anchor image.
    std::vector<std::filesystem::path> group_paths;
    std::vector<std::uint64_t> group_source_seqs;
    int group_anchor_index = 0;
    std::string group_key;
    std::vector<std::shared_ptr<const cv::Mat>> group_images;
};

class LiveFrameSource {
public:
    using GroupHandler = std::function<bool(RawFrame)>;

    LiveFrameSource(
        std::size_t group_size,
        std::size_t group_stride,
        std::size_t anchor_index,
        GroupHandler on_group);

    bool submit_frame(std::uint64_t source_seq, const cv::Mat& image);
    void close();

private:
    const std::size_t group_size_;
    const std::size_t group_stride_;
    const std::size_t anchor_index_;
    GroupHandler on_group_;
    std::mutex mutex_;
    bool closed_ = false;
    FrameSeq next_group_seq_ = 0;
    std::vector<std::pair<std::uint64_t, std::shared_ptr<const cv::Mat>>> frames_;
};

struct DirectorySourceOptions {
    std::filesystem::path directory;
    std::size_t queue_capacity = 3U;
    int poll_ms = 50;
    std::uint64_t max_frames = 0;
    FrameSeq start_frame_seq = 0;
    std::size_t group_size = 1U;
    std::size_t group_stride = 1U;
    std::size_t group_anchor_index = 0U;
    std::unordered_set<std::string> skip_image_names;
    std::unordered_set<std::string> skip_group_keys;
    bool once = false;
};

class DirectoryFrameSource {
public:
    explicit DirectoryFrameSource(DirectorySourceOptions options);

    void run(
        BoundedQueue<RawFrame>& output,
        std::atomic<bool>& stop_requested,
        const std::function<void(const RawFrame&)>& on_received,
        const std::function<void(const RawFrame&)>& on_coalesced);

private:
    struct Observation {
        std::uintmax_t size = 0;
        std::filesystem::file_time_type modified{};
        int stable_scans = 0;
    };

    DirectorySourceOptions options_;
    std::unordered_map<std::string, Observation> observations_;
    std::unordered_set<std::string> processed_;

    static bool is_image(const std::filesystem::path& path);
    std::vector<std::filesystem::path> scan() const;
};

}  // namespace omnivggt::observer
