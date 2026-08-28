#pragma once

#include "stream_types.hpp"

#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace omnivggt::observer {

class VersionStore {
public:
    static VersionStore create_new(
        const std::filesystem::path& run_dir,
        const CanvasState& initial_state,
        std::uint32_t snapshot_interval = 60U);

    static VersionStore open_existing(const std::filesystem::path& run_dir);

    VersionStore() = default;

    const std::filesystem::path& run_dir() const noexcept { return run_dir_; }
    int width() const noexcept { return width_; }
    int height() const noexcept { return height_; }
    std::uint32_t snapshot_interval() const noexcept { return snapshot_interval_; }
    CommitVersion latest_version() const noexcept { return latest_version_; }
    CommitVersion latest_snapshot_version() const noexcept {
        return snapshots_.empty() ? 0U : snapshots_.back().first;
    }
    FrameSeq latest_frame() const noexcept { return latest_frame_; }

    std::uint64_t append_delta(const PointCloudDelta& delta);
    void append_frame(const FrameRecord& record, std::uint64_t delta_offset = std::numeric_limits<std::uint64_t>::max());
    void append_metrics(const std::string& csv_line);

    bool should_snapshot(const PointCloudDelta& delta) const noexcept;
    void write_snapshot(const CanvasState& state);

    PointCloudDelta read_delta(CommitVersion version) const;
    Snapshot load_snapshot_for_version(CommitVersion version) const;
    CanvasState recover_state() const;
    ReplayBundle build_replay(FrameSeq target_frame) const;

    std::vector<FrameRecord> frame_records() const { return frames_; }
    std::optional<FrameRecord> find_frame(FrameSeq frame_seq) const;
    FrameRecord frame_at_or_before(FrameSeq frame_seq) const;
    std::vector<DeltaIndexEntry> delta_index() const { return delta_index_; }
    void compact(std::size_t keep_groups);

private:
    std::filesystem::path run_dir_;
    std::filesystem::path deltas_path_;
    std::filesystem::path delta_index_path_;
    std::filesystem::path frame_index_path_;
    std::filesystem::path metrics_path_;
    int width_ = 0;
    int height_ = 0;
    std::uint32_t snapshot_interval_ = 60U;
    CommitVersion latest_version_ = 0;
    FrameSeq latest_frame_ = 0;
    std::vector<FrameRecord> frames_;
    std::vector<DeltaIndexEntry> delta_index_;
    std::vector<std::pair<CommitVersion, std::filesystem::path>> snapshots_;

    static void write_meta(const std::filesystem::path& run_dir, const CanvasState& state, std::uint32_t interval);
    static VersionStore load_files(const std::filesystem::path& run_dir);
    void load_meta();
    void load_indices();
    void load_snapshots();
    std::filesystem::path snapshot_path(CommitVersion version) const;
    std::optional<DeltaIndexEntry> find_delta_index(CommitVersion version) const;
    std::optional<std::pair<CommitVersion, std::filesystem::path>> find_snapshot(CommitVersion version) const;
    static std::vector<std::uint8_t> read_file(const std::filesystem::path& path);
    static void write_file(const std::filesystem::path& path, const std::vector<std::uint8_t>& data);
};

}  // namespace omnivggt::observer
