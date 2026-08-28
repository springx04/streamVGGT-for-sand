#pragma once

#include "binary_io.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace omnivggt::observer {

using FrameSeq = std::uint64_t;
using CommitVersion = std::uint64_t;

enum class FrameStatus : std::uint8_t {
    Received = 0,
    Aligning = 1,
    DiffReady = 2,
    Inferencing = 3,
    Committed = 4,
    NoChange = 5,
    Coalesced = 6,
    Failed = 7,
};

const char* frame_status_name(FrameStatus status) noexcept;

struct FrameRecord {
    FrameSeq frame_seq = 0;
    CommitVersion base_version = 0;
    CommitVersion commit_version = 0;
    FrameStatus status = FrameStatus::Received;
    float changed_ratio = 0.0f;
    float align_error_px = -1.0f;
    std::uint32_t changed_point_count = 0;
    std::uint32_t valid_point_count = 0;
    std::string image_name;
};

struct AnchorCamera {
    float fx = 1.0f;
    float fy = 1.0f;
    float cx = 0.0f;
    float cy = 0.0f;
    float depth_scale = 1.0f;
    float depth_bias = 0.0f;
};

struct SlotValue {
    float depth = 0.0f;
    float confidence = 0.0f;
    std::uint32_t rgba = 0;
    std::uint32_t last_update_frame = 0;
    std::uint8_t valid = 0;
    // World-coordinate X/Y (joint B=1,S=3 mode).  Zero in the legacy
    // aligned-depth residual path; populated by process_world_group.
    float x = 0.0f;
    float y = 0.0f;
};

struct SlotDelta {
    std::uint32_t slot_id = 0;
    SlotValue before;
    SlotValue after;
};

struct SupportDelta {
    std::uint32_t slot_id = 0;
    std::uint8_t before = 0;
    std::uint8_t after = 0;
};

// A CandidatePatch is produced by the inference worker and is not allowed to
// mutate the authoritative state.  The commit worker turns it into a
// PointCloudDelta after checking base_version.
struct SlotUpdate {
    std::uint32_t slot_id = 0;
    SlotValue after;
};

struct CandidatePatch {
    FrameSeq frame_seq = 0;
    CommitVersion base_version = 0;
    int width = 0;
    int height = 0;
    float changed_ratio = 0.0f;
    bool scene_jump = false;
    bool initialize_canvas = false;
    AnchorCamera anchor_camera;
    std::vector<std::uint32_t> anchor_rgba;
    std::vector<SlotUpdate> updates;
    std::vector<std::uint32_t> observed_slots;
    std::vector<std::uint32_t> cleared_support_slots;
};

struct PointCloudDelta {
    FrameSeq frame_seq = 0;
    FrameSeq previous_frame = 0;
    CommitVersion from_version = 0;
    CommitVersion to_version = 0;
    float changed_ratio = 0.0f;
    std::uint32_t valid_point_count = 0;
    bool scene_jump = false;
    bool initializes_anchor = false;
    AnchorCamera anchor_camera;
    std::vector<std::uint32_t> anchor_rgba;
    std::vector<SlotDelta> changes;
    std::vector<SupportDelta> support_changes;
    std::vector<std::uint32_t> dirty_tile_ids;
};

struct CanvasState {
    int width = 0;
    int height = 0;
    std::vector<float> depth;
    std::vector<float> confidence;
    std::vector<std::uint32_t> rgba;
    std::vector<std::uint32_t> last_update_frame;
    std::vector<std::uint8_t> valid;
    std::vector<std::uint8_t> support;
    std::vector<std::uint32_t> anchor_rgba;
    AnchorCamera anchor_camera;
    CommitVersion version = 0;
    FrameSeq last_frame = 0;
    bool initialized = false;

    std::size_t slot_count() const noexcept;
    bool shape_valid() const noexcept;
    void reset(int new_width, int new_height);
};

struct Snapshot {
    CanvasState state;
};

struct DeltaIndexEntry {
    CommitVersion version = 0;
    std::uint64_t file_offset = 0;
    std::uint32_t record_size = 0;
};

struct ReplayBundle {
    FrameSeq target_frame = 0;
    CommitVersion target_version = 0;
    Snapshot snapshot;
    std::vector<PointCloudDelta> deltas;
};

std::uint32_t pack_rgba(std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a = 255) noexcept;
std::array<std::uint8_t, 4> unpack_rgba(std::uint32_t rgba) noexcept;

SlotValue slot_value_at(const CanvasState& state, std::uint32_t slot_id);
void set_slot_value(CanvasState& state, std::uint32_t slot_id, const SlotValue& value);
bool slot_value_equal(const SlotValue& lhs, const SlotValue& rhs) noexcept;

PointCloudDelta commit_patch(CanvasState& state, const CandidatePatch& patch);
void apply_delta_forward(CanvasState& state, const PointCloudDelta& delta);
void apply_delta_backward(CanvasState& state, const PointCloudDelta& delta);
std::uint64_t hash_state(const CanvasState& state) noexcept;

void write_anchor_camera(BinaryWriter& writer, const AnchorCamera& camera);
AnchorCamera read_anchor_camera(BinaryReader& reader);
void write_slot_value(BinaryWriter& writer, const SlotValue& value);
SlotValue read_slot_value(BinaryReader& reader);
void write_frame_record(BinaryWriter& writer, const FrameRecord& record);
FrameRecord read_frame_record(BinaryReader& reader);
void write_delta(BinaryWriter& writer, const PointCloudDelta& delta);
PointCloudDelta read_delta(BinaryReader& reader);
void write_snapshot(BinaryWriter& writer, const Snapshot& snapshot);
Snapshot read_snapshot(BinaryReader& reader);

}  // namespace omnivggt::observer
