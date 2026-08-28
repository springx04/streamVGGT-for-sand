#include "stream_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace omnivggt::observer {

namespace {

constexpr std::uint32_t kTileSize = 32U;

std::uint32_t float_bits(const float value) noexcept {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void hash_bytes(std::uint64_t& hash, const void* data, const std::size_t size) noexcept {
    constexpr std::uint64_t kPrime = 1099511628211ULL;
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= kPrime;
    }
}

void ensure_slot(const CanvasState& state, const std::uint32_t slot_id) {
    if (!state.shape_valid() || slot_id >= state.slot_count()) {
        throw std::out_of_range("point slot id is outside CanvasState");
    }
}

void write_u32_vector(BinaryWriter& writer, const std::vector<std::uint32_t>& values) {
    if (values.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::length_error("uint32 vector is too large to serialize");
    }
    writer.u32(static_cast<std::uint32_t>(values.size()));
    for (const std::uint32_t value : values) {
        writer.u32(value);
    }
}

std::vector<std::uint32_t> read_u32_vector(BinaryReader& reader) {
    const std::uint32_t count = reader.u32();
    std::vector<std::uint32_t> values;
    values.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        values.push_back(reader.u32());
    }
    return values;
}

void write_float_vector(BinaryWriter& writer, const std::vector<float>& values) {
    if (values.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::length_error("float vector is too large to serialize");
    }
    writer.u32(static_cast<std::uint32_t>(values.size()));
    for (const float value : values) {
        writer.f32(value);
    }
}

std::vector<float> read_float_vector(BinaryReader& reader) {
    const std::uint32_t count = reader.u32();
    std::vector<float> values;
    values.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        values.push_back(reader.f32());
    }
    return values;
}

void write_byte_vector(BinaryWriter& writer, const std::vector<std::uint8_t>& values) {
    if (values.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::length_error("byte vector is too large to serialize");
    }
    writer.u32(static_cast<std::uint32_t>(values.size()));
    writer.bytes(values);
}

std::vector<std::uint8_t> read_byte_vector(BinaryReader& reader) {
    const std::uint32_t count = reader.u32();
    return reader.bytes(count);
}

}  // namespace

const char* frame_status_name(const FrameStatus status) noexcept {
    switch (status) {
    case FrameStatus::Received:
        return "Received";
    case FrameStatus::Aligning:
        return "Aligning";
    case FrameStatus::DiffReady:
        return "DiffReady";
    case FrameStatus::Inferencing:
        return "Inferencing";
    case FrameStatus::Committed:
        return "Committed";
    case FrameStatus::NoChange:
        return "NoChange";
    case FrameStatus::Coalesced:
        return "Coalesced";
    case FrameStatus::Failed:
        return "Failed";
    }
    return "Unknown";
}

std::size_t CanvasState::slot_count() const noexcept {
    if (width <= 0 || height <= 0) {
        return 0;
    }
    return static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
}

bool CanvasState::shape_valid() const noexcept {
    const std::size_t n = slot_count();
    return n > 0U
        && depth.size() == n
        && confidence.size() == n
        && rgba.size() == n
        && last_update_frame.size() == n
        && valid.size() == n
        && support.size() == n
        && anchor_rgba.size() == n;
}

void CanvasState::reset(const int new_width, const int new_height) {
    if (new_width <= 0 || new_height <= 0) {
        throw std::invalid_argument("CanvasState dimensions must be positive");
    }
    width = new_width;
    height = new_height;
    const std::size_t n = slot_count();
    depth.assign(n, 0.0f);
    confidence.assign(n, 0.0f);
    rgba.assign(n, 0U);
    last_update_frame.assign(n, 0U);
    valid.assign(n, 0U);
    support.assign(n, 0U);
    anchor_rgba.assign(n, 0U);
    anchor_camera = AnchorCamera{};
    version = 0;
    last_frame = 0;
    initialized = false;
}

std::uint32_t pack_rgba(
    const std::uint8_t r,
    const std::uint8_t g,
    const std::uint8_t b,
    const std::uint8_t a) noexcept {
    return static_cast<std::uint32_t>(r)
        | (static_cast<std::uint32_t>(g) << 8U)
        | (static_cast<std::uint32_t>(b) << 16U)
        | (static_cast<std::uint32_t>(a) << 24U);
}

std::array<std::uint8_t, 4> unpack_rgba(const std::uint32_t rgba) noexcept {
    return {
        static_cast<std::uint8_t>(rgba & 0xffU),
        static_cast<std::uint8_t>((rgba >> 8U) & 0xffU),
        static_cast<std::uint8_t>((rgba >> 16U) & 0xffU),
        static_cast<std::uint8_t>((rgba >> 24U) & 0xffU),
    };
}

SlotValue slot_value_at(const CanvasState& state, const std::uint32_t slot_id) {
    ensure_slot(state, slot_id);
    const std::size_t i = slot_id;
    return SlotValue{state.depth[i], state.confidence[i], state.rgba[i], state.last_update_frame[i], state.valid[i]};
}

void set_slot_value(CanvasState& state, const std::uint32_t slot_id, const SlotValue& value) {
    ensure_slot(state, slot_id);
    const std::size_t i = slot_id;
    state.depth[i] = value.depth;
    state.confidence[i] = value.confidence;
    state.rgba[i] = value.rgba;
    state.last_update_frame[i] = value.last_update_frame;
    state.valid[i] = value.valid;
}

bool slot_value_equal(const SlotValue& lhs, const SlotValue& rhs) noexcept {
    return float_bits(lhs.depth) == float_bits(rhs.depth)
        && float_bits(lhs.confidence) == float_bits(rhs.confidence)
        && lhs.rgba == rhs.rgba
        && lhs.last_update_frame == rhs.last_update_frame
        && lhs.valid == rhs.valid
        && float_bits(lhs.x) == float_bits(rhs.x)
        && float_bits(lhs.y) == float_bits(rhs.y);
}

PointCloudDelta commit_patch(CanvasState& state, const CandidatePatch& patch) {
    if (!state.shape_valid()) {
        if (!patch.initialize_canvas || patch.anchor_rgba.empty()) {
            throw std::runtime_error("first patch must initialize CanvasState");
        }
        if (patch.width <= 0 || patch.height <= 0) {
            throw std::runtime_error("first patch must specify CanvasState dimensions");
        }
        state.reset(patch.width, patch.height);
    }
    if (state.version != patch.base_version) {
        throw std::runtime_error("candidate patch base_version does not match authoritative state");
    }

    PointCloudDelta delta;
    delta.frame_seq = patch.frame_seq;
    delta.previous_frame = state.last_frame;
    delta.from_version = state.version;
    delta.changed_ratio = patch.changed_ratio;
    delta.scene_jump = patch.scene_jump;
    delta.initializes_anchor = patch.initialize_canvas;
    delta.anchor_camera = patch.anchor_camera;
    if (patch.initialize_canvas) {
        if (patch.anchor_rgba.size() != state.slot_count()) {
            throw std::runtime_error("anchor image size does not match CanvasState");
        }
        state.anchor_rgba = patch.anchor_rgba;
        state.anchor_camera = patch.anchor_camera;
        state.initialized = true;
        delta.anchor_rgba = patch.anchor_rgba;
    }

    for (const SlotUpdate& update : patch.updates) {
        const SlotValue before = slot_value_at(state, update.slot_id);
        const SlotValue after = update.after;
        if (slot_value_equal(before, after)) {
            continue;
        }
        delta.changes.push_back(SlotDelta{update.slot_id, before, after});
        set_slot_value(state, update.slot_id, after);
    }

    std::vector<std::uint8_t> observed(state.slot_count(), 0U);
    for (const std::uint32_t slot_id : patch.observed_slots) {
        ensure_slot(state, slot_id);
        observed[slot_id] = 1U;
    }
    for (const std::uint32_t slot_id : patch.observed_slots) {
        const std::uint8_t before = state.support[slot_id];
        const std::uint8_t after = 1U;
        if (before != after) {
            delta.support_changes.push_back(SupportDelta{slot_id, before, after});
            state.support[slot_id] = after;
        }
    }
    for (const std::uint32_t slot_id : patch.cleared_support_slots) {
        ensure_slot(state, slot_id);
        const std::uint8_t before = state.support[slot_id];
        const std::uint8_t after = 0U;
        if (before != after) {
            delta.support_changes.push_back(SupportDelta{slot_id, before, after});
            state.support[slot_id] = after;
        }
    }
    (void)observed;

    if (delta.changes.empty() && delta.support_changes.empty() && !delta.initializes_anchor) {
        delta.to_version = state.version;
        delta.valid_point_count = static_cast<std::uint32_t>(
            std::count(state.valid.begin(), state.valid.end(), static_cast<std::uint8_t>(1U)));
        return delta;
    }

    state.version += 1U;
    state.last_frame = patch.frame_seq;
    delta.to_version = state.version;
    delta.valid_point_count = static_cast<std::uint32_t>(
        std::count(state.valid.begin(), state.valid.end(), static_cast<std::uint8_t>(1U)));

    const std::size_t tile_count_x = (static_cast<std::size_t>(state.width) + kTileSize - 1U) / kTileSize;
    std::vector<std::uint8_t> dirty(tile_count_x * ((static_cast<std::size_t>(state.height) + kTileSize - 1U) / kTileSize), 0U);
    const auto add_tile = [&](const std::uint32_t slot_id) {
        const std::size_t x = slot_id % static_cast<std::size_t>(state.width);
        const std::size_t y = slot_id / static_cast<std::size_t>(state.width);
        const std::size_t tile = (y / kTileSize) * tile_count_x + (x / kTileSize);
        if (tile < dirty.size()) {
            dirty[tile] = 1U;
        }
    };
    for (const SlotDelta& change : delta.changes) {
        add_tile(change.slot_id);
    }
    for (const SupportDelta& change : delta.support_changes) {
        add_tile(change.slot_id);
    }
    for (std::size_t i = 0; i < dirty.size(); ++i) {
        if (dirty[i] != 0U) {
            delta.dirty_tile_ids.push_back(static_cast<std::uint32_t>(i));
        }
    }
    return delta;
}

void apply_delta_forward(CanvasState& state, const PointCloudDelta& delta) {
    if (state.version != delta.from_version) {
        throw std::runtime_error("forward delta version mismatch");
    }
    if (delta.initializes_anchor) {
        if (delta.anchor_rgba.size() != state.slot_count()) {
            throw std::runtime_error("forward delta anchor size mismatch");
        }
        state.anchor_rgba = delta.anchor_rgba;
        state.anchor_camera = delta.anchor_camera;
        state.initialized = true;
    }
    for (const SlotDelta& change : delta.changes) {
        set_slot_value(state, change.slot_id, change.after);
    }
    for (const SupportDelta& change : delta.support_changes) {
        ensure_slot(state, change.slot_id);
        state.support[change.slot_id] = change.after;
    }
    state.version = delta.to_version;
    state.last_frame = delta.frame_seq;
}

void apply_delta_backward(CanvasState& state, const PointCloudDelta& delta) {
    if (state.version != delta.to_version) {
        throw std::runtime_error("backward delta version mismatch");
    }
    for (auto it = delta.changes.rbegin(); it != delta.changes.rend(); ++it) {
        set_slot_value(state, it->slot_id, it->before);
    }
    for (auto it = delta.support_changes.rbegin(); it != delta.support_changes.rend(); ++it) {
        ensure_slot(state, it->slot_id);
        state.support[it->slot_id] = it->before;
    }
    if (delta.initializes_anchor) {
        std::fill(state.anchor_rgba.begin(), state.anchor_rgba.end(), 0U);
        state.anchor_camera = AnchorCamera{};
        state.initialized = false;
    }
    state.version = delta.from_version;
    state.last_frame = delta.previous_frame;
}

std::uint64_t hash_state(const CanvasState& state) noexcept {
    std::uint64_t hash = 1469598103934665603ULL;
    hash_bytes(hash, &state.width, sizeof(state.width));
    hash_bytes(hash, &state.height, sizeof(state.height));
    hash_bytes(hash, state.depth.data(), state.depth.size() * sizeof(float));
    hash_bytes(hash, state.confidence.data(), state.confidence.size() * sizeof(float));
    hash_bytes(hash, state.rgba.data(), state.rgba.size() * sizeof(std::uint32_t));
    hash_bytes(hash, state.last_update_frame.data(), state.last_update_frame.size() * sizeof(std::uint32_t));
    hash_bytes(hash, state.valid.data(), state.valid.size() * sizeof(std::uint8_t));
    hash_bytes(hash, state.support.data(), state.support.size() * sizeof(std::uint8_t));
    hash_bytes(hash, state.anchor_rgba.data(), state.anchor_rgba.size() * sizeof(std::uint32_t));
    hash_bytes(hash, &state.anchor_camera, sizeof(state.anchor_camera));
    hash_bytes(hash, &state.version, sizeof(state.version));
    hash_bytes(hash, &state.last_frame, sizeof(state.last_frame));
    hash_bytes(hash, &state.initialized, sizeof(state.initialized));
    return hash;
}

void write_anchor_camera(BinaryWriter& writer, const AnchorCamera& camera) {
    writer.f32(camera.fx);
    writer.f32(camera.fy);
    writer.f32(camera.cx);
    writer.f32(camera.cy);
    writer.f32(camera.depth_scale);
    writer.f32(camera.depth_bias);
}

AnchorCamera read_anchor_camera(BinaryReader& reader) {
    AnchorCamera camera;
    camera.fx = reader.f32();
    camera.fy = reader.f32();
    camera.cx = reader.f32();
    camera.cy = reader.f32();
    camera.depth_scale = reader.f32();
    camera.depth_bias = reader.f32();
    return camera;
}

void write_slot_value(BinaryWriter& writer, const SlotValue& value) {
    writer.f32(value.depth);
    writer.f32(value.confidence);
    writer.u32(value.rgba);
    writer.u32(value.last_update_frame);
    writer.u8(value.valid);
    writer.f32(value.x);
    writer.f32(value.y);
}

SlotValue read_slot_value(BinaryReader& reader) {
    SlotValue value;
    value.depth = reader.f32();
    value.confidence = reader.f32();
    value.rgba = reader.u32();
    value.last_update_frame = reader.u32();
    value.valid = reader.u8();
    value.x = reader.f32();
    value.y = reader.f32();
    return value;
}

void write_frame_record(BinaryWriter& writer, const FrameRecord& record) {
    writer.u64(record.frame_seq);
    writer.u64(record.base_version);
    writer.u64(record.commit_version);
    writer.u8(static_cast<std::uint8_t>(record.status));
    writer.f32(record.changed_ratio);
    writer.f32(record.align_error_px);
    writer.u32(record.changed_point_count);
    writer.u32(record.valid_point_count);
    writer.string(record.image_name);
}

FrameRecord read_frame_record(BinaryReader& reader) {
    FrameRecord record;
    record.frame_seq = reader.u64();
    record.base_version = reader.u64();
    record.commit_version = reader.u64();
    record.status = static_cast<FrameStatus>(reader.u8());
    record.changed_ratio = reader.f32();
    record.align_error_px = reader.f32();
    record.changed_point_count = reader.u32();
    record.valid_point_count = reader.u32();
    record.image_name = reader.string();
    return record;
}

void write_delta(BinaryWriter& writer, const PointCloudDelta& delta) {
    writer.u64(delta.frame_seq);
    writer.u64(delta.previous_frame);
    writer.u64(delta.from_version);
    writer.u64(delta.to_version);
    writer.f32(delta.changed_ratio);
    writer.u32(delta.valid_point_count);
    writer.boolean(delta.scene_jump);
    writer.boolean(delta.initializes_anchor);
    write_anchor_camera(writer, delta.anchor_camera);
    write_u32_vector(writer, delta.anchor_rgba);

    if (delta.changes.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::length_error("delta changes are too large to serialize");
    }
    writer.u32(static_cast<std::uint32_t>(delta.changes.size()));
    for (const SlotDelta& change : delta.changes) {
        writer.u32(change.slot_id);
        write_slot_value(writer, change.before);
        write_slot_value(writer, change.after);
    }

    if (delta.support_changes.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::length_error("delta support changes are too large to serialize");
    }
    writer.u32(static_cast<std::uint32_t>(delta.support_changes.size()));
    for (const SupportDelta& change : delta.support_changes) {
        writer.u32(change.slot_id);
        writer.u8(change.before);
        writer.u8(change.after);
    }
    write_u32_vector(writer, delta.dirty_tile_ids);
}

PointCloudDelta read_delta(BinaryReader& reader) {
    PointCloudDelta delta;
    delta.frame_seq = reader.u64();
    delta.previous_frame = reader.u64();
    delta.from_version = reader.u64();
    delta.to_version = reader.u64();
    delta.changed_ratio = reader.f32();
    delta.valid_point_count = reader.u32();
    delta.scene_jump = reader.boolean();
    delta.initializes_anchor = reader.boolean();
    delta.anchor_camera = read_anchor_camera(reader);
    delta.anchor_rgba = read_u32_vector(reader);

    const std::uint32_t change_count = reader.u32();
    delta.changes.reserve(change_count);
    for (std::uint32_t i = 0; i < change_count; ++i) {
        SlotDelta change;
        change.slot_id = reader.u32();
        change.before = read_slot_value(reader);
        change.after = read_slot_value(reader);
        delta.changes.push_back(change);
    }

    const std::uint32_t support_count = reader.u32();
    delta.support_changes.reserve(support_count);
    for (std::uint32_t i = 0; i < support_count; ++i) {
        SupportDelta change;
        change.slot_id = reader.u32();
        change.before = reader.u8();
        change.after = reader.u8();
        delta.support_changes.push_back(change);
    }
    delta.dirty_tile_ids = read_u32_vector(reader);
    return delta;
}

void write_snapshot(BinaryWriter& writer, const Snapshot& snapshot) {
    const CanvasState& state = snapshot.state;
    writer.u32(0x4e53564fU);  // OVSN
    writer.u16(1U);
    writer.u16(0U);
    writer.u32(static_cast<std::uint32_t>(state.width));
    writer.u32(static_cast<std::uint32_t>(state.height));
    writer.u64(state.version);
    writer.u64(state.last_frame);
    writer.boolean(state.initialized);
    writer.u8(0U);
    writer.u16(0U);
    write_anchor_camera(writer, state.anchor_camera);
    write_float_vector(writer, state.depth);
    write_float_vector(writer, state.confidence);
    write_u32_vector(writer, state.rgba);
    write_u32_vector(writer, state.last_update_frame);
    write_byte_vector(writer, state.valid);
    write_byte_vector(writer, state.support);
    write_u32_vector(writer, state.anchor_rgba);
}

Snapshot read_snapshot(BinaryReader& reader) {
    const std::uint32_t magic = reader.u32();
    if (magic != 0x4e53564fU) {
        throw std::runtime_error("invalid snapshot magic");
    }
    const std::uint16_t schema = reader.u16();
    if (schema != 1U) {
        throw std::runtime_error("unsupported snapshot schema");
    }
    (void)reader.u16();
    CanvasState state;
    state.width = static_cast<int>(reader.u32());
    state.height = static_cast<int>(reader.u32());
    state.version = reader.u64();
    state.last_frame = reader.u64();
    state.initialized = reader.boolean();
    (void)reader.u8();
    (void)reader.u16();
    state.anchor_camera = read_anchor_camera(reader);
    state.depth = read_float_vector(reader);
    state.confidence = read_float_vector(reader);
    state.rgba = read_u32_vector(reader);
    state.last_update_frame = read_u32_vector(reader);
    state.valid = read_byte_vector(reader);
    state.support = read_byte_vector(reader);
    state.anchor_rgba = read_u32_vector(reader);
    if (!state.shape_valid()) {
        throw std::runtime_error("snapshot arrays do not match dimensions");
    }
    return Snapshot{std::move(state)};
}

}  // namespace omnivggt::observer
