#include "version_store.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace omnivggt::observer {

namespace {

constexpr std::uint32_t kRunMagic = 0x4d52564fU;    // OVRM
constexpr std::uint32_t kDeltaMagic = 0x4c44564fU;  // OVDL
constexpr std::uint16_t kSchema = 1U;
constexpr std::size_t kDeltaHeaderSize = 48U;

std::uint32_t crc32(const std::vector<std::uint8_t>& data) noexcept {
    std::uint32_t crc = 0xffffffffU;
    for (const std::uint8_t byte : data) {
        crc ^= byte;
        for (int bit = 0; bit < 8; ++bit) {
            const std::uint32_t mask = static_cast<std::uint32_t>(-(static_cast<int>(crc & 1U)));
            crc = (crc >> 1U) ^ (0xedb88320U & mask);
        }
    }
    return ~crc;
}

std::string version_file_name(const CommitVersion version) {
    std::ostringstream name;
    name << "snapshot_" << std::setw(20) << std::setfill('0') << version << ".bin";
    return name.str();
}

}  // namespace

VersionStore VersionStore::create_new(
    const std::filesystem::path& run_dir,
    const CanvasState& initial_state,
    const std::uint32_t snapshot_interval) {
    if (!initial_state.shape_valid()) {
        throw std::invalid_argument("initial VersionStore state has invalid shape");
    }
    std::filesystem::create_directories(run_dir / "snapshots");
    VersionStore store;
    store.run_dir_ = run_dir;
    store.deltas_path_ = run_dir / "deltas.bin";
    store.delta_index_path_ = run_dir / "deltas.idx";
    store.frame_index_path_ = run_dir / "frame_index.bin";
    store.metrics_path_ = run_dir / "metrics.csv";
    store.width_ = initial_state.width;
    store.height_ = initial_state.height;
    store.snapshot_interval_ = std::max<std::uint32_t>(1U, snapshot_interval);
    store.write_meta(run_dir, initial_state, store.snapshot_interval_);
    store.write_file(store.deltas_path_, {});
    store.write_file(store.delta_index_path_, {});
    store.write_file(store.frame_index_path_, {});
    store.write_file(store.metrics_path_, { });
    store.write_snapshot(initial_state);
    return store;
}

VersionStore VersionStore::open_existing(const std::filesystem::path& run_dir) {
    return load_files(run_dir);
}

void VersionStore::write_meta(
    const std::filesystem::path& run_dir,
    const CanvasState& state,
    const std::uint32_t interval) {
    BinaryWriter writer;
    writer.u32(kRunMagic);
    writer.u16(kSchema);
    writer.u16(0U);
    writer.u32(static_cast<std::uint32_t>(state.width));
    writer.u32(static_cast<std::uint32_t>(state.height));
    writer.u32(interval);
    writer.u64(state.version);
    write_anchor_camera(writer, state.anchor_camera);
    write_file(run_dir / "run_meta.bin", writer.data());
}

VersionStore VersionStore::load_files(const std::filesystem::path& run_dir) {
    if (!std::filesystem::is_directory(run_dir)) {
        throw std::runtime_error("history run directory does not exist: " + run_dir.string());
    }
    VersionStore store;
    store.run_dir_ = run_dir;
    store.deltas_path_ = run_dir / "deltas.bin";
    store.delta_index_path_ = run_dir / "deltas.idx";
    store.frame_index_path_ = run_dir / "frame_index.bin";
    store.metrics_path_ = run_dir / "metrics.csv";
    store.load_meta();
    store.load_indices();
    store.load_snapshots();
    if (!store.snapshots_.empty()) {
        store.latest_version_ = std::max(store.latest_version_, store.snapshots_.back().first);
    }
    if (!store.frames_.empty()) {
        for (const FrameRecord& record : store.frames_) {
            store.latest_frame_ = std::max(store.latest_frame_, record.frame_seq);
            store.latest_version_ = std::max(store.latest_version_, record.commit_version);
        }
    }
    return store;
}

void VersionStore::load_meta() {
    const std::vector<std::uint8_t> data = read_file(run_dir_ / "run_meta.bin");
    BinaryReader reader(data);
    if (reader.u32() != kRunMagic) {
        throw std::runtime_error("invalid run_meta.bin magic");
    }
    if (reader.u16() != kSchema) {
        throw std::runtime_error("unsupported run_meta.bin schema");
    }
    (void)reader.u16();
    width_ = static_cast<int>(reader.u32());
    height_ = static_cast<int>(reader.u32());
    snapshot_interval_ = std::max<std::uint32_t>(1U, reader.u32());
    latest_version_ = reader.u64();
    (void)read_anchor_camera(reader);
    if (width_ <= 0 || height_ <= 0) {
        throw std::runtime_error("run_meta.bin contains invalid dimensions");
    }
}

void VersionStore::load_indices() {
    const std::vector<std::uint8_t> delta_data = read_file(delta_index_path_);
    if (delta_data.size() % 20U != 0U) {
        throw std::runtime_error("deltas.idx has a partial entry");
    }
    BinaryReader delta_reader(delta_data);
    while (delta_reader.remaining() != 0U) {
        DeltaIndexEntry entry;
        entry.version = delta_reader.u64();
        entry.file_offset = delta_reader.u64();
        entry.record_size = delta_reader.u32();
        delta_index_.push_back(entry);
    }

    const std::vector<std::uint8_t> frame_data = read_file(frame_index_path_);
    BinaryReader frame_reader(frame_data);
    while (frame_reader.remaining() != 0U) {
        FrameRecord record = read_frame_record(frame_reader);
        if (frame_reader.remaining() < sizeof(std::uint64_t)) {
            throw std::runtime_error("frame_index.bin has a partial delta offset");
        }
        const std::uint64_t delta_offset = frame_reader.u64();
        (void)delta_offset;
        frames_.push_back(std::move(record));
    }
}

void VersionStore::load_snapshots() {
    const std::filesystem::path snapshots_dir = run_dir_ / "snapshots";
    if (!std::filesystem::is_directory(snapshots_dir)) {
        throw std::runtime_error("history is missing snapshots directory");
    }
    for (const auto& entry : std::filesystem::directory_iterator(snapshots_dir)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".bin") {
            continue;
        }
        const std::string stem = entry.path().stem().string();
        constexpr const char* prefix = "snapshot_";
        if (stem.rfind(prefix, 0U) != 0U) {
            continue;
        }
        try {
            const CommitVersion version = static_cast<CommitVersion>(std::stoull(stem.substr(9U)));
            snapshots_.emplace_back(version, entry.path());
        } catch (const std::exception&) {
            throw std::runtime_error("invalid snapshot file name: " + entry.path().string());
        }
    }
    std::sort(snapshots_.begin(), snapshots_.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.first < rhs.first;
    });
}

std::filesystem::path VersionStore::snapshot_path(const CommitVersion version) const {
    return run_dir_ / "snapshots" / version_file_name(version);
}

std::optional<DeltaIndexEntry> VersionStore::find_delta_index(const CommitVersion version) const {
    const auto it = std::lower_bound(
        delta_index_.begin(), delta_index_.end(), version, [](const DeltaIndexEntry& entry, const CommitVersion value) {
            return entry.version < value;
        });
    if (it == delta_index_.end() || it->version != version) {
        return std::nullopt;
    }
    return *it;
}

std::optional<std::pair<CommitVersion, std::filesystem::path>> VersionStore::find_snapshot(
    const CommitVersion version) const {
    std::optional<std::pair<CommitVersion, std::filesystem::path>> result;
    for (const auto& snapshot : snapshots_) {
        if (snapshot.first > version) {
            break;
        }
        result = snapshot;
    }
    return result;
}

std::uint64_t VersionStore::append_delta(const PointCloudDelta& delta) {
    if (delta.to_version <= delta.from_version) {
        throw std::invalid_argument("cannot append a no-op delta");
    }
    BinaryWriter payload_writer;
    write_delta(payload_writer, delta);
    const std::vector<std::uint8_t>& payload = payload_writer.data();
    if (payload.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::length_error("delta payload is too large");
    }
    BinaryWriter header;
    header.u32(kDeltaMagic);
    header.u16(kSchema);
    header.u16(0U);
    header.u64(delta.frame_seq);
    header.u64(delta.from_version);
    header.u64(delta.to_version);
    header.u32(static_cast<std::uint32_t>(delta.changes.size()));
    header.u32(static_cast<std::uint32_t>(payload.size()));
    header.u32(static_cast<std::uint32_t>(payload.size()));
    header.u32(crc32(payload));
    if (header.data().size() != kDeltaHeaderSize) {
        throw std::logic_error("delta header size changed unexpectedly");
    }

    std::error_code file_size_error;
    const std::uintmax_t existing_size = std::filesystem::exists(deltas_path_)
        ? std::filesystem::file_size(deltas_path_, file_size_error)
        : 0U;
    if (file_size_error) {
        throw std::runtime_error("failed to determine deltas.bin append offset: " + file_size_error.message());
    }
    const std::uint64_t offset = static_cast<std::uint64_t>(existing_size);

    std::ofstream out(deltas_path_, std::ios::binary | std::ios::app);
    if (!out) {
        throw std::runtime_error("failed to open deltas.bin for append");
    }
    out.write(reinterpret_cast<const char*>(header.data().data()), static_cast<std::streamsize>(header.data().size()));
    out.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
    out.flush();
    if (!out) {
        throw std::runtime_error("failed to append deltas.bin");
    }

    const std::uint32_t record_size = static_cast<std::uint32_t>(header.data().size() + payload.size());
    BinaryWriter index_writer;
    index_writer.u64(delta.to_version);
    index_writer.u64(offset);
    index_writer.u32(record_size);
    std::ofstream index_out(delta_index_path_, std::ios::binary | std::ios::app);
    if (!index_out) {
        throw std::runtime_error("failed to open deltas.idx for append");
    }
    index_out.write(
        reinterpret_cast<const char*>(index_writer.data().data()),
        static_cast<std::streamsize>(index_writer.data().size()));
    index_out.flush();
    if (!index_out) {
        throw std::runtime_error("failed to append deltas.idx");
    }
    delta_index_.push_back(DeltaIndexEntry{delta.to_version, offset, record_size});
    latest_version_ = std::max(latest_version_, delta.to_version);
    return offset;
}

void VersionStore::append_frame(const FrameRecord& record, const std::uint64_t delta_offset) {
    BinaryWriter writer;
    write_frame_record(writer, record);
    writer.u64(delta_offset);
    std::ofstream out(frame_index_path_, std::ios::binary | std::ios::app);
    if (!out) {
        throw std::runtime_error("failed to open frame_index.bin for append");
    }
    out.write(reinterpret_cast<const char*>(writer.data().data()), static_cast<std::streamsize>(writer.data().size()));
    out.flush();
    if (!out) {
        throw std::runtime_error("failed to append frame_index.bin");
    }
    frames_.push_back(record);
    latest_frame_ = std::max(latest_frame_, record.frame_seq);
    latest_version_ = std::max(latest_version_, record.commit_version);
}

void VersionStore::append_metrics(const std::string& csv_line) {
    std::ofstream out(metrics_path_, std::ios::app);
    if (!out) {
        throw std::runtime_error("failed to open metrics.csv for append");
    }
    out << csv_line;
    if (csv_line.empty() || csv_line.back() != '\n') {
        out << '\n';
    }
}

bool VersionStore::should_snapshot(const PointCloudDelta& delta) const noexcept {
    return delta.to_version != 0U && delta.to_version % snapshot_interval_ == 0U;
}

void VersionStore::write_snapshot(const CanvasState& state) {
    if (!state.shape_valid()) {
        throw std::invalid_argument("cannot snapshot an invalid CanvasState");
    }
    const std::filesystem::path final_path = snapshot_path(state.version);
    if (std::filesystem::exists(final_path)) {
        return;
    }
    BinaryWriter writer;
    ::omnivggt::observer::write_snapshot(writer, Snapshot{state});
    const std::filesystem::path temp_path = final_path.string() + ".tmp";
    write_file(temp_path, writer.data());
    std::error_code error;
    std::filesystem::rename(temp_path, final_path, error);
    if (error) {
        std::filesystem::remove(temp_path);
        throw std::runtime_error("failed to publish snapshot: " + error.message());
    }
    snapshots_.emplace_back(state.version, final_path);
    std::sort(snapshots_.begin(), snapshots_.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.first < rhs.first;
    });
}

PointCloudDelta VersionStore::read_delta(const CommitVersion version) const {
    const auto entry = find_delta_index(version);
    if (!entry.has_value()) {
        throw std::runtime_error("delta version is not indexed: " + std::to_string(version));
    }
    std::ifstream in(deltas_path_, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open deltas.bin for read");
    }
    in.seekg(static_cast<std::streamoff>(entry->file_offset));
    std::vector<std::uint8_t> header_data(kDeltaHeaderSize);
    in.read(reinterpret_cast<char*>(header_data.data()), static_cast<std::streamsize>(header_data.size()));
    if (!in) {
        throw std::runtime_error("truncated delta header");
    }
    BinaryReader header_reader(header_data);
    if (header_reader.u32() != kDeltaMagic || header_reader.u16() != kSchema) {
        throw std::runtime_error("invalid delta header");
    }
    (void)header_reader.u16();
    const FrameSeq frame_seq = header_reader.u64();
    const CommitVersion from_version = header_reader.u64();
    const CommitVersion to_version = header_reader.u64();
    const std::uint32_t change_count = header_reader.u32();
    const std::uint32_t uncompressed_size = header_reader.u32();
    const std::uint32_t compressed_size = header_reader.u32();
    const std::uint32_t expected_crc = header_reader.u32();
    if (compressed_size != uncompressed_size || compressed_size > entry->record_size) {
        throw std::runtime_error("unsupported compressed delta payload");
    }
    std::vector<std::uint8_t> payload(compressed_size);
    in.read(reinterpret_cast<char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
    if (!in || crc32(payload) != expected_crc) {
        throw std::runtime_error("delta payload CRC check failed");
    }
    BinaryReader payload_reader(payload);
    PointCloudDelta delta = ::omnivggt::observer::read_delta(payload_reader);
    if (payload_reader.remaining() != 0U
        || delta.frame_seq != frame_seq
        || delta.from_version != from_version
        || delta.to_version != to_version
        || delta.changes.size() != change_count) {
        throw std::runtime_error("delta payload/header mismatch");
    }
    return delta;
}

Snapshot VersionStore::load_snapshot_for_version(const CommitVersion version) const {
    const auto entry = find_snapshot(version);
    if (!entry.has_value()) {
        throw std::runtime_error("history contains no snapshot at or before version " + std::to_string(version));
    }
    const std::vector<std::uint8_t> data = read_file(entry->second);
    BinaryReader reader(data);
    Snapshot snapshot = read_snapshot(reader);
    if (reader.remaining() != 0U || snapshot.state.version != entry->first) {
        throw std::runtime_error("snapshot file/version mismatch");
    }
    return snapshot;
}

CanvasState VersionStore::recover_state() const {
    Snapshot snapshot = load_snapshot_for_version(latest_version_);
    CanvasState state = std::move(snapshot.state);
    for (CommitVersion version = state.version + 1U; version <= latest_version_; ++version) {
        const PointCloudDelta delta = read_delta(version);
        apply_delta_forward(state, delta);
    }
    if (!frames_.empty()) {
        state.last_frame = frames_.back().frame_seq;
    }
    return state;
}

ReplayBundle VersionStore::build_replay(const FrameSeq target_frame) const {
    ReplayBundle bundle;
    bundle.target_frame = target_frame;
    const FrameRecord record = frame_at_or_before(target_frame);
    bundle.target_version = record.commit_version;
    bundle.snapshot = load_snapshot_for_version(bundle.target_version);
    for (CommitVersion version = bundle.snapshot.state.version + 1U; version <= bundle.target_version; ++version) {
        bundle.deltas.push_back(read_delta(version));
    }
    return bundle;
}

std::optional<FrameRecord> VersionStore::find_frame(const FrameSeq frame_seq) const {
    const auto it = std::find_if(frames_.begin(), frames_.end(), [frame_seq](const FrameRecord& record) {
        return record.frame_seq == frame_seq;
    });
    if (it == frames_.end()) {
        return std::nullopt;
    }
    return *it;
}

FrameRecord VersionStore::frame_at_or_before(const FrameSeq frame_seq) const {
    FrameRecord result;
    bool found = false;
    for (const FrameRecord& record : frames_) {
        if (record.frame_seq <= frame_seq && (!found || record.frame_seq > result.frame_seq)) {
            result = record;
            found = true;
        }
    }
    if (!found) {
        result.frame_seq = 0;
        result.commit_version = 0;
        result.base_version = 0;
        result.status = FrameStatus::NoChange;
    }
    return result;
}

std::vector<std::uint8_t> VersionStore::read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to read history file: " + path.string());
    }
    in.seekg(0, std::ios::end);
    const std::streamoff length = in.tellg();
    if (length < 0) {
        throw std::runtime_error("failed to determine file size: " + path.string());
    }
    in.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> data(static_cast<std::size_t>(length));
    if (!data.empty()) {
        in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
    }
    if (!in && !data.empty()) {
        throw std::runtime_error("failed to read history file: " + path.string());
    }
    return data;
}

void VersionStore::write_file(const std::filesystem::path& path, const std::vector<std::uint8_t>& data) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to write history file: " + path.string());
    }
    if (!data.empty()) {
        out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    }
    out.flush();
    if (!out) {
        throw std::runtime_error("failed to write history file: " + path.string());
    }
}

}  // namespace omnivggt::observer
