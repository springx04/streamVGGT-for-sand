#include "bounded_queue.hpp"
#include "frame_source.hpp"
#include "history_compactor.hpp"
#include "inference_pipeline.hpp"
#include "inflight_gate.hpp"
#include "protocol.hpp"
#include "tcp_transport.hpp"
#include "version_store.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;
using namespace omnivggt::observer;

namespace {

std::atomic<bool> g_stop_requested{false};

void on_signal(int) { g_stop_requested.store(true); }

struct ServerArgs {
    InferenceOptions inference;
    fs::path image_dir;
    fs::path output_dir = "setc/observer_output";
    fs::path run_dir;
    std::uint64_t num_images = 0;
    std::uint16_t port = 37651;
    std::uint32_t snapshot_interval = 60;
    std::size_t history_keep_groups = 0U;
    std::size_t queue_capacity = 3;
    int poll_ms = 50;
    bool once = false;
    bool resume = false;
    std::size_t input_group_size = 3U;
    std::size_t input_group_stride = 1U;
    std::size_t group_anchor_index = 1U;
};

std::string require_value(int& index, const int argc, char** argv, const std::string& key) {
    if (index + 1 >= argc) {
        throw std::runtime_error("missing value for " + key);
    }
    return argv[++index];
}

void usage() {
    std::cerr
        << "Usage:\n"
        << "  omnivggt_stream_server --model model.pt --image_dir images [options]\n\n"
        << "Options:\n"
        << "  --model-pair model.pt  Two-frame TorchScript graph for later frames.\n"
        << "  --model-pair-dir DIR  Dynamic two-frame bucket artifacts (WxH in filename).\n"
        << "  --pair-letterbox      Use one pair graph with aspect-preserving edge padding.\n"
        << "  --model-group3 model.pt  Independent B=3,S=1 three-image graph.\n"
        << "  --input-group-size N  Logical input group size (1 or 3), default 3.\n"
        << "  --input-group-stride N Sliding group stride, default 1.\n"
        << "  --group-anchor-index N Anchor within a three-image group, default 1.\n"
        << "  --group-model-width W --group-model-height H  B=3,S=1 graph dimensions.\n"
        << "  --output_dir DIR       Parent directory for independent run history.\n"
        << "  --run_dir DIR          Explicit run directory; use --resume to reopen it.\n"
        << "  --num_images N         Process at most N images (0 means watch continuously).\n"
        << "  --once                 Stop after the current directory has been consumed.\n"
        << "  --resume               Recover CanvasState and history from --run_dir.\n"
        << "  --port N               Viewer TCP port, default 37651.\n"
        << "  --queue_capacity N     Latest-frame queue capacity, default 3.\n"
        << "  --queue-capacity N     Hyphenated alias; use a large value for offline replay.\n"
        << "  --poll_ms N            Directory polling interval, default 50.\n"
        << "  --snapshot_interval N  Snapshot every N committed versions, default 60.\n"
        << "  --history-keep-groups N  Keep approximately the newest N logical groups.\n"
        << "  --height H --width W   Fixed model/canvas dimensions, default 518x518.\n"
        << "  --target-size H --target-width W  Python-launcher compatible aliases.\n"
        << "  --canvas-width W --canvas-height H  Padded aligned-canvas dimensions.\n"
        << "  --first-model-width W --first-model-height H  First-frame model bucket.\n"
        << "  --device cuda          GPU inference only.\n"
        << "  --dtype float32|float16|bfloat16\n"
        << "  --no_change_ratio V    Skip inference below this change ratio.\n"
        << "  --min_conf V           Minimum confidence for replacement points.\n"
        << "  --image_l1_thr V       Photometric difference threshold.\n"
        << "  --dilate_ksize K       Change mask dilation size.\n"
        << "  --scene_jump_ratio V   Large-change marker threshold.\n"
        << "  --save_debug           Write warped/change masks under the run directory.\n";
}

ServerArgs parse_args(const int argc, char** argv) {
    ServerArgs args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--model") {
            args.inference.model = require_value(i, argc, argv, key);
        } else if (key == "--model-group3" || key == "--model_group3") {
            args.inference.group_model = require_value(i, argc, argv, key);
        } else if (key == "--model-pair" || key == "--model_pair") {
            args.inference.pair_model = require_value(i, argc, argv, key);
        } else if (key == "--model-pair-dir" || key == "--model_pair_dir") {
            args.inference.pair_model_dir = require_value(i, argc, argv, key);
        } else if (key == "--pair-letterbox") {
            args.inference.pair_letterbox = true;
        } else if (key == "--image_dir" || key == "--image-dir") {
            args.image_dir = require_value(i, argc, argv, key);
        } else if (key == "--output_dir" || key == "--output-dir") {
            args.output_dir = require_value(i, argc, argv, key);
        } else if (key == "--run_dir") {
            args.run_dir = require_value(i, argc, argv, key);
        } else if (key == "--num_images") {
            args.num_images = std::stoull(require_value(i, argc, argv, key));
        } else if (key == "--input-group-size" || key == "--input_group_size") {
            args.input_group_size = static_cast<std::size_t>(std::stoull(require_value(i, argc, argv, key)));
        } else if (key == "--input-group-stride" || key == "--input_group_stride") {
            args.input_group_stride = static_cast<std::size_t>(std::stoull(require_value(i, argc, argv, key)));
        } else if (key == "--group-anchor-index" || key == "--group_anchor_index") {
            args.group_anchor_index = static_cast<std::size_t>(std::stoull(require_value(i, argc, argv, key)));
        } else if (key == "--group-model-width" || key == "--group_model_width") {
            args.inference.group_width = std::stoi(require_value(i, argc, argv, key));
        } else if (key == "--group-model-height" || key == "--group_model_height") {
            args.inference.group_height = std::stoi(require_value(i, argc, argv, key));
        } else if (key == "--port") {
            args.port = static_cast<std::uint16_t>(std::stoul(require_value(i, argc, argv, key)));
        } else if (key == "--queue_capacity" || key == "--queue-capacity") {
            args.queue_capacity = static_cast<std::size_t>(std::stoull(require_value(i, argc, argv, key)));
        } else if (key == "--poll_ms") {
            args.poll_ms = std::stoi(require_value(i, argc, argv, key));
        } else if (key == "--snapshot_interval") {
            args.snapshot_interval = static_cast<std::uint32_t>(std::stoul(require_value(i, argc, argv, key)));
        } else if (key == "--history-keep-groups") {
            args.history_keep_groups = static_cast<std::size_t>(
                std::stoull(require_value(i, argc, argv, key)));
        } else if (key == "--height" || key == "--target-size") {
            args.inference.height = std::stoi(require_value(i, argc, argv, key));
        } else if (key == "--width" || key == "--target-width") {
            args.inference.width = std::stoi(require_value(i, argc, argv, key));
        } else if (key == "--canvas-width") {
            args.inference.canvas_width = std::stoi(require_value(i, argc, argv, key));
        } else if (key == "--canvas-height") {
            args.inference.canvas_height = std::stoi(require_value(i, argc, argv, key));
        } else if (key == "--first-model-width") {
            args.inference.first_model_width = std::stoi(require_value(i, argc, argv, key));
        } else if (key == "--first-model-height") {
            args.inference.first_model_height = std::stoi(require_value(i, argc, argv, key));
        } else if (key == "--device") {
            args.inference.device = require_value(i, argc, argv, key);
        } else if (key == "--dtype") {
            args.inference.dtype = require_value(i, argc, argv, key);
        } else if (key == "--image_l1_thr") {
            args.inference.image_l1_thr = std::stod(require_value(i, argc, argv, key));
        } else if (key == "--no_change_ratio") {
            args.inference.no_change_ratio = std::stod(require_value(i, argc, argv, key));
        } else if (key == "--scene_jump_ratio") {
            args.inference.scene_jump_ratio = std::stod(require_value(i, argc, argv, key));
        } else if (key == "--min_conf") {
            args.inference.min_conf = std::stod(require_value(i, argc, argv, key));
        } else if (key == "--dilate_ksize") {
            args.inference.dilate_ksize = std::stoi(require_value(i, argc, argv, key));
        } else if (key == "--once") {
            args.once = true;
        } else if (key == "--resume") {
            args.resume = true;
        } else if (key == "--save_debug") {
            args.inference.save_debug = true;
        } else if (key == "--no-save-debug") {
            args.inference.save_debug = false;
        } else if (key == "--help" || key == "-h") {
            usage();
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }
    if (args.image_dir.empty()) {
        usage();
        throw std::runtime_error("--image_dir is required");
    }
    if (args.input_group_size != 1U && args.input_group_size != 3U) {
        throw std::runtime_error("--input-group-size must be 1 or 3");
    }
    if (args.input_group_stride == 0U || args.input_group_stride > args.input_group_size) {
        throw std::runtime_error("--input-group-stride must be in [1, input-group-size]");
    }
    if (args.input_group_size == 3U) {
        if (args.group_anchor_index >= args.input_group_size) {
            throw std::runtime_error("--group-anchor-index must be 0, 1 or 2");
        }
        if (args.inference.group_model.empty()) {
            if (args.inference.model.empty()) {
                throw std::runtime_error("--model is required for three-image observation mode");
            }
            if (!args.inference.pair_model.empty() && !fs::is_regular_file(args.inference.pair_model)) {
                throw std::runtime_error("--model-pair is not a regular file: "
                    + args.inference.pair_model);
            }
            args.inference.group_mode = false;
            args.inference.group_observation_mode = true;
        } else if (!fs::is_regular_file(args.inference.group_model)) {
            throw std::runtime_error("--model-group3 is not a regular file: "
                + args.inference.group_model);
        } else if (args.inference.group_width <= 0 || args.inference.group_height <= 0
            || args.inference.group_width % 14 != 0 || args.inference.group_height % 14 != 0) {
            throw std::runtime_error("group model dimensions must be positive multiples of 14");
        } else {
            args.inference.group_mode = true;
            args.inference.group_observation_mode = false;
        }
    } else if (args.inference.model.empty()) {
        throw std::runtime_error("--model is required when --input-group-size=1");
    } else {
        args.group_anchor_index = 0U;
        args.inference.group_mode = false;
        args.inference.group_observation_mode = false;
    }
    if (!fs::is_directory(args.image_dir)) {
        throw std::runtime_error("--image_dir is not a directory: " + args.image_dir.string());
    }
    if (args.inference.width <= 0 || args.inference.height <= 0
        || args.inference.width % 14 != 0 || args.inference.height % 14 != 0) {
        throw std::runtime_error("--width and --height must be positive multiples of 14");
    }
    if (args.inference.canvas_width <= 0) {
        args.inference.canvas_width = args.inference.width;
    }
    if (args.inference.canvas_height <= 0) {
        args.inference.canvas_height = args.inference.height;
    }
    if (args.inference.first_model_width <= 0) {
        args.inference.first_model_width = args.inference.width;
    }
    if (args.inference.first_model_height <= 0) {
        args.inference.first_model_height = args.inference.height;
    }
    if (args.inference.canvas_width <= 0 || args.inference.canvas_height <= 0
        || args.inference.first_model_width % 14 != 0
        || args.inference.first_model_height % 14 != 0) {
        throw std::runtime_error("canvas and first-frame model dimensions must be positive; model dimensions must be multiples of 14");
    }
    if (args.resume && args.run_dir.empty()) {
        throw std::runtime_error("--resume requires --run_dir");
    }
    return args;
}

fs::path make_run_dir(const ServerArgs& args) {
    if (!args.run_dir.empty()) {
        return args.run_dir;
    }
    fs::create_directories(args.output_dir);
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm local_time{};
#ifdef _WIN32
    localtime_s(&local_time, &time);
#else
    localtime_r(&time, &local_time);
#endif
    std::ostringstream name;
    name << "run_" << std::put_time(&local_time, "%Y%m%d_%H%M%S");
    fs::path candidate = args.output_dir / name.str();
    int suffix = 1;
    while (fs::exists(candidate)) {
        candidate = args.output_dir / (name.str() + "_" + std::to_string(suffix++));
    }
    return candidate;
}

class ClientSession;

class StreamRuntime {
public:
    StreamRuntime(const ServerArgs& args, const fs::path& run_dir)
        : args_(args), run_dir_(run_dir) {
        group_manifest_.open(run_dir_ / "input_groups.csv", std::ios::out | std::ios::app);
        if (!group_manifest_) {
            throw std::runtime_error("failed to open input_groups.csv under " + run_dir_.string());
        }
        if (group_manifest_.tellp() == std::streampos(0)) {
            group_manifest_ << "frame_seq,group_key,source_seq_0,image_0,source_seq_1,image_1,"
                "source_seq_2,image_2,anchor_index,status\n";
        }
        if (args.resume) {
            store_ = VersionStore::open_existing(run_dir_);
            if (store_.width() != args.inference.canvas_width || store_.height() != args.inference.canvas_height) {
                throw std::runtime_error("resume dimensions do not match server dimensions");
            }
            state_ = store_.recover_state();
            live_head_frame_.store(store_.latest_frame());
        } else {
            state_.reset(args.inference.canvas_width, args.inference.canvas_height);
            store_ = VersionStore::create_new(run_dir_, state_, args.snapshot_interval);
            store_.append_metrics(
                "frame_seq,image,total_ms,read_ms,align2d_ms,diff_ms,model_ms,depth_align_ms,patch_ms,"
                "changed_ratio,changed_point_count,valid_point_count,homography_inliers,homography_error_px,"
                "roi_width,roi_height,model_input_width,model_input_height,"
                "photometric_changed_ratio,support_changed_ratio,skipped_model,fallback,"
                "group_size,group_stride,group_anchor_index,forward_calls,forward_batch_size,"
                "forward_sequence_size,group_fused_sources,group_rejected_sources,group_max_depth_residual");
        }
    }

    CanvasState snapshot() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return state_;
    }

    HelloMessage hello() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return HelloMessage{
            state_.width,
            state_.height,
            live_head_frame_.load(),
            state_.version,
            run_dir_.filename().string()};
    }

    Snapshot current_snapshot() const { return Snapshot{snapshot()}; }

    FrameSeq next_frame_sequence() const {
        std::lock_guard<std::mutex> lock(history_mutex_);
        return store_.frame_records().empty() ? 0U : store_.latest_frame() + 1U;
    }

    std::unordered_set<std::string> completed_image_names() const {
        std::lock_guard<std::mutex> lock(history_mutex_);
        std::unordered_set<std::string> names;
        for (const FrameRecord& record : store_.frame_records()) {
            if ((record.status == FrameStatus::Committed || record.status == FrameStatus::NoChange)
                && !record.image_name.empty()) {
                names.insert(record.image_name);
            }
        }
        return names;
    }

    std::unordered_set<std::string> completed_group_keys() {
        std::lock_guard<std::mutex> lock(group_manifest_mutex_);
        group_manifest_.flush();
        std::ifstream input(run_dir_ / "input_groups.csv");
        std::unordered_map<std::string, std::string> latest_status;
        std::string line;
        if (!std::getline(input, line)) {
            return {};
        }
        while (std::getline(input, line)) {
            std::size_t cursor = 0;
            std::string ignored_frame;
            std::string group_key;
            if (!read_csv_field(line, cursor, ignored_frame)
                || !read_csv_field(line, cursor, group_key)) {
                continue;
            }
            const std::size_t status_separator = line.rfind(',');
            if (status_separator == std::string::npos
                || status_separator + 1U >= line.size()) {
                continue;
            }
            latest_status[group_key] = line.substr(status_separator + 1U);
        }
        std::unordered_set<std::string> completed;
        for (const auto& [group_key, status] : latest_status) {
            if (status == "Committed" || status == "NoChange") {
                completed.insert(group_key);
            }
        }
        return completed;
    }

    bool has_group_manifest_entries() {
        std::lock_guard<std::mutex> lock(group_manifest_mutex_);
        group_manifest_.flush();
        std::ifstream input(run_dir_ / "input_groups.csv");
        std::string line;
        return static_cast<bool>(std::getline(input, line))
            && static_cast<bool>(std::getline(input, line))
            && !line.empty();
    }

    ReplayBundle replay(const FrameSeq frame_seq) const {
        std::lock_guard<std::mutex> lock(history_mutex_);
        return store_.build_replay(frame_seq);
    }

    CommitVersion latest_snapshot_version() const {
        std::lock_guard<std::mutex> lock(history_mutex_);
        return store_.latest_snapshot_version();
    }

    void note_received(const RawFrame& raw) {
        append_group_manifest(raw, "Received");
        const CommitVersion version = current_version();
        live_head_frame_.store(std::max(live_head_frame_.load(), raw.frame_seq));
        FrameRecord record;
        record.frame_seq = raw.frame_seq;
        record.base_version = version;
        record.commit_version = version;
        record.status = FrameStatus::Received;
        record.image_name = raw.path.filename().string();
        publish(make_frame_status(record));
        publish(make_live_head(raw.frame_seq, version));
    }

    void record_coalesced(const RawFrame& raw) {
        append_group_manifest(raw, "Coalesced");
        FrameRecord record;
        record.frame_seq = raw.frame_seq;
        record.base_version = current_version();
        record.commit_version = record.base_version;
        record.status = FrameStatus::Coalesced;
        record.image_name = raw.path.filename().string();
        {
            std::lock_guard<std::mutex> lock(history_mutex_);
            store_.append_frame(record);
        }
        live_head_frame_.store(std::max(live_head_frame_.load(), raw.frame_seq));
        publish(make_frame_status(record));
        publish(make_live_head(raw.frame_seq, record.commit_version));
    }

    void record_failed(const RawFrame& raw, const std::string& message) {
        append_group_manifest(raw, "Failed");
        FrameRecord record;
        record.frame_seq = raw.frame_seq;
        record.base_version = current_version();
        record.commit_version = record.base_version;
        record.status = FrameStatus::Failed;
        record.image_name = raw.path.filename().string();
        {
            std::lock_guard<std::mutex> lock(history_mutex_);
            store_.append_frame(record);
        }
        std::cerr << "frame=" << raw.frame_seq << " failed: " << message << "\n";
        publish(make_frame_status(record));
        publish(make_live_head(raw.frame_seq, record.commit_version));
        mark_frame_handled(raw.frame_seq);
    }

    void commit(CandidateCommit candidate) {
        if (!candidate.has_patch) {
            append_group_status(candidate.frame.frame_seq,
                candidate.frame.status == FrameStatus::NoChange ? "NoChange" : "Committed");
            commit_no_change(candidate);
            return;
        }

        PointCloudDelta delta;
        CanvasState committed_state;
        try {
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                if (state_.version != candidate.patch.base_version) {
                    throw std::runtime_error("stale CandidatePatch base_version");
                }
                delta = commit_patch(state_, candidate.patch);
                committed_state = state_;
            }
        } catch (const std::exception& error) {
            RawFrame raw = pending_group_or_default(candidate.frame.frame_seq, candidate.frame.image_name);
            record_failed(raw, error.what());
            return;
        }

        if (delta.to_version == delta.from_version) {
            candidate.frame.status = FrameStatus::NoChange;
            candidate.frame.base_version = delta.from_version;
            candidate.frame.commit_version = delta.from_version;
            append_group_status(candidate.frame.frame_seq, "NoChange");
            commit_no_change(candidate);
            return;
        }

        candidate.frame.status = FrameStatus::Committed;
        candidate.frame.base_version = delta.from_version;
        candidate.frame.commit_version = delta.to_version;
        candidate.frame.changed_point_count = static_cast<std::uint32_t>(delta.changes.size());
        candidate.frame.valid_point_count = delta.valid_point_count;
        append_group_status(candidate.frame.frame_seq, "Committed");
        std::uint64_t delta_offset = 0;
        {
            std::lock_guard<std::mutex> lock(history_mutex_);
            delta_offset = store_.append_delta(delta);
            store_.append_frame(candidate.frame, delta_offset);
            store_.append_metrics(candidate.metrics.csv_line());
            if (store_.should_snapshot(delta)) {
                store_.write_snapshot(committed_state);
                if (args_.history_keep_groups != 0U) {
                    HistoryCompactor::compact(store_, args_.history_keep_groups);
                }
            }
        }
        live_head_frame_.store(std::max(live_head_frame_.load(), candidate.frame.frame_seq));
        publish(make_frame_status(candidate.frame));
        publish(make_delta(MessageType::Delta, delta));
        publish(make_live_head(candidate.frame.frame_seq, candidate.frame.commit_version));
        mark_frame_handled(candidate.frame.frame_seq);
        std::cout << "frame=" << candidate.frame.frame_seq
                  << " status=Committed version=" << candidate.frame.commit_version
                  << " changed=" << std::fixed << std::setprecision(4) << candidate.frame.changed_ratio
                  << " points=" << candidate.frame.valid_point_count << " delta=" << delta.changes.size() << "\n";
    }

    void add_client(const std::shared_ptr<ClientSession>& client) {
        std::lock_guard<std::mutex> lock(client_mutex_);
        clients_.push_back(client);
    }

    void remove_dead_clients();
    void publish(const Packet& packet);

    CommitVersion current_version() const {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return state_.version;
    }

    void request_stop() { g_stop_requested.store(true); }

    // A candidate is built from a snapshot of the authoritative Canvas.  Do
    // not let the inference worker build the next candidate until the commit
    // worker has handled this one; otherwise two consecutive candidates can
    // carry the same base_version and the second one is rejected as stale.
    void wait_until_frame_handled(const FrameSeq frame_seq) {
        std::unique_lock<std::mutex> lock(commit_ack_mutex_);
        commit_ack_condition_.wait(lock, [this, frame_seq] {
            return handled_frames_.find(frame_seq) != handled_frames_.end()
                || g_stop_requested.load();
        });
        handled_frames_.erase(frame_seq);
    }

private:
    static std::string csv_escape(const std::string& value) {
        std::string escaped = value;
        std::size_t position = 0;
        while ((position = escaped.find('"', position)) != std::string::npos) {
            escaped.insert(position, 1, '"');
            position += 2;
        }
        return "\"" + escaped + "\"";
    }

    static bool read_csv_field(
        const std::string& line,
        std::size_t& cursor,
        std::string& value) {
        if (cursor >= line.size()) {
            return false;
        }
        if (line[cursor] != '"') {
            const std::size_t end = line.find(',', cursor);
            value = line.substr(cursor, end == std::string::npos ? end : end - cursor);
            cursor = end == std::string::npos ? line.size() : end + 1U;
            return true;
        }
        ++cursor;
        value.clear();
        while (cursor < line.size()) {
            if (line[cursor] == '"') {
                if (cursor + 1U < line.size() && line[cursor + 1U] == '"') {
                    value.push_back('"');
                    cursor += 2U;
                    continue;
                }
                ++cursor;
                if (cursor < line.size() && line[cursor] == ',') {
                    ++cursor;
                }
                return true;
            }
            value.push_back(line[cursor++]);
        }
        return false;
    }

    void append_group_manifest(const RawFrame& raw, const char* status) {
        if (raw.group_paths.empty()) {
            return;
        }
        std::lock_guard<std::mutex> lock(group_manifest_mutex_);
        group_manifest_ << raw.frame_seq << ',' << csv_escape(raw.group_key);
        for (std::size_t index = 0; index < 3U; ++index) {
            const std::uint64_t source_seq = index < raw.group_source_seqs.size()
                ? raw.group_source_seqs[index] : static_cast<std::uint64_t>(index);
            group_manifest_ << ',' << source_seq << ','
                << csv_escape(index < raw.group_paths.size()
                    ? raw.group_paths[index].filename().string() : std::string());
        }
        group_manifest_ << ',' << raw.group_anchor_index << ',' << status << '\n';
        group_manifest_.flush();
        if (std::string(status) == "Received") {
            RawFrame metadata = raw;
            metadata.group_images.clear();
            pending_groups_[raw.frame_seq] = std::move(metadata);
        } else {
            pending_groups_.erase(raw.frame_seq);
        }
    }

    RawFrame pending_group_or_default(const FrameSeq frame_seq, const std::string& image_name) {
        std::lock_guard<std::mutex> lock(group_manifest_mutex_);
        const auto found = pending_groups_.find(frame_seq);
        if (found != pending_groups_.end()) {
            return found->second;
        }
        return RawFrame{frame_seq, image_name};
    }

    void append_group_status(const FrameSeq frame_seq, const char* status) {
        RawFrame raw;
        {
            std::lock_guard<std::mutex> lock(group_manifest_mutex_);
            const auto found = pending_groups_.find(frame_seq);
            if (found == pending_groups_.end()) {
                return;
            }
            raw = found->second;
        }
        append_group_manifest(raw, status);
    }

    void commit_no_change(CandidateCommit& candidate) {
        const CommitVersion version = current_version();
        candidate.frame.base_version = version;
        candidate.frame.commit_version = version;
        if (candidate.frame.valid_point_count == 0U) {
            const CanvasState state = snapshot();
            candidate.frame.valid_point_count = static_cast<std::uint32_t>(
                std::count(state.valid.begin(), state.valid.end(), static_cast<std::uint8_t>(1U)));
        }
        {
            std::lock_guard<std::mutex> lock(history_mutex_);
            store_.append_frame(candidate.frame);
            store_.append_metrics(candidate.metrics.csv_line());
        }
        live_head_frame_.store(std::max(live_head_frame_.load(), candidate.frame.frame_seq));
        publish(make_frame_status(candidate.frame));
        publish(make_live_head(candidate.frame.frame_seq, version));
        mark_frame_handled(candidate.frame.frame_seq);
        std::cout << "frame=" << candidate.frame.frame_seq << " status="
                  << frame_status_name(candidate.frame.status) << " version=" << version << "\n";
    }

    void mark_frame_handled(const FrameSeq frame_seq) {
        {
            std::lock_guard<std::mutex> lock(commit_ack_mutex_);
            handled_frames_.insert(frame_seq);
        }
        commit_ack_condition_.notify_all();
    }

    const ServerArgs& args_;
    fs::path run_dir_;
    mutable std::mutex state_mutex_;
    mutable std::mutex history_mutex_;
    mutable std::mutex client_mutex_;
    mutable std::mutex commit_ack_mutex_;
    std::condition_variable commit_ack_condition_;
    CanvasState state_;
    VersionStore store_;
    std::atomic<FrameSeq> live_head_frame_{0};
    std::unordered_set<FrameSeq> handled_frames_;
    std::vector<std::shared_ptr<ClientSession>> clients_;
    mutable std::mutex group_manifest_mutex_;
    std::ofstream group_manifest_;
    std::unordered_map<FrameSeq, RawFrame> pending_groups_;
};

class ClientSession : public std::enable_shared_from_this<ClientSession> {
public:
    using HelloProvider = std::function<HelloMessage()>;
    using SnapshotProvider = std::function<Snapshot()>;
    using ReplayProvider = std::function<ReplayBundle(FrameSeq)>;
    using ResyncVersionProvider = std::function<CommitVersion()>;

    ClientSession(
        TcpSocket socket,
        HelloProvider hello_provider,
        SnapshotProvider snapshot_provider,
        ReplayProvider replay_provider,
        ResyncVersionProvider resync_version_provider)
        : socket_(std::move(socket)),
          hello_provider_(std::move(hello_provider)),
          snapshot_provider_(std::move(snapshot_provider)),
          replay_provider_(std::move(replay_provider)),
          resync_version_provider_(std::move(resync_version_provider)),
          live_queue_(256U),
          replay_queue_(64U) {}

    ~ClientSession() {
        request_stop();
        join();
    }

    void start() {
        const HelloMessage hello = hello_provider_();
        enqueue_live(make_hello(hello));
        enqueue_live(make_snapshot(snapshot_provider_()));
        enqueue_live(make_live_head(hello.live_head_frame, hello.commit_version));
        writer_thread_ = std::thread(&ClientSession::writer_loop, this);
        reader_thread_ = std::thread(&ClientSession::reader_loop, this);
        replay_thread_ = std::thread(&ClientSession::replay_loop, this);
    }

    bool alive() const noexcept { return !stopping_.load(); }

    void enqueue_live(Packet packet) {
        if (stopping_.load()) {
            return;
        }
        const std::optional<Packet> dropped = live_queue_.push_latest(std::move(packet));
        if (dropped.has_value() && !stopping_.load()) {
            live_queue_.push_latest(make_resync_required(resync_version_provider_()));
            live_queue_.push_latest(make_snapshot(snapshot_provider_()));
        }
    }

    bool enqueue_replay(Packet packet) {
        if (!stopping_.load()) {
            const std::optional<Packet> dropped = replay_queue_.push_latest(std::move(packet));
            return !dropped.has_value();
        }
        return false;
    }

    void request_stop() {
        if (!stopping_.exchange(true)) {
            socket_.shutdown_both();
            live_queue_.close();
            replay_queue_.close();
            request_condition_.notify_all();
        }
    }

    void join() {
        if (reader_thread_.joinable()) {
            reader_thread_.join();
        }
        if (writer_thread_.joinable()) {
            writer_thread_.join();
        }
        if (replay_thread_.joinable()) {
            replay_thread_.join();
        }
    }

private:
    void writer_loop() {
        try {
            while (!stopping_.load() || !live_queue_.empty() || !replay_queue_.empty()) {
                std::optional<Packet> packet = live_queue_.try_pop();
                if (!packet.has_value()) {
                    packet = replay_queue_.try_pop();
                }
                if (!packet.has_value()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    continue;
                }
                socket_.send_packet(*packet);
            }
        } catch (const std::exception&) {
            request_stop();
        }
    }

    void reader_loop() {
        try {
            Packet packet;
            while (!stopping_.load() && socket_.receive_packet(packet)) {
                if (packet.type == MessageType::Subscribe) {
                    continue;
                }
                if (packet.type == MessageType::ReplayRequest) {
                    const ReplayRequestMessage request = decode_replay_request(packet);
                    {
                        std::lock_guard<std::mutex> lock(request_mutex_);
                        latest_request_ = request;
                    }
                    latest_generation_.store(request.generation);
                    request_condition_.notify_one();
                }
            }
        } catch (const std::exception&) {
            // Disconnects are normal for a viewer; the session is simply removed.
        }
        request_stop();
    }

    void replay_loop() {
        while (!stopping_.load()) {
            std::optional<ReplayRequestMessage> request;
            {
                std::unique_lock<std::mutex> lock(request_mutex_);
                request_condition_.wait(lock, [this] {
                    return stopping_.load() || latest_request_.has_value();
                });
                if (stopping_.load()) {
                    return;
                }
                request = latest_request_;
                latest_request_.reset();
            }
            try {
                const ReplayBundle bundle = replay_provider_(request->target_frame);
                if (!enqueue_replay(make_replay_begin(ReplayBeginMessage{
                        request->generation, bundle.target_frame, bundle.target_version, bundle.snapshot}))) {
                    enqueue_live(make_error("replay queue overflow or client stopped"));
                    continue;
                }
                bool replay_ok = true;
                for (const PointCloudDelta& delta : bundle.deltas) {
                    if (stopping_.load() || latest_generation_.load() != request->generation) {
                        replay_ok = false;
                        break;
                    }
                    if (!enqueue_replay(make_delta(MessageType::ReplayDelta, delta, request->generation))) {
                        replay_ok = false;
                        break;
                    }
                }
                if (replay_ok && !stopping_.load() && latest_generation_.load() == request->generation) {
                    if (!enqueue_replay(make_replay_end(ReplayEndMessage{
                            request->generation, bundle.target_frame, bundle.target_version}))) {
                        enqueue_live(make_error("replay queue overflow; seek again"));
                    }
                } else if (!stopping_.load() && latest_generation_.load() == request->generation) {
                    enqueue_live(make_error("replay queue overflow; seek again"));
                }
            } catch (const std::exception& error) {
                enqueue_live(make_error(error.what()));
            }
        }
    }

    TcpSocket socket_;
    HelloProvider hello_provider_;
    SnapshotProvider snapshot_provider_;
    ReplayProvider replay_provider_;
    ResyncVersionProvider resync_version_provider_;
    BoundedQueue<Packet> live_queue_;
    BoundedQueue<Packet> replay_queue_;
    std::atomic<bool> stopping_{false};
    std::atomic<std::uint64_t> latest_generation_{0};
    std::mutex request_mutex_;
    std::condition_variable request_condition_;
    std::optional<ReplayRequestMessage> latest_request_;
    std::thread writer_thread_;
    std::thread reader_thread_;
    std::thread replay_thread_;
};

void StreamRuntime::remove_dead_clients() {
    std::lock_guard<std::mutex> lock(client_mutex_);
    clients_.erase(
        std::remove_if(clients_.begin(), clients_.end(), [](const std::shared_ptr<ClientSession>& client) {
            return !client->alive();
        }),
        clients_.end());
}

void StreamRuntime::publish(const Packet& packet) {
    remove_dead_clients();
    std::vector<std::shared_ptr<ClientSession>> clients;
    {
        std::lock_guard<std::mutex> lock(client_mutex_);
        clients = clients_;
    }
    for (const auto& client : clients) {
        client->enqueue_live(packet);
    }
}

void accept_loop(
    TcpListener& listener,
    StreamRuntime& runtime,
    std::atomic<bool>& stop_requested) {
    while (!stop_requested.load()) {
        try {
            TcpSocket socket = listener.accept_one();
            auto client = std::make_shared<ClientSession>(
                std::move(socket),
                [&runtime] { return runtime.hello(); },
                [&runtime] { return runtime.current_snapshot(); },
                [&runtime](const FrameSeq target) { return runtime.replay(target); },
                [&runtime] { return runtime.latest_snapshot_version(); });
            runtime.add_client(client);
            client->start();
            std::cout << "viewer connected\n";
        } catch (const std::exception& error) {
            if (!stop_requested.load()) {
                std::cerr << "accept loop: " << error.what() << "\n";
            }
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const ServerArgs args = parse_args(argc, argv);
        const fs::path run_dir = make_run_dir(args);
        if (args.resume) {
            if (!fs::is_directory(run_dir)) {
                throw std::runtime_error("cannot resume missing run directory: " + run_dir.string());
            }
            if (args.input_group_size == 3U
                && !fs::is_regular_file(run_dir / "input_groups.csv")) {
                throw std::runtime_error(
                    "cannot resume three-image run without input_groups.csv: " + run_dir.string());
            }
        } else if (fs::exists(run_dir)) {
            throw std::runtime_error("run directory already exists; pass --resume to recover it: " + run_dir.string());
        }

        fs::create_directories(run_dir);
        ServerArgs configured_args = args;
        configured_args.inference.debug_dir = run_dir / "debug";
        configured_args.inference.group_stride = static_cast<int>(args.input_group_stride);
        StreamRuntime runtime(configured_args, run_dir);
        if (args.resume && args.input_group_size == 3U
            && !runtime.has_group_manifest_entries()) {
            throw std::runtime_error(
                "cannot resume a three-image run from an empty group manifest: " + run_dir.string());
        }
        SocketRuntime socket_runtime;
        TcpListener listener;
        listener.listen_on(args.port);
        std::cout << "omnivggt_stream_server listening on port " << args.port << "\n"
                  << "run_dir=" << run_dir << "\n";

        std::atomic<bool> stop_requested{false};
        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);

        BoundedQueue<RawFrame> frame_queue(args.queue_capacity);
        BoundedQueue<PreparedInput> prepared_queue(args.queue_capacity);
        BoundedQueue<CandidateCommit> commit_queue(args.queue_capacity + 2U);
        InFlightGate inflight_gate;

        DirectorySourceOptions source_options;
        source_options.directory = args.image_dir;
        source_options.queue_capacity = args.queue_capacity;
        source_options.poll_ms = args.poll_ms;
        source_options.max_frames = args.num_images;
        source_options.group_size = args.input_group_size;
        source_options.group_stride = args.input_group_stride;
        source_options.group_anchor_index = args.group_anchor_index;
        source_options.once = args.once;
        source_options.start_frame_seq = args.resume ? runtime.next_frame_sequence() : 0U;
        if (args.resume && args.input_group_size == 1U) {
            source_options.skip_image_names = runtime.completed_image_names();
        } else if (args.resume && args.input_group_size == 3U) {
            source_options.skip_group_keys = runtime.completed_group_keys();
        }
        DirectoryFrameSource source(source_options);

        std::thread accept_thread([&] { accept_loop(listener, runtime, stop_requested); });
        std::thread source_thread([&] {
            try {
                source.run(
                    frame_queue,
                    stop_requested,
                    [&runtime](const RawFrame& raw) { runtime.note_received(raw); },
                    [&runtime](const RawFrame& raw) { runtime.record_coalesced(raw); });
            } catch (const std::exception& error) {
                std::cerr << "frame source: " << error.what() << "\n";
                stop_requested.store(true);
                frame_queue.close();
            }
        });

        std::thread prepare_thread([&] {
            try {
                FramePreprocessor preprocessor(configured_args.inference);
                while (true) {
                    const std::optional<RawFrame> raw = frame_queue.pop();
                    if (!raw.has_value()) {
                        break;
                    }
                    inflight_gate.acquire();
                    try {
                        PreparedInput prepared = preprocessor.prepare(*raw);
                        if (!prepared_queue.push_wait(std::move(prepared))) {
                            inflight_gate.release();
                            break;
                        }
                    } catch (const std::exception& error) {
                        runtime.record_failed(*raw, error.what());
                        inflight_gate.release();
                    }
                }
            } catch (const std::exception& error) {
                std::cerr << "preprocess worker: " << error.what() << "\n";
                stop_requested.store(true);
                frame_queue.close();
            }
            prepared_queue.close();
        });

        std::thread inference_thread([&] {
            try {
                // LibTorch and the CUDA context are owned by this worker only.
                InferenceEngine engine(configured_args.inference);
                // Drain every frame already accepted before the source closes.
                // In --once/--num_images mode the queue may be closed while it
                // still contains the final frame.
                while (true) {
                    const std::optional<PreparedInput> prepared = prepared_queue.pop();
                    if (!prepared.has_value()) {
                        break;
                    }
                    try {
                        const CanvasState state = runtime.snapshot();
                        CandidateCommit candidate = engine.process_prepared(*prepared, state);
                        const FrameSeq frame_seq = candidate.frame.frame_seq;
                        if (!commit_queue.push_wait(std::move(candidate))) {
                            inflight_gate.release();
                            break;
                        }
                        runtime.wait_until_frame_handled(frame_seq);
                        inflight_gate.release();
                    } catch (const std::exception& error) {
                        runtime.record_failed(prepared->raw, error.what());
                        inflight_gate.release();
                    }
                }
            } catch (const std::exception& error) {
                std::cerr << "inference worker: " << error.what() << "\n";
                stop_requested.store(true);
                frame_queue.close();
                prepared_queue.close();
            }
            commit_queue.close();
        });

        std::thread commit_thread([&] {
            while (true) {
                const std::optional<CandidateCommit> candidate = commit_queue.pop();
                if (!candidate.has_value()) {
                    break;
                }
                runtime.commit(*candidate);
            }
            if (args.once) {
                stop_requested.store(true);
            }
        });

        if (args.once) {
            source_thread.join();
            prepare_thread.join();
            inference_thread.join();
            commit_thread.join();
        } else {
            while (!g_stop_requested.load() && !stop_requested.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            stop_requested.store(true);
            frame_queue.close();
            prepared_queue.close();
            commit_queue.close();
            source_thread.join();
            prepare_thread.join();
            inference_thread.join();
            commit_thread.join();
        }

        stop_requested.store(true);
        listener.close();
        for (int i = 0; i < 20 && accept_thread.joinable(); ++i) {
            // Closing the listening socket wakes accept on both supported platforms.
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            if (i == 19) {
                break;
            }
        }
        if (accept_thread.joinable()) {
            accept_thread.join();
        }
        std::cout << "server stopped; history is in " << run_dir << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << "\n";
        return 1;
    }
}
