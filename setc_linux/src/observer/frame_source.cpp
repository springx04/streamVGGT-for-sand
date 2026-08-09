#include "frame_source.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <sstream>
#include <thread>
#include <vector>

namespace omnivggt::observer {

namespace {

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

}  // namespace

DirectoryFrameSource::DirectoryFrameSource(DirectorySourceOptions options) : options_(std::move(options)) {
    if (options_.directory.empty() || !std::filesystem::is_directory(options_.directory)) {
        throw std::invalid_argument("frame source directory does not exist: " + options_.directory.string());
    }
    options_.poll_ms = std::max(options_.poll_ms, 10);
    options_.queue_capacity = std::max<std::size_t>(options_.queue_capacity, 1U);
    options_.group_size = std::max<std::size_t>(options_.group_size, 1U);
    options_.group_stride = std::max<std::size_t>(options_.group_stride, 1U);
    if (options_.group_anchor_index >= options_.group_size) {
        throw std::invalid_argument("group anchor index must be smaller than group size");
    }
}

bool DirectoryFrameSource::is_image(const std::filesystem::path& path) {
    const std::string extension = lower(path.extension().string());
    return extension == ".png" || extension == ".jpg" || extension == ".jpeg" || extension == ".bmp";
}

std::vector<std::filesystem::path> DirectoryFrameSource::scan() const {
    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::directory_iterator(options_.directory)) {
        if (entry.is_regular_file() && is_image(entry.path())) {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

void DirectoryFrameSource::run(
    BoundedQueue<RawFrame>& output,
    std::atomic<bool>& stop_requested,
    const std::function<void(const RawFrame&)>& on_received,
    const std::function<void(const RawFrame&)>& on_coalesced) {
    FrameSeq next_sequence = options_.start_frame_seq;
    std::uint64_t emitted_count = 0;
    std::size_t next_group_start = 0U;
    std::unordered_set<std::string> emitted_group_keys = options_.skip_group_keys;
    int empty_scans = 0;
    while (!stop_requested.load()) {
        const std::vector<std::filesystem::path> paths = scan();
        bool emitted = false;
        std::vector<std::filesystem::path> stable_paths;
        stable_paths.reserve(paths.size());
        for (const auto& path : paths) {
            const std::string key = std::filesystem::absolute(path).lexically_normal().string();
            if (options_.group_size == 1U && processed_.find(key) != processed_.end()) {
                continue;
            }
            if (options_.skip_image_names.find(path.filename().string()) != options_.skip_image_names.end()) {
                if (options_.group_size == 1U) {
                    processed_.insert(key);
                }
                continue;
            }

            std::error_code error;
            const std::uintmax_t size = std::filesystem::file_size(path, error);
            const auto modified = std::filesystem::last_write_time(path, error);
            if (error) {
                continue;
            }
            Observation& observation = observations_[key];
            if (observation.size == size && observation.modified == modified) {
                ++observation.stable_scans;
            } else {
                observation.size = size;
                observation.modified = modified;
                observation.stable_scans = 1;
            }
            if (observation.stable_scans >= 2) {
                stable_paths.push_back(path);
            }
        }

        if (options_.group_size == 1U) {
            for (const auto& path : stable_paths) {
                if (options_.max_frames != 0U && emitted_count >= options_.max_frames) {
                    stop_requested.store(true);
                    break;
                }
                RawFrame frame;
                frame.frame_seq = next_sequence++;
                frame.path = path;
                ++emitted_count;
                processed_.insert(std::filesystem::absolute(path).lexically_normal().string());
                emitted = true;
                if (on_received) {
                    on_received(frame);
                }
                const std::optional<RawFrame> dropped = output.push_latest(frame);
                if (dropped.has_value() && on_coalesced) {
                    on_coalesced(*dropped);
                }
            }
        } else {
            const std::size_t source_limit = options_.max_frames == 0U
                ? stable_paths.size()
                : std::min<std::size_t>(stable_paths.size(), static_cast<std::size_t>(options_.max_frames));
            while (next_group_start + options_.group_size <= source_limit) {
                const std::size_t end = next_group_start + options_.group_size;
                std::ostringstream key_stream;
                for (std::size_t index = next_group_start; index < end; ++index) {
                    if (index != next_group_start) {
                        key_stream << '|';
                    }
                    key_stream << std::filesystem::absolute(stable_paths[index]).lexically_normal().string();
                }
                const std::string group_key = key_stream.str();
                next_group_start += options_.group_stride;
                if (!emitted_group_keys.insert(group_key).second) {
                    continue;
                }
                RawFrame frame;
                frame.frame_seq = next_sequence++;
                frame.group_anchor_index = static_cast<int>(options_.group_anchor_index);
                frame.group_key = group_key;
                frame.group_paths.assign(
                    stable_paths.begin() + static_cast<std::ptrdiff_t>(next_group_start - options_.group_stride),
                    stable_paths.begin() + static_cast<std::ptrdiff_t>(end));
                frame.group_source_seqs.resize(frame.group_paths.size());
                for (std::size_t member = 0; member < frame.group_source_seqs.size(); ++member) {
                    frame.group_source_seqs[member] = static_cast<std::uint64_t>(
                        next_group_start - options_.group_stride + member);
                }
                frame.path = frame.group_paths[options_.group_anchor_index];
                ++emitted_count;
                emitted = true;
                if (on_received) {
                    on_received(frame);
                }
                const std::optional<RawFrame> dropped = output.push_latest(frame);
                if (dropped.has_value() && on_coalesced) {
                    on_coalesced(*dropped);
                }
            }
            if (options_.once && options_.max_frames != 0U
                && stable_paths.size() >= options_.group_size
                && next_group_start + options_.group_size > source_limit) {
                stop_requested.store(true);
            }
        }

        if (emitted) {
            empty_scans = 0;
        } else {
            ++empty_scans;
        }

        if (options_.once && options_.group_size == 1U
            && (options_.max_frames != 0U && emitted_count >= options_.max_frames)) {
            break;
        }
        if (options_.once && options_.group_size > 1U && stop_requested.load()) {
            break;
        }
        if (options_.once && options_.max_frames == 0U && empty_scans >= 3) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(options_.poll_ms));
    }
    output.close();
}

}  // namespace omnivggt::observer
