#include "frame_source.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
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
    int empty_scans = 0;
    while (!stop_requested.load()) {
        const std::vector<std::filesystem::path> paths = scan();
        bool emitted = false;
        for (const auto& path : paths) {
            const std::string key = std::filesystem::absolute(path).lexically_normal().string();
            if (processed_.find(key) != processed_.end()) {
                continue;
            }
            if (options_.skip_image_names.find(path.filename().string()) != options_.skip_image_names.end()) {
                processed_.insert(key);
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
            if (observation.stable_scans < 2) {
                continue;
            }

            if (options_.max_frames != 0U && emitted_count >= options_.max_frames) {
                stop_requested.store(true);
                break;
            }
            RawFrame frame{next_sequence++, path};
            ++emitted_count;
            processed_.insert(key);
            emitted = true;
            if (on_received) {
                on_received(frame);
            }
            const std::optional<RawFrame> dropped = output.push_latest(frame);
            if (dropped.has_value() && on_coalesced) {
                on_coalesced(*dropped);
            }
        }

        if (emitted) {
            empty_scans = 0;
        } else {
            ++empty_scans;
        }

        if (options_.once && (options_.max_frames != 0U && emitted_count >= options_.max_frames)) {
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
