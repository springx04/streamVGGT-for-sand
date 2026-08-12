#include "pointcloud_export.hpp"
#include "version_store.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace omnivggt::observer;
namespace fs = std::filesystem;

namespace {

struct Args {
    fs::path run_dir;
    FrameSeq frame = std::numeric_limits<FrameSeq>::max();
    fs::path output_ply;
};

Args parse_args(const int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--run_dir" && ++i < argc) {
            args.run_dir = argv[i];
        } else if (key == "--frame" && ++i < argc) {
            args.frame = std::stoull(argv[i]);
        } else if (key == "--output_ply" && ++i < argc) {
            args.output_ply = argv[i];
        } else if (key == "--help" || key == "-h") {
            std::cout << "Usage: omnivggt_replay_log --run_dir RUN [--frame N] [--output_ply FILE]"
                         "\n";
            std::exit(0);
        } else {
            throw std::runtime_error("invalid or incomplete argument: " + key);
        }
    }
    if (args.run_dir.empty()) {
        throw std::runtime_error("--run_dir is required");
    }
    return args;
}

void write_ply(const fs::path& path, const CanvasState& state) {
    const std::vector<ExportPoint> points = export_clean_canvas_points(state, state.last_frame);
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to write PLY: " + path.string());
    }
    output << "ply\nformat ascii 1.0\nelement vertex " << points.size() << "\n"
           << "property float x\nproperty float y\nproperty float z\n"
           << "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
    output << std::fixed << std::setprecision(7);
    for (const ExportPoint& point : points) {
        output << point.x << ' ' << point.y << ' ' << point.z << ' '
               << static_cast<int>(point.r) << ' '
               << static_cast<int>(point.g) << ' '
               << static_cast<int>(point.b) << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const VersionStore store = VersionStore::open_existing(args.run_dir);
        const FrameSeq target = args.frame == std::numeric_limits<FrameSeq>::max()
            ? store.latest_frame()
            : args.frame;
        const ReplayBundle bundle = store.build_replay(target);
        CanvasState state = bundle.snapshot.state;
        for (const PointCloudDelta& delta : bundle.deltas) {
            apply_delta_forward(state, delta);
        }
        std::cout << "replayed frame=" << bundle.target_frame << " version=" << state.version
                  << " points=" << std::count(state.valid.begin(), state.valid.end(), static_cast<std::uint8_t>(1U))
                  << " hash=" << hash_state(state) << "\n";
        if (!args.output_ply.empty()) {
            write_ply(args.output_ply, state);
            std::cout << "wrote " << args.output_ply << "\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << "\n";
        return 1;
    }
}
