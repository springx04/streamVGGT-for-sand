#include "hikvision_capture.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using omnivggt::hikvision::CaptureOutputOptions;
using omnivggt::hikvision::HikvisionCamera;
using omnivggt::hikvision::HikvisionCameraOptions;

struct Arguments {
    HikvisionCameraOptions camera;
    CaptureOutputOptions output;
    std::uint64_t frames = 0U;
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
        << "  omnivggt_hikvision_capture --output-dir DIR [options]\n\n"
        << "Acquisition options:\n"
        << "  --device-index N       MVS enumeration index, default 0.\n"
        << "  --timeout-ms N         GetImageBuffer timeout, default 1000.\n"
        << "  --frames N             Save N frames; 0 means continuous.\n"
        << "  --camera-name NAME     Override camera id stored in JSON/CSV.\n"
        << "  --trigger-software     Use software trigger before each frame.\n\n"
        << "Intrinsic source (choose one):\n"
        << "  --calibration-file FILE  OpenCV YAML/XML with camera_matrix/K and optional D.\n"
        << "  --fx-node NAME --fy-node NAME --cx-node NAME --cy-node NAME\n"
        << "                         Read model-specific MVS float nodes.\n"
        << "  --k1-node NAME --k2-node NAME --p1-node NAME --p2-node NAME --k3-node NAME\n"
        << "                         Optional complete 5-term distortion nodes.\n\n"
        << "Output:\n"
        << "  --output-dir DIR      images/*.png, cameras/*.json, and frames.csv.\n"
        << "  --png-compression N   PNG compression 0..9, default 3.\n";
}

Arguments parse_args(const int argc, char** argv) {
    Arguments args;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (key == "--device-index") {
            args.camera.device_index = static_cast<std::size_t>(
                std::stoull(require_value(index, argc, argv, key)));
        } else if (key == "--timeout-ms") {
            args.camera.timeout_ms = std::stoi(require_value(index, argc, argv, key));
        } else if (key == "--frames") {
            args.frames = std::stoull(require_value(index, argc, argv, key));
        } else if (key == "--camera-name") {
            args.camera.camera_name = require_value(index, argc, argv, key);
        } else if (key == "--calibration-file") {
            args.camera.calibration_file = require_value(index, argc, argv, key);
        } else if (key == "--trigger-software") {
            args.camera.software_trigger = true;
        } else if (key == "--fx-node") {
            args.camera.intrinsic_nodes.fx = require_value(index, argc, argv, key);
        } else if (key == "--fy-node") {
            args.camera.intrinsic_nodes.fy = require_value(index, argc, argv, key);
        } else if (key == "--cx-node") {
            args.camera.intrinsic_nodes.cx = require_value(index, argc, argv, key);
        } else if (key == "--cy-node") {
            args.camera.intrinsic_nodes.cy = require_value(index, argc, argv, key);
        } else if (key == "--k1-node") {
            args.camera.intrinsic_nodes.k1 = require_value(index, argc, argv, key);
        } else if (key == "--k2-node") {
            args.camera.intrinsic_nodes.k2 = require_value(index, argc, argv, key);
        } else if (key == "--p1-node") {
            args.camera.intrinsic_nodes.p1 = require_value(index, argc, argv, key);
        } else if (key == "--p2-node") {
            args.camera.intrinsic_nodes.p2 = require_value(index, argc, argv, key);
        } else if (key == "--k3-node") {
            args.camera.intrinsic_nodes.k3 = require_value(index, argc, argv, key);
        } else if (key == "--output-dir") {
            args.output.output_dir = require_value(index, argc, argv, key);
        } else if (key == "--png-compression") {
            args.output.png_compression = std::stoi(require_value(index, argc, argv, key));
        } else if (key == "--help" || key == "-h") {
            usage();
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }
    if (args.output.output_dir.empty()) {
        usage();
        throw std::runtime_error("--output-dir is required");
    }
    if (args.camera.timeout_ms <= 0) {
        throw std::runtime_error("--timeout-ms must be positive");
    }
    if (args.output.png_compression < 0 || args.output.png_compression > 9) {
        throw std::runtime_error("--png-compression must be in [0, 9]");
    }
    if (args.camera.calibration_file.empty()
        && !args.camera.intrinsic_nodes.has_matrix_nodes()) {
        throw std::runtime_error(
            "provide --calibration-file or all four intrinsic node names; "
            "MVS does not provide a universal calibrated K matrix for ordinary 2-D cameras");
    }
    if (!args.camera.calibration_file.empty()
        && args.camera.intrinsic_nodes.has_matrix_nodes()) {
        std::cerr << "warning: --calibration-file takes precedence over intrinsic node names\n";
    }
    if (args.camera.intrinsic_nodes.has_any_distortion_node()
        && !args.camera.intrinsic_nodes.has_complete_distortion_nodes()) {
        throw std::runtime_error("provide all five distortion node names or none");
    }
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Arguments args = parse_args(argc, argv);
        HikvisionCamera camera(args.camera);
        camera.open();
        camera.start();
        std::cout << "opened MVS camera " << camera.camera_id()
                  << "; saving to " << args.output.output_dir << "\n";
        omnivggt::hikvision::capture_to_directory(
            camera,
            args.output,
            args.frames,
            [](const omnivggt::hikvision::CapturedFrame& frame) {
                std::cout << "saved frame=" << frame.capture_index
                          << " device_frame=" << frame.device_frame_number
                          << " size=" << frame.bgr.cols << 'x' << frame.bgr.rows << '\n';
                return true;
            });
        camera.stop();
        camera.close();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
