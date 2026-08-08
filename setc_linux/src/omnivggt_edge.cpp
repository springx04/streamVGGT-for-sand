#include <torch/script.h>
#include <torch/torch.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace fs = std::filesystem;

struct Args {
    std::string model;
    std::string image_dir;
    std::string camera_dir;
    std::string depth_dir;
    std::string output_dir = "output";
    std::string device = "cpu";
    std::string dtype = "float32";
    int num_images = 0;
    int height = 518;
    int width = 518;
    float max_depth = 100.0f;
    double conf_percentile = 25.0;
    bool strict_shape = false;
    bool no_ply = false;
};

struct FrameData {
    fs::path image_path;
    std::string stem;
    cv::Mat rgb;
    cv::Mat depth;
    cv::Mat mask;
    int original_width = 0;
    int original_height = 0;
    int resized_height = 0;
    int crop_y = 0;
    int pad_top = 0;
    float scale_x = 1.0f;
    float scale_y = 1.0f;
    std::array<float, 12> extrinsic{};
    std::array<float, 9> intrinsic{};
    bool has_camera = false;
    bool has_depth = false;
};

static void usage() {
    std::cerr
        << "Usage:\n"
        << "  omnivggt_edge --model model.pt --image_dir images --output_dir out "
        << "--num_images N --height H --width W [--device cpu|cuda]\n\n"
        << "Optional:\n"
        << "  --camera_dir DIR       Camera txt files matching image stems.\n"
        << "  --depth_dir DIR        Depth PNG files matching image stems.\n"
        << "  --dtype float32|float16|bfloat16\n"
        << "  --conf_percentile P    Drop the lowest P percent confidence points.\n"
        << "  --strict_shape         Error instead of padding short resized images.\n"
        << "  --no_ply               Skip pointcloud.ply writing.\n";
}

static std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return s;
}

static bool is_image_file(const fs::path& path) {
    const std::string ext = lower(path.extension().string());
    return ext == ".png" || ext == ".jpg" || ext == ".jpeg";
}

static std::string require_value(int& i, int argc, char** argv, const std::string& key) {
    if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + key);
    }
    ++i;
    return argv[i];
}

static Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--model") {
            args.model = require_value(i, argc, argv, key);
        } else if (key == "--image_dir") {
            args.image_dir = require_value(i, argc, argv, key);
        } else if (key == "--camera_dir") {
            args.camera_dir = require_value(i, argc, argv, key);
        } else if (key == "--depth_dir") {
            args.depth_dir = require_value(i, argc, argv, key);
        } else if (key == "--output_dir") {
            args.output_dir = require_value(i, argc, argv, key);
        } else if (key == "--device") {
            args.device = require_value(i, argc, argv, key);
        } else if (key == "--dtype") {
            args.dtype = require_value(i, argc, argv, key);
        } else if (key == "--num_images") {
            args.num_images = std::stoi(require_value(i, argc, argv, key));
        } else if (key == "--height") {
            args.height = std::stoi(require_value(i, argc, argv, key));
        } else if (key == "--width") {
            args.width = std::stoi(require_value(i, argc, argv, key));
        } else if (key == "--max_depth") {
            args.max_depth = std::stof(require_value(i, argc, argv, key));
        } else if (key == "--conf_percentile") {
            args.conf_percentile = std::stod(require_value(i, argc, argv, key));
        } else if (key == "--strict_shape") {
            args.strict_shape = true;
        } else if (key == "--no_ply") {
            args.no_ply = true;
        } else if (key == "--help" || key == "-h") {
            usage();
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + key);
        }
    }

    if (args.model.empty() || args.image_dir.empty()) {
        usage();
        throw std::runtime_error("--model and --image_dir are required");
    }
    if (args.num_images <= 0) {
        throw std::runtime_error("--num_images must be positive and must match the exported artifact");
    }
    if (args.height <= 0 || args.width <= 0 || args.height % 14 != 0 || args.width % 14 != 0) {
        throw std::runtime_error("--height and --width must be positive multiples of 14");
    }
    if (args.conf_percentile < 0.0 || args.conf_percentile > 100.0) {
        throw std::runtime_error("--conf_percentile must be in [0, 100]");
    }
    if (args.dtype != "float32" && args.dtype != "float16" && args.dtype != "bfloat16") {
        throw std::runtime_error("--dtype must be float32, float16, or bfloat16");
    }
    return args;
}

static std::vector<fs::path> list_images(const fs::path& dir, int limit) {
    if (!fs::is_directory(dir)) {
        throw std::runtime_error("Not a directory: " + dir.string());
    }

    std::vector<fs::path> paths;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file() && is_image_file(entry.path())) {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());
    if (static_cast<int>(paths.size()) < limit) {
        throw std::runtime_error("Not enough images in " + dir.string());
    }
    paths.resize(static_cast<size_t>(limit));
    return paths;
}

static int round_to_multiple_14(double value) {
    int rounded = static_cast<int>(std::round(value / 14.0)) * 14;
    return std::max(14, rounded);
}

static cv::Mat read_rgb(const fs::path& image_path) {
    cv::Mat input = cv::imread(image_path.string(), cv::IMREAD_UNCHANGED);
    if (input.empty()) {
        throw std::runtime_error("Failed to read image: " + image_path.string());
    }

    cv::Mat rgb;
    if (input.channels() == 1) {
        cv::cvtColor(input, rgb, cv::COLOR_GRAY2RGB);
    } else if (input.channels() == 3) {
        cv::cvtColor(input, rgb, cv::COLOR_BGR2RGB);
    } else if (input.channels() == 4) {
        cv::cvtColor(input, rgb, cv::COLOR_BGRA2RGB);
    } else {
        throw std::runtime_error("Unsupported image channel count: " + image_path.string());
    }
    return rgb;
}

static FrameData preprocess_frame(const fs::path& image_path, const Args& args) {
    FrameData frame;
    frame.image_path = image_path;
    frame.stem = image_path.stem().string();

    cv::Mat rgb = read_rgb(image_path);
    frame.original_width = rgb.cols;
    frame.original_height = rgb.rows;
    frame.scale_x = static_cast<float>(args.width) / static_cast<float>(rgb.cols);
    frame.resized_height = round_to_multiple_14(static_cast<double>(rgb.rows) * frame.scale_x);
    frame.scale_y = static_cast<float>(frame.resized_height) / static_cast<float>(rgb.rows);

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(args.width, frame.resized_height), 0.0, 0.0, cv::INTER_CUBIC);

    if (frame.resized_height > args.height) {
        frame.crop_y = (frame.resized_height - args.height) / 2;
        frame.rgb = resized(cv::Rect(0, frame.crop_y, args.width, args.height)).clone();
    } else if (frame.resized_height < args.height) {
        if (args.strict_shape) {
            std::ostringstream oss;
            oss << "Image " << image_path << " resized to height " << frame.resized_height
                << ", but exported model expects " << args.height;
            throw std::runtime_error(oss.str());
        }
        frame.pad_top = (args.height - frame.resized_height) / 2;
        frame.rgb = cv::Mat(args.height, args.width, CV_8UC3, cv::Scalar(0, 0, 0));
        resized.copyTo(frame.rgb(cv::Rect(0, frame.pad_top, args.width, frame.resized_height)));
    } else {
        frame.rgb = resized;
    }

    frame.depth = cv::Mat::zeros(args.height, args.width, CV_32FC1);
    frame.mask = cv::Mat::zeros(args.height, args.width, CV_32FC1);
    return frame;
}

static std::vector<double> parse_numbers(const std::string& line) {
    std::istringstream iss(line);
    std::vector<double> values;
    double value = 0.0;
    while (iss >> value) {
        values.push_back(value);
    }
    return values;
}

static std::vector<std::string> read_non_comment_lines(const fs::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open camera file: " + path.string());
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        const auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) {
            continue;
        }
        if (line[first] == '#') {
            continue;
        }
        lines.push_back(line.substr(first));
    }
    return lines;
}

static std::array<float, 12> invert_3x4(const std::array<float, 12>& c2w) {
    std::array<float, 12> w2c{};
    w2c[0] = c2w[0];
    w2c[1] = c2w[4];
    w2c[2] = c2w[8];
    w2c[4] = c2w[1];
    w2c[5] = c2w[5];
    w2c[6] = c2w[9];
    w2c[8] = c2w[2];
    w2c[9] = c2w[6];
    w2c[10] = c2w[10];

    const float tx = c2w[3];
    const float ty = c2w[7];
    const float tz = c2w[11];
    w2c[3] = -(w2c[0] * tx + w2c[1] * ty + w2c[2] * tz);
    w2c[7] = -(w2c[4] * tx + w2c[5] * ty + w2c[6] * tz);
    w2c[11] = -(w2c[8] * tx + w2c[9] * ty + w2c[10] * tz);
    return w2c;
}

static bool load_camera(FrameData& frame, const fs::path& camera_dir) {
    if (camera_dir.empty()) {
        return false;
    }
    const fs::path camera_path = camera_dir / (frame.stem + ".txt");
    if (!fs::exists(camera_path)) {
        return false;
    }

    const std::vector<std::string> lines = read_non_comment_lines(camera_path);
    if (lines.size() < 6) {
        throw std::runtime_error("Camera file must have at least 6 numeric lines: " + camera_path.string());
    }

    std::array<float, 12> c2w{};
    std::array<float, 9> k{};
    for (int row = 0; row < 3; ++row) {
        const auto values = parse_numbers(lines[static_cast<size_t>(row)]);
        if (values.size() != 4) {
            throw std::runtime_error("Invalid extrinsic row in " + camera_path.string());
        }
        for (int col = 0; col < 4; ++col) {
            c2w[static_cast<size_t>(row * 4 + col)] = static_cast<float>(values[static_cast<size_t>(col)]);
        }
    }
    for (int row = 0; row < 3; ++row) {
        const auto values = parse_numbers(lines[static_cast<size_t>(row + 3)]);
        if (values.size() != 3) {
            throw std::runtime_error("Invalid intrinsic row in " + camera_path.string());
        }
        for (int col = 0; col < 3; ++col) {
            k[static_cast<size_t>(row * 3 + col)] = static_cast<float>(values[static_cast<size_t>(col)]);
        }
    }

    k[0] *= frame.scale_x;
    k[4] *= frame.scale_y;
    k[2] *= frame.scale_x;
    k[5] = k[5] * frame.scale_y - static_cast<float>(frame.crop_y) + static_cast<float>(frame.pad_top);

    frame.extrinsic = invert_3x4(c2w);
    frame.intrinsic = k;
    frame.has_camera = true;
    return true;
}

static cv::Mat convert_depth_to_float(const cv::Mat& input) {
    cv::Mat single;
    if (input.channels() == 1) {
        single = input;
    } else {
        std::vector<cv::Mat> channels;
        cv::split(input, channels);
        single = channels.front();
    }

    cv::Mat as_float;
    single.convertTo(as_float, CV_32FC1);
    return as_float;
}

static bool load_depth_png(FrameData& frame, const fs::path& depth_dir, const Args& args) {
    if (depth_dir.empty()) {
        return false;
    }
    const fs::path depth_path = depth_dir / (frame.stem + ".png");
    if (!fs::exists(depth_path)) {
        return false;
    }

    cv::Mat raw = cv::imread(depth_path.string(), cv::IMREAD_UNCHANGED);
    if (raw.empty()) {
        throw std::runtime_error("Failed to read depth: " + depth_path.string());
    }

    cv::Mat depth = convert_depth_to_float(raw);
    cv::Mat transposed;
    cv::transpose(depth, transposed);
    depth = transposed;

    cv::Mat resized;
    cv::resize(depth, resized, cv::Size(args.width, frame.resized_height), 0.0, 0.0, cv::INTER_NEAREST);

    cv::Mat fixed;
    if (frame.resized_height > args.height) {
        fixed = resized(cv::Rect(0, frame.crop_y, args.width, args.height)).clone();
    } else if (frame.resized_height < args.height) {
        fixed = cv::Mat::zeros(args.height, args.width, CV_32FC1);
        resized.copyTo(fixed(cv::Rect(0, frame.pad_top, args.width, frame.resized_height)));
    } else {
        fixed = resized;
    }

    frame.depth = cv::Mat::zeros(args.height, args.width, CV_32FC1);
    frame.mask = cv::Mat::zeros(args.height, args.width, CV_32FC1);
    for (int y = 0; y < args.height; ++y) {
        for (int x = 0; x < args.width; ++x) {
            float value = fixed.at<float>(y, x);
            if (!std::isfinite(value) || value > args.max_depth || value < 1e-5f) {
                value = 0.0f;
            }
            frame.depth.at<float>(y, x) = value;
            frame.mask.at<float>(y, x) = value > 0.0f ? 1.0f : 0.0f;
        }
    }
    frame.has_depth = true;
    return true;
}

static std::vector<FrameData> load_frames(const Args& args) {
    const std::vector<fs::path> image_paths = list_images(args.image_dir, args.num_images);
    std::vector<FrameData> frames;
    frames.reserve(image_paths.size());
    for (const auto& path : image_paths) {
        FrameData frame = preprocess_frame(path, args);
        if (!args.camera_dir.empty()) {
            load_camera(frame, args.camera_dir);
        }
        if (!args.depth_dir.empty()) {
            load_depth_png(frame, args.depth_dir, args);
        }
        frames.push_back(std::move(frame));
    }
    return frames;
}

#ifdef _WIN32
static void load_libtorch_cuda_dlls() {
    const std::array<const char*, 2> dlls = {"c10_cuda.dll", "torch_cuda.dll"};
    for (const char* dll : dlls) {
        HMODULE handle = LoadLibraryA(dll);
        if (handle == nullptr) {
            const DWORD error = GetLastError();
            std::ostringstream oss;
            oss << "Failed to load " << dll << " before CUDA initialization. "
                << "Make sure the LibTorch CUDA lib directory is in PATH. "
                << "GetLastError=" << error;
            throw std::runtime_error(oss.str());
        }
    }
}
#endif

static torch::Device parse_device(const std::string& value) {
    if (value == "cpu") {
        return torch::Device(torch::kCPU);
    }
    if (value == "cuda") {
#ifdef _WIN32
        load_libtorch_cuda_dlls();
#endif
        if (!torch::cuda::is_available()) {
            throw std::runtime_error("CUDA requested but torch::cuda::is_available() is false");
        }
        return torch::Device(torch::kCUDA);
    }
    throw std::runtime_error("Unsupported --device: " + value);
}

static torch::ScalarType parse_dtype(const std::string& value) {
    if (value == "float32") {
        return torch::kFloat32;
    }
    if (value == "float16") {
        return torch::kFloat16;
    }
    if (value == "bfloat16") {
        return torch::kBFloat16;
    }
    throw std::runtime_error("Unsupported --dtype: " + value);
}

static torch::Tensor make_images_tensor(const std::vector<FrameData>& frames, int height, int width) {
    const int64_t s = static_cast<int64_t>(frames.size());
    std::vector<float> data(static_cast<size_t>(s) * 3U * static_cast<size_t>(height) * static_cast<size_t>(width));
    for (int64_t i = 0; i < s; ++i) {
        const cv::Mat& rgb = frames[static_cast<size_t>(i)].rgb;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const cv::Vec3b pixel = rgb.at<cv::Vec3b>(y, x);
                for (int c = 0; c < 3; ++c) {
                    const size_t offset =
                        (((static_cast<size_t>(i) * 3U + static_cast<size_t>(c)) * static_cast<size_t>(height)
                          + static_cast<size_t>(y))
                         * static_cast<size_t>(width))
                        + static_cast<size_t>(x);
                    data[offset] = static_cast<float>(pixel[c]) / 255.0f;
                }
            }
        }
    }
    return torch::from_blob(data.data(), {1, s, 3, height, width}, torch::kFloat32).clone();
}

static torch::Tensor make_extrinsics_tensor(const std::vector<FrameData>& frames) {
    const int64_t s = static_cast<int64_t>(frames.size());
    std::vector<float> data(static_cast<size_t>(s) * 12U, 0.0f);
    for (int64_t i = 0; i < s; ++i) {
        const auto& ext = frames[static_cast<size_t>(i)].extrinsic;
        std::copy(ext.begin(), ext.end(), data.begin() + i * 12);
    }
    return torch::from_blob(data.data(), {1, s, 3, 4}, torch::kFloat32).clone();
}

static torch::Tensor make_intrinsics_tensor(const std::vector<FrameData>& frames) {
    const int64_t s = static_cast<int64_t>(frames.size());
    std::vector<float> data(static_cast<size_t>(s) * 9U, 0.0f);
    for (int64_t i = 0; i < s; ++i) {
        const auto& k = frames[static_cast<size_t>(i)].intrinsic;
        std::copy(k.begin(), k.end(), data.begin() + i * 9);
    }
    return torch::from_blob(data.data(), {1, s, 3, 3}, torch::kFloat32).clone();
}

static torch::Tensor make_depth_tensor(const std::vector<FrameData>& frames, int height, int width) {
    const int64_t s = static_cast<int64_t>(frames.size());
    std::vector<float> data(static_cast<size_t>(s) * static_cast<size_t>(height) * static_cast<size_t>(width), 0.0f);
    for (int64_t i = 0; i < s; ++i) {
        const cv::Mat& depth = frames[static_cast<size_t>(i)].depth;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const size_t offset =
                    ((static_cast<size_t>(i) * static_cast<size_t>(height) + static_cast<size_t>(y))
                     * static_cast<size_t>(width))
                    + static_cast<size_t>(x);
                data[offset] = depth.at<float>(y, x);
            }
        }
    }
    return torch::from_blob(data.data(), {1, s, height, width, 1}, torch::kFloat32).clone();
}

static torch::Tensor make_mask_tensor(const std::vector<FrameData>& frames, int height, int width) {
    const int64_t s = static_cast<int64_t>(frames.size());
    std::vector<float> data(static_cast<size_t>(s) * static_cast<size_t>(height) * static_cast<size_t>(width), 0.0f);
    for (int64_t i = 0; i < s; ++i) {
        const cv::Mat& mask = frames[static_cast<size_t>(i)].mask;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const size_t offset =
                    ((static_cast<size_t>(i) * static_cast<size_t>(height) + static_cast<size_t>(y))
                     * static_cast<size_t>(width))
                    + static_cast<size_t>(x);
                data[offset] = mask.at<float>(y, x);
            }
        }
    }
    return torch::from_blob(data.data(), {1, s, height, width}, torch::kFloat32).clone();
}

static float percentile(std::vector<float> values, double percent) {
    values.erase(
        std::remove_if(values.begin(), values.end(), [](float v) { return !std::isfinite(v); }),
        values.end());
    if (values.empty()) {
        return 0.0f;
    }
    const double clamped = std::max(0.0, std::min(100.0, percent));
    const size_t index = static_cast<size_t>(std::floor((clamped / 100.0) * static_cast<double>(values.size() - 1)));
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(index), values.end());
    return values[index];
}

static void write_pointcloud_ply(
    const fs::path& path,
    const torch::Tensor& world_points_cpu,
    const torch::Tensor& conf_cpu,
    const std::vector<FrameData>& frames,
    int height,
    int width,
    double conf_percent) {
    torch::Tensor points_contig = world_points_cpu.contiguous();
    torch::Tensor conf_contig = conf_cpu.contiguous();
    auto points = points_contig.accessor<float, 5>();
    auto conf = conf_contig.accessor<float, 4>();
    const int64_t s = world_points_cpu.size(1);

    std::vector<float> conf_values;
    conf_values.reserve(static_cast<size_t>(s) * static_cast<size_t>(height) * static_cast<size_t>(width));
    for (int64_t i = 0; i < s; ++i) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                conf_values.push_back(conf[0][i][y][x]);
            }
        }
    }
    const float threshold = conf_percent <= 0.0 ? 0.0f : percentile(conf_values, conf_percent);

    struct Vertex {
        float x;
        float y;
        float z;
        unsigned char r;
        unsigned char g;
        unsigned char b;
    };
    std::vector<Vertex> vertices;
    vertices.reserve(conf_values.size());

    for (int64_t i = 0; i < s; ++i) {
        const cv::Mat& rgb = frames[static_cast<size_t>(i)].rgb;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const float c = conf[0][i][y][x];
                const float px = points[0][i][y][x][0];
                const float py = points[0][i][y][x][1];
                const float pz = points[0][i][y][x][2];
                if (c < threshold || c <= 1e-5f || !std::isfinite(px) || !std::isfinite(py) || !std::isfinite(pz)) {
                    continue;
                }
                const cv::Vec3b color = rgb.at<cv::Vec3b>(y, x);
                vertices.push_back(Vertex{px, py, pz, color[0], color[1], color[2]});
            }
        }
    }

    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to write PLY: " + path.string());
    }
    out << "ply\n";
    out << "format ascii 1.0\n";
    out << "element vertex " << vertices.size() << "\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "property uchar red\n";
    out << "property uchar green\n";
    out << "property uchar blue\n";
    out << "end_header\n";
    out << std::fixed << std::setprecision(7);
    for (const Vertex& v : vertices) {
        out << v.x << ' ' << v.y << ' ' << v.z << ' '
            << static_cast<int>(v.r) << ' ' << static_cast<int>(v.g) << ' ' << static_cast<int>(v.b) << '\n';
    }
    std::cout << "Wrote " << vertices.size() << " points to " << path << "\n";
}

static std::array<float, 9> quat_xyzw_to_mat(const float x, const float y, const float z, const float w) {
    const float denom = x * x + y * y + z * z + w * w;
    if (denom < 1e-12f) {
        return {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    }
    const float two_s = 2.0f / denom;
    return {
        1.0f - two_s * (y * y + z * z),
        two_s * (x * y - z * w),
        two_s * (x * z + y * w),
        two_s * (x * y + z * w),
        1.0f - two_s * (x * x + z * z),
        two_s * (y * z - x * w),
        two_s * (x * z - y * w),
        two_s * (y * z + x * w),
        1.0f - two_s * (x * x + y * y),
    };
}

static void write_cameras(
    const fs::path& path,
    const torch::Tensor& pose_cpu,
    const std::vector<FrameData>& frames,
    int height,
    int width) {
    torch::Tensor pose_contig = pose_cpu.contiguous();
    auto pose = pose_contig.accessor<float, 3>();
    const int64_t s = pose_cpu.size(1);
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to write cameras: " + path.string());
    }
    out << std::fixed << std::setprecision(8);
    for (int64_t i = 0; i < s; ++i) {
        const float tx = pose[0][i][0];
        const float ty = pose[0][i][1];
        const float tz = pose[0][i][2];
        const float qx = pose[0][i][3];
        const float qy = pose[0][i][4];
        const float qz = pose[0][i][5];
        const float qw = pose[0][i][6];
        const float fov_h = std::max(1e-6f, pose[0][i][7]);
        const float fov_w = std::max(1e-6f, pose[0][i][8]);
        const std::array<float, 9> r = quat_xyzw_to_mat(qx, qy, qz, qw);
        const float fy = (static_cast<float>(height) / 2.0f) / std::tan(fov_h / 2.0f);
        const float fx = (static_cast<float>(width) / 2.0f) / std::tan(fov_w / 2.0f);

        out << "# frame " << i << " " << frames[static_cast<size_t>(i)].image_path.string() << "\n";
        out << "# predicted world_to_camera extrinsic 3x4\n";
        out << r[0] << ' ' << r[1] << ' ' << r[2] << ' ' << tx << "\n";
        out << r[3] << ' ' << r[4] << ' ' << r[5] << ' ' << ty << "\n";
        out << r[6] << ' ' << r[7] << ' ' << r[8] << ' ' << tz << "\n";
        out << "# predicted intrinsic 3x3\n";
        out << fx << " 0 " << (static_cast<float>(width) / 2.0f) << "\n";
        out << "0 " << fy << ' ' << (static_cast<float>(height) / 2.0f) << "\n";
        out << "0 0 1\n\n";
    }
    std::cout << "Wrote cameras to " << path << "\n";
}

static std::string shape_string(const torch::Tensor& tensor) {
    std::ostringstream oss;
    oss << '[';
    for (int64_t i = 0; i < tensor.dim(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << tensor.size(i);
    }
    oss << ']';
    return oss.str();
}

static void write_summary(
    const fs::path& path,
    const Args& args,
    const std::vector<FrameData>& frames,
    const std::vector<torch::Tensor>& outputs) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to write summary: " + path.string());
    }
    out << "model=" << args.model << "\n";
    out << "image_dir=" << args.image_dir << "\n";
    out << "num_images=" << args.num_images << "\n";
    out << "height=" << args.height << "\n";
    out << "width=" << args.width << "\n";
    out << "device=" << args.device << "\n";
    out << "dtype=" << args.dtype << "\n";
    out << "conf_percentile=" << args.conf_percentile << "\n";
    out << "frames:\n";
    for (size_t i = 0; i < frames.size(); ++i) {
        const FrameData& frame = frames[i];
        out << "  " << i << ": " << frame.image_path.string()
            << ", original=" << frame.original_width << "x" << frame.original_height
            << ", resized_height=" << frame.resized_height
            << ", crop_y=" << frame.crop_y
            << ", pad_top=" << frame.pad_top
            << ", camera=" << (frame.has_camera ? "yes" : "no")
            << ", depth=" << (frame.has_depth ? "yes" : "no") << "\n";
    }
    static const std::array<const char*, 5> names = {
        "pose_enc", "depth", "depth_conf", "world_points", "world_points_conf"};
    out << "outputs:\n";
    for (size_t i = 0; i < outputs.size() && i < names.size(); ++i) {
        out << "  " << names[i] << "=" << shape_string(outputs[i]) << "\n";
    }
}

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        fs::create_directories(args.output_dir);

        std::vector<FrameData> frames = load_frames(args);
        std::cout << "Loaded " << frames.size() << " frames\n";

        torch::NoGradGuard no_grad;
        const torch::Device device = parse_device(args.device);
        const torch::ScalarType dtype = parse_dtype(args.dtype);
        torch::jit::script::Module module = torch::jit::load(args.model, device);
        module.eval();

        torch::Tensor images = make_images_tensor(frames, args.height, args.width).to(device, dtype);
        torch::Tensor extrinsics = make_extrinsics_tensor(frames).to(device, dtype);
        torch::Tensor intrinsics = make_intrinsics_tensor(frames).to(device, dtype);
        torch::Tensor depth = make_depth_tensor(frames, args.height, args.width).to(device, dtype);
        torch::Tensor mask = make_mask_tensor(frames, args.height, args.width).to(device, dtype);

        std::vector<torch::jit::IValue> inputs;
        inputs.reserve(5);
        inputs.emplace_back(images);
        inputs.emplace_back(extrinsics);
        inputs.emplace_back(intrinsics);
        inputs.emplace_back(depth);
        inputs.emplace_back(mask);

        std::cout << "Running OmniVGGT TorchScript inference\n";
        const auto output_ivalue = module.forward(inputs);
        const auto output_tuple = output_ivalue.toTuple();
        if (output_tuple->elements().size() != 5) {
            throw std::runtime_error("Expected 5 output tensors from TorchScript module");
        }

        std::vector<torch::Tensor> outputs;
        outputs.reserve(5);
        for (const auto& item : output_tuple->elements()) {
            outputs.push_back(item.toTensor().to(torch::kCPU).contiguous());
        }

        const fs::path output_dir(args.output_dir);
        write_cameras(output_dir / "cameras.txt", outputs[0], frames, args.height, args.width);
        if (!args.no_ply) {
            write_pointcloud_ply(
                output_dir / "pointcloud.ply",
                outputs[3],
                outputs[4],
                frames,
                args.height,
                args.width,
                args.conf_percentile);
        }
        write_summary(output_dir / "summary.txt", args, frames, outputs);
        std::cout << "Done. Outputs are in " << output_dir << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << "\n";
        return 1;
    }
}
