#include <torch/script.h>
#include <torch/torch.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
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
    std::string output_dir = "setc/stream_output";
    std::string device = "cuda";
    std::string dtype = "float32";
    int num_images = 0;
    int height = 518;
    int width = 518;
    double image_l1_thr = 12.0 / 255.0;
    double no_change_ratio = 0.001;
    double scene_jump_ratio = 0.35;
    double min_conf = 0.1;
    double conf_percentile = 25.0;
    int dilate_ksize = 3;
    bool save_debug = false;
};

struct Frame {
    int frame_id = 0;
    fs::path path;
    cv::Mat rgb_u8;      // RGB, HxWx3, uint8
    cv::Mat rgb_f;       // RGB, HxWx3, float32 [0,1]
    cv::Mat support;     // HxW, uint8 mask
    int original_width = 0;
    int original_height = 0;
    int resized_height = 0;
    int crop_y = 0;
    int pad_top = 0;
};

struct StreamMetrics {
    int frame_id = 0;
    std::string image;
    double read_ms = 0.0;
    double align2d_ms = 0.0;
    double diff_ms = 0.0;
    double model_ms = 0.0;
    double depth_align_ms = 0.0;
    double fuse_ms = 0.0;
    double total_ms = 0.0;
    double changed_ratio = 0.0;
    double photometric_changed_ratio = 0.0;
    double support_changed_ratio = 0.0;
    int fused_pixels = 0;
    int point_count = 0;
    int homography_inliers = 0;
    double homography_error_px = -1.0;
    bool skipped_model = false;
    std::string fallback;
};

struct Prediction {
    cv::Mat depth;
    cv::Mat conf;
};

class Timer {
public:
    Timer() : start_(std::chrono::steady_clock::now()) {}

    double ms() const {
        const auto end = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

private:
    std::chrono::steady_clock::time_point start_;
};

struct CanvasState {
    cv::Mat anchor_rgb_u8;
    cv::Mat rgb_f;
    cv::Mat depth;
    cv::Mat conf;
    cv::Mat weight;
    cv::Mat valid;
    cv::Mat support;
    bool initialized = false;
};

static void usage() {
    std::cerr
        << "Usage:\n"
        << "  omnivggt_stream --model model.pt --image_dir images --output_dir out "
        << "--num_images N --height H --width W --device cuda\n\n"
        << "Options:\n"
        << "  --dtype float32|float16|bfloat16\n"
        << "  --image_l1_thr V       Photometric change threshold, default 12/255.\n"
        << "  --no_change_ratio V    Skip model below this changed ratio after initialization.\n"
        << "  --scene_jump_ratio V   Mark large changes in metrics.\n"
        << "  --min_conf V           Minimum depth confidence for fusion.\n"
        << "  --conf_percentile P    PLY export confidence percentile.\n"
        << "  --dilate_ksize K       Change-mask dilation kernel size.\n"
        << "  --save_debug           Save warped images and change masks.\n";
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
        } else if (key == "--image_l1_thr") {
            args.image_l1_thr = std::stod(require_value(i, argc, argv, key));
        } else if (key == "--no_change_ratio") {
            args.no_change_ratio = std::stod(require_value(i, argc, argv, key));
        } else if (key == "--scene_jump_ratio") {
            args.scene_jump_ratio = std::stod(require_value(i, argc, argv, key));
        } else if (key == "--min_conf") {
            args.min_conf = std::stod(require_value(i, argc, argv, key));
        } else if (key == "--conf_percentile") {
            args.conf_percentile = std::stod(require_value(i, argc, argv, key));
        } else if (key == "--dilate_ksize") {
            args.dilate_ksize = std::stoi(require_value(i, argc, argv, key));
        } else if (key == "--save_debug") {
            args.save_debug = true;
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
        throw std::runtime_error("--num_images must be positive");
    }
    if (args.height <= 0 || args.width <= 0 || args.height % 14 != 0 || args.width % 14 != 0) {
        throw std::runtime_error("--height and --width must be positive multiples of 14");
    }
    if (args.dtype != "float32" && args.dtype != "float16" && args.dtype != "bfloat16") {
        throw std::runtime_error("--dtype must be float32, float16, or bfloat16");
    }
    if (args.device != "cuda") {
        throw std::runtime_error("Streaming C++ deployment is GPU-only here; use --device cuda");
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

static Frame load_frame(const fs::path& path, int frame_id, const Args& args) {
    Frame frame;
    frame.frame_id = frame_id;
    frame.path = path;
    cv::Mat rgb = read_rgb(path);
    frame.original_width = rgb.cols;
    frame.original_height = rgb.rows;

    const double scale_x = static_cast<double>(args.width) / static_cast<double>(rgb.cols);
    frame.resized_height = round_to_multiple_14(static_cast<double>(rgb.rows) * scale_x);

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(args.width, frame.resized_height), 0.0, 0.0, cv::INTER_CUBIC);
    if (frame.resized_height > args.height) {
        frame.crop_y = (frame.resized_height - args.height) / 2;
        frame.rgb_u8 = resized(cv::Rect(0, frame.crop_y, args.width, args.height)).clone();
    } else if (frame.resized_height < args.height) {
        frame.pad_top = (args.height - frame.resized_height) / 2;
        frame.rgb_u8 = cv::Mat(args.height, args.width, CV_8UC3, cv::Scalar(0, 0, 0));
        resized.copyTo(frame.rgb_u8(cv::Rect(0, frame.pad_top, args.width, frame.resized_height)));
    } else {
        frame.rgb_u8 = resized;
    }

    frame.rgb_u8.convertTo(frame.rgb_f, CV_32FC3, 1.0 / 255.0);
    std::vector<cv::Mat> channels;
    cv::split(frame.rgb_f, channels);
    cv::Mat sum = channels[0] + channels[1] + channels[2];
    cv::threshold(sum, frame.support, 0.03, 255.0, cv::THRESH_BINARY);
    frame.support.convertTo(frame.support, CV_8UC1);
    return frame;
}

#ifdef _WIN32
static void load_libtorch_cuda_dlls() {
    const std::array<const char*, 2> dlls = {"c10_cuda.dll", "torch_cuda.dll"};
    for (const char* dll : dlls) {
        HMODULE handle = LoadLibraryA(dll);
        if (handle == nullptr) {
            const DWORD error = GetLastError();
            std::ostringstream oss;
            oss << "Failed to load " << dll << " before CUDA initialization. GetLastError=" << error;
            throw std::runtime_error(oss.str());
        }
    }
}
#endif

static torch::Device parse_device(const std::string& value) {
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

static torch::Tensor make_images_tensor(const Frame& frame, int height, int width) {
    std::vector<float> data(3U * static_cast<size_t>(height) * static_cast<size_t>(width));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const cv::Vec3b pixel = frame.rgb_u8.at<cv::Vec3b>(y, x);
            for (int c = 0; c < 3; ++c) {
                const size_t offset =
                    ((static_cast<size_t>(c) * static_cast<size_t>(height) + static_cast<size_t>(y))
                     * static_cast<size_t>(width))
                    + static_cast<size_t>(x);
                data[offset] = static_cast<float>(pixel[c]) / 255.0f;
            }
        }
    }
    return torch::from_blob(data.data(), {1, 1, 3, height, width}, torch::kFloat32).clone();
}

static Prediction run_model(
    torch::jit::script::Module& module,
    const Frame& frame,
    int height,
    int width,
    const torch::Device& device,
    torch::ScalarType dtype) {
    torch::Tensor images = make_images_tensor(frame, height, width).to(device, dtype);
    torch::Tensor extrinsics = torch::zeros({1, 1, 3, 4}, torch::TensorOptions().dtype(torch::kFloat32)).to(device, dtype);
    torch::Tensor intrinsics = torch::zeros({1, 1, 3, 3}, torch::TensorOptions().dtype(torch::kFloat32)).to(device, dtype);
    torch::Tensor depth_in = torch::zeros({1, 1, height, width, 1}, torch::TensorOptions().dtype(torch::kFloat32)).to(device, dtype);
    torch::Tensor mask = torch::zeros({1, 1, height, width}, torch::TensorOptions().dtype(torch::kFloat32)).to(device, dtype);

    std::vector<torch::jit::IValue> inputs;
    inputs.reserve(5);
    inputs.emplace_back(images);
    inputs.emplace_back(extrinsics);
    inputs.emplace_back(intrinsics);
    inputs.emplace_back(depth_in);
    inputs.emplace_back(mask);

    const auto output_ivalue = module.forward(inputs);
    const auto output_tuple = output_ivalue.toTuple();
    if (output_tuple->elements().size() != 5) {
        throw std::runtime_error("Expected 5 output tensors from TorchScript module");
    }

    torch::Tensor depth = output_tuple->elements()[1].toTensor().to(torch::kCPU).contiguous();
    torch::Tensor conf = output_tuple->elements()[2].toTensor().to(torch::kCPU).contiguous();
    auto depth_acc = depth.accessor<float, 5>();
    auto conf_acc = conf.accessor<float, 4>();

    cv::Mat depth_mat(height, width, CV_32FC1);
    cv::Mat conf_mat(height, width, CV_32FC1);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            depth_mat.at<float>(y, x) = depth_acc[0][0][y][x][0];
            conf_mat.at<float>(y, x) = conf_acc[0][0][y][x];
        }
    }
    return Prediction{depth_mat, conf_mat};
}

static cv::Mat gray_u8(const cv::Mat& rgb_u8) {
    cv::Mat gray;
    cv::cvtColor(rgb_u8, gray, cv::COLOR_RGB2GRAY);
    return gray;
}

static cv::Mat translation_h(double dx, double dy) {
    cv::Mat h = cv::Mat::eye(3, 3, CV_32FC1);
    h.at<float>(0, 2) = static_cast<float>(dx);
    h.at<float>(1, 2) = static_cast<float>(dy);
    return h;
}

static std::pair<cv::Mat, StreamMetrics> estimate_homography(const Frame& curr, const cv::Mat& anchor_rgb_u8, StreamMetrics metrics) {
    if (curr.frame_id == 0) {
        return {cv::Mat::eye(3, 3, CV_32FC1), metrics};
    }

    const cv::Mat curr_gray = gray_u8(curr.rgb_u8);
    const cv::Mat anchor_gray = gray_u8(anchor_rgb_u8);

    cv::Ptr<cv::Feature2D> detector;
    int norm = cv::NORM_HAMMING;
    try {
        detector = cv::SIFT::create(1500);
        norm = cv::NORM_L2;
    } catch (const cv::Exception&) {
        detector = cv::ORB::create(1500);
        norm = cv::NORM_HAMMING;
    }

    std::vector<cv::KeyPoint> kp_curr;
    std::vector<cv::KeyPoint> kp_anchor;
    cv::Mat des_curr;
    cv::Mat des_anchor;
    detector->detectAndCompute(curr_gray, cv::noArray(), kp_curr, des_curr);
    detector->detectAndCompute(anchor_gray, cv::noArray(), kp_anchor, des_anchor);

    auto phase_fallback = [&](const std::string& reason) {
        cv::Mat curr_f;
        cv::Mat anchor_f;
        curr_gray.convertTo(curr_f, CV_32FC1, 1.0 / 255.0);
        anchor_gray.convertTo(anchor_f, CV_32FC1, 1.0 / 255.0);
        const cv::Point2d shift = cv::phaseCorrelate(curr_f, anchor_f);
        metrics.fallback = reason;
        return translation_h(shift.x, shift.y);
    };

    if (des_curr.empty() || des_anchor.empty() || kp_curr.size() < 8 || kp_anchor.size() < 8) {
        return {phase_fallback("few_features"), metrics};
    }

    cv::BFMatcher matcher(norm);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(des_curr, des_anchor, knn, 2);
    std::vector<cv::DMatch> good;
    for (const auto& pair : knn) {
        if (pair.size() == 2 && pair[0].distance < 0.75f * pair[1].distance) {
            good.push_back(pair[0]);
        }
    }
    if (good.size() < 8) {
        return {phase_fallback("few_matches"), metrics};
    }

    std::vector<cv::Point2f> pts_curr;
    std::vector<cv::Point2f> pts_anchor;
    pts_curr.reserve(good.size());
    pts_anchor.reserve(good.size());
    for (const auto& match : good) {
        pts_curr.push_back(kp_curr[static_cast<size_t>(match.queryIdx)].pt);
        pts_anchor.push_back(kp_anchor[static_cast<size_t>(match.trainIdx)].pt);
    }

    cv::Mat inlier_mask;
    cv::Mat h = cv::findHomography(pts_curr, pts_anchor, cv::RANSAC, 3.0, inlier_mask);
    if (h.empty() || inlier_mask.empty() || cv::countNonZero(inlier_mask) < 8) {
        return {phase_fallback("bad_homography"), metrics};
    }
    h.convertTo(h, CV_32FC1);
    metrics.homography_inliers = cv::countNonZero(inlier_mask);

    std::vector<cv::Point2f> projected;
    cv::perspectiveTransform(pts_curr, projected, h);
    std::vector<double> errors;
    for (size_t i = 0; i < projected.size(); ++i) {
        if (inlier_mask.at<unsigned char>(static_cast<int>(i), 0) == 0) {
            continue;
        }
        const cv::Point2f d = projected[i] - pts_anchor[i];
        errors.push_back(std::sqrt(static_cast<double>(d.x * d.x + d.y * d.y)));
    }
    if (!errors.empty()) {
        std::nth_element(errors.begin(), errors.begin() + static_cast<std::ptrdiff_t>(errors.size() / 2), errors.end());
        metrics.homography_error_px = errors[errors.size() / 2];
    }
    return {h, metrics};
}

static cv::Mat warp_like(const cv::Mat& src, const cv::Mat& h, cv::Size size, int interp, int border_type = cv::BORDER_CONSTANT) {
    cv::Mat dst;
    cv::warpPerspective(src, dst, h, size, interp, border_type, cv::Scalar::all(0));
    return dst;
}

static cv::Mat dilate_mask(const cv::Mat& mask, int ksize) {
    if (ksize <= 1 || cv::countNonZero(mask) == 0) {
        return mask.clone();
    }
    cv::Mat out;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(ksize, ksize));
    cv::dilate(mask, out, kernel);
    return out;
}

static cv::Mat anchor_ring_mask_stream(
    const cv::Mat& change_mask,
    const cv::Mat& old_valid,
    const cv::Mat& current_valid,
    int inner_radius = 8,
    int outer_radius = 32) {
    cv::Mat result = cv::Mat::zeros(change_mask.size(), CV_8UC1);
    if (change_mask.empty() || cv::countNonZero(change_mask) == 0
        || cv::countNonZero(old_valid) == 0 || cv::countNonZero(current_valid) == 0) {
        return result;
    }
    const cv::Mat inner_kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(inner_radius * 2 + 1, inner_radius * 2 + 1));
    const cv::Mat outer_kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(outer_radius * 2 + 1, outer_radius * 2 + 1));
    cv::Mat inner;
    cv::Mat outer;
    cv::dilate(change_mask, inner, inner_kernel);
    cv::dilate(change_mask, outer, outer_kernel);
    cv::subtract(outer, inner, result);
    cv::bitwise_and(result, old_valid, result);
    cv::bitwise_and(result, current_valid, result);
    return result;
}

static cv::Mat stream_model_valid(
    const cv::Mat& depth,
    const cv::Mat& confidence,
    const cv::Mat& valid_warp,
    float min_confidence) {
    cv::Mat result = cv::Mat::zeros(valid_warp.size(), CV_8UC1);
    for (int y = 0; y < valid_warp.rows; ++y) {
        for (int x = 0; x < valid_warp.cols; ++x) {
            const float z = depth.at<float>(y, x);
            const float c = confidence.at<float>(y, x);
            if (valid_warp.at<unsigned char>(y, x) != 0
                && std::isfinite(z) && z > 0.0f
                && std::isfinite(c) && c >= min_confidence) {
                result.at<unsigned char>(y, x) = 255;
            }
        }
    }
    return result;
}

static cv::Mat stream_depth_continuity(
    const cv::Mat& model_depth,
    const cv::Mat& canvas_depth,
    const cv::Mat& canvas_valid,
    const cv::Mat& apply_mask,
    const cv::Mat& anchor_ring) {
    cv::Mat result = model_depth.clone();
    if (cv::countNonZero(apply_mask) == 0 || cv::countNonZero(anchor_ring) < 64) {
        return result;
    }
    cv::Mat old_f;
    canvas_valid.convertTo(old_f, CV_32FC1, 1.0 / 255.0);
    cv::Mat source = cv::Mat::zeros(canvas_depth.size(), CV_32FC1);
    canvas_depth.copyTo(source, canvas_valid);
    cv::Mat numerator;
    cv::Mat denominator;
    cv::GaussianBlur(source, numerator, cv::Size(), 6.0, 6.0);
    cv::GaussianBlur(old_f, denominator, cv::Size(), 6.0, 6.0);
    cv::Mat safe_denominator;
    cv::add(denominator, cv::Scalar(1e-6), safe_denominator);
    cv::Mat smooth;
    cv::divide(numerator, safe_denominator, smooth);

    cv::Mat target_inverse;
    cv::bitwise_not(apply_mask, target_inverse);
    cv::Mat distance_to_target;
    cv::distanceTransform(target_inverse, distance_to_target, cv::DIST_L2, 3);
    cv::Mat boundary;
    cv::compare(distance_to_target, 2.5, boundary, cv::CMP_LE);
    cv::bitwise_and(boundary, canvas_valid, boundary);
    cv::Mat boundary_f;
    boundary.convertTo(boundary_f, CV_32FC1, 1.0 / 255.0);
    cv::Mat residual = canvas_depth - smooth;
    cv::Mat residual_masked;
    cv::multiply(residual, boundary_f, residual_masked);
    cv::Mat correction_num;
    cv::Mat correction_den;
    cv::GaussianBlur(residual_masked, correction_num, cv::Size(), 3.0, 3.0);
    cv::GaussianBlur(boundary_f, correction_den, cv::Size(), 3.0, 3.0);
    cv::Mat safe_correction_den;
    cv::add(correction_den, cv::Scalar(1e-6), safe_correction_den);
    cv::Mat correction;
    cv::divide(correction_num, safe_correction_den, correction);
    cv::Mat local = smooth + correction;
    cv::Mat local_support;
    cv::compare(denominator, 0.02, local_support, cv::CMP_GT);
    cv::Mat usable;
    cv::bitwise_and(apply_mask, local_support, usable);
    local.copyTo(result, usable);

    cv::Mat not_old;
    cv::bitwise_not(canvas_valid, not_old);
    cv::Mat new_only;
    cv::bitwise_and(apply_mask, not_old, new_only);
    cv::bitwise_and(new_only, local_support, new_only);
    local.copyTo(result, new_only);
    cv::Mat old_overlap;
    cv::bitwise_and(apply_mask, canvas_valid, old_overlap);
    canvas_depth.copyTo(result, old_overlap);
    return result;
}

static cv::Mat stream_texture_transfer(
    const cv::Mat& current_rgb,
    const cv::Mat& canvas_rgb,
    const cv::Mat& canvas_valid,
    const cv::Mat& current_valid,
    const cv::Mat& apply_mask,
    const cv::Mat& support_change,
    const cv::Mat& anchor_ring) {
    cv::Mat result = current_rgb.clone();
    if (cv::countNonZero(apply_mask) == 0 || cv::countNonZero(anchor_ring) == 0) {
        return result;
    }
    cv::Mat old_f;
    cv::Mat new_f;
    canvas_valid.convertTo(old_f, CV_32FC1, 1.0 / 255.0);
    current_valid.convertTo(new_f, CV_32FC1, 1.0 / 255.0);
    const double sigma = 24.0;
    cv::Mat old_den;
    cv::Mat new_den;
    cv::GaussianBlur(old_f, old_den, cv::Size(), sigma, sigma);
    cv::GaussianBlur(new_f, new_den, cv::Size(), sigma, sigma);

    std::vector<cv::Mat> old_channels;
    std::vector<cv::Mat> new_channels;
    cv::split(canvas_rgb, old_channels);
    cv::split(current_rgb, new_channels);
    std::vector<cv::Mat> correction_channels;
    correction_channels.reserve(3);
    cv::Mat stable;
    cv::bitwise_and(anchor_ring, canvas_valid, stable);
    cv::bitwise_and(stable, current_valid, stable);
    if (cv::countNonZero(stable) < 128) {
        cv::Mat not_support;
        cv::bitwise_not(support_change, not_support);
        cv::bitwise_and(canvas_valid, current_valid, stable);
        cv::bitwise_and(stable, not_support, stable);
    }
    if (cv::countNonZero(stable) < 128) {
        return result;
    }

    cv::Mat stable_f;
    stable.convertTo(stable_f, CV_32FC1, 1.0 / 255.0);
    for (int channel = 0; channel < 3; ++channel) {
        cv::Mat old_product;
        cv::Mat new_product;
        cv::multiply(old_channels[channel], old_f, old_product);
        cv::multiply(new_channels[channel], new_f, new_product);
        cv::Mat old_num;
        cv::Mat new_num;
        cv::GaussianBlur(old_product, old_num, cv::Size(), sigma, sigma);
        cv::GaussianBlur(new_product, new_num, cv::Size(), sigma, sigma);
        cv::Mat safe_old_den;
        cv::Mat safe_new_den;
        cv::add(old_den, cv::Scalar(1e-5), safe_old_den);
        cv::add(new_den, cv::Scalar(1e-5), safe_new_den);
        cv::Mat old_field;
        cv::Mat new_field;
        cv::divide(old_num, safe_old_den, old_field);
        cv::divide(new_num, safe_new_den, new_field);
        cv::Mat residual = old_field - new_field;
        std::vector<float> values;
        values.reserve(static_cast<std::size_t>(cv::countNonZero(stable)));
        for (int y = 0; y < residual.rows; ++y) {
            for (int x = 0; x < residual.cols; ++x) {
                if (stable.at<unsigned char>(y, x) != 0) {
                    values.push_back(residual.at<float>(y, x));
                }
            }
        }
        const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2U);
        std::nth_element(values.begin(), middle, values.end());
        const float center = *middle;
        std::vector<float> deviations;
        deviations.reserve(values.size());
        for (const float value : values) {
            deviations.push_back(std::abs(value - center));
        }
        const auto deviation_middle = deviations.begin()
            + static_cast<std::ptrdiff_t>(deviations.size() / 2U);
        std::nth_element(deviations.begin(), deviation_middle, deviations.end());
        const float limit = std::max(0.04f, 3.0f * 1.4826f * std::max(*deviation_middle, 1e-4f));
        cv::Mat clipped = cv::Mat::zeros(residual.size(), CV_32FC1);
        for (int y = 0; y < residual.rows; ++y) {
            for (int x = 0; x < residual.cols; ++x) {
                if (stable.at<unsigned char>(y, x) != 0) {
                    clipped.at<float>(y, x) = std::clamp(
                        residual.at<float>(y, x), center - limit, center + limit);
                }
            }
        }
        cv::Mat product;
        cv::multiply(clipped, stable_f, product);
        cv::Mat numerator;
        cv::Mat denominator;
        cv::GaussianBlur(product, numerator, cv::Size(), sigma, sigma);
        cv::GaussianBlur(stable_f, denominator, cv::Size(), sigma, sigma);
        cv::Mat safe_denominator;
        cv::add(denominator, cv::Scalar(1e-5), safe_denominator);
        cv::Mat correction;
        cv::divide(numerator, safe_denominator, correction);
        for (int y = 0; y < correction.rows; ++y) {
            for (int x = 0; x < correction.cols; ++x) {
                correction.at<float>(y, x) = std::clamp(
                    correction.at<float>(y, x), -0.05f, 0.05f);
            }
        }
        correction_channels.push_back(correction);
    }
    cv::Mat correction;
    cv::merge(correction_channels, correction);
    cv::Mat transferred = current_rgb.clone();
    for (int y = 0; y < transferred.rows; ++y) {
        for (int x = 0; x < transferred.cols; ++x) {
            if (apply_mask.at<unsigned char>(y, x) == 0) {
                continue;
            }
            cv::Vec3f value = current_rgb.at<cv::Vec3f>(y, x) + correction.at<cv::Vec3f>(y, x);
            for (int channel = 0; channel < 3; ++channel) {
                value[channel] = std::clamp(value[channel], 0.0f, 1.0f);
            }
            transferred.at<cv::Vec3f>(y, x) = value;
        }
    }

    cv::Mat non_anchor;
    cv::bitwise_not(anchor_ring, non_anchor);
    cv::Mat distance_to_ring;
    cv::distanceTransform(non_anchor, distance_to_ring, cv::DIST_L2, 3);
    cv::Mat overlap;
    cv::bitwise_and(apply_mask, support_change, overlap);
    cv::bitwise_and(overlap, canvas_valid, overlap);
    cv::bitwise_and(overlap, current_valid, overlap);
    for (int y = 0; y < transferred.rows; ++y) {
        for (int x = 0; x < transferred.cols; ++x) {
            if (overlap.at<unsigned char>(y, x) == 0) {
                continue;
            }
            const float old_mix = std::clamp(
                1.0f - distance_to_ring.at<float>(y, x) / 16.0f, 0.65f, 0.95f);
            transferred.at<cv::Vec3f>(y, x) =
                transferred.at<cv::Vec3f>(y, x) * (1.0f - old_mix)
                + canvas_rgb.at<cv::Vec3f>(y, x) * old_mix;
        }
    }
    transferred.copyTo(result, apply_mask);
    return result;
}

static cv::Mat compute_change_mask(
    const Frame& frame,
    const cv::Mat& h,
    CanvasState& canvas,
    const Args& args,
    StreamMetrics& metrics,
    cv::Mat& warped_rgb_f,
    cv::Mat& valid_warp,
    cv::Mat& support_change,
    cv::Mat& photometric_change) {
    if (!canvas.initialized) {
        warped_rgb_f = frame.rgb_f.clone();
        valid_warp = frame.support.clone();
        support_change = valid_warp.clone();
        photometric_change = cv::Mat::zeros(valid_warp.size(), CV_8UC1);
        metrics.support_changed_ratio = static_cast<double>(cv::countNonZero(valid_warp)) / static_cast<double>(valid_warp.total());
        return valid_warp.clone();
    }

    warped_rgb_f = warp_like(frame.rgb_f, h, cv::Size(args.width, args.height), cv::INTER_LINEAR);
    cv::Mat warped_support = warp_like(frame.support, h, cv::Size(args.width, args.height), cv::INTER_NEAREST);
    cv::threshold(warped_support, valid_warp, 127.0, 255.0, cv::THRESH_BINARY);

    cv::Mat overlap;
    cv::bitwise_and(valid_warp, canvas.support, overlap);
    cv::bitwise_and(overlap, canvas.valid, overlap);

    cv::Mat absdiff;
    cv::absdiff(warped_rgb_f, canvas.rgb_f, absdiff);
    std::vector<cv::Mat> channels;
    cv::split(absdiff, channels);
    cv::Mat diff = (channels[0] + channels[1] + channels[2]) / 3.0f;

    cv::Mat photo_mask = cv::Mat::zeros(args.height, args.width, CV_8UC1);
    cv::Mat thresholded;
    cv::threshold(diff, thresholded, args.image_l1_thr, 255.0, cv::THRESH_BINARY);
    thresholded.convertTo(thresholded, CV_8UC1);
    cv::bitwise_and(thresholded, overlap, photo_mask);

    cv::Mat inv_support;
    cv::bitwise_not(canvas.support, inv_support);
    cv::bitwise_and(valid_warp, inv_support, support_change);

    cv::Mat change;
    cv::bitwise_or(photo_mask, support_change, change);
    change = dilate_mask(change, args.dilate_ksize);
    cv::bitwise_and(change, valid_warp, change);

    metrics.photometric_changed_ratio = static_cast<double>(cv::countNonZero(photo_mask)) / static_cast<double>(photo_mask.total());
    metrics.support_changed_ratio = static_cast<double>(cv::countNonZero(support_change)) / static_cast<double>(support_change.total());
    photometric_change = photo_mask;
    return change;
}

static cv::Mat align_depth_to_canvas(
    const cv::Mat& warped_depth,
    const cv::Mat& warped_conf,
    const cv::Mat& valid_warp,
    const cv::Mat& anchor_mask,
    const CanvasState& canvas,
    const Args& args) {
    if (!canvas.initialized) {
        return warped_depth.clone();
    }
    const bool use_anchor = !anchor_mask.empty() && cv::countNonZero(anchor_mask) >= 128;
    const cv::Mat& calibration_mask = use_anchor ? anchor_mask : valid_warp;
    double sum_z = 0.0;
    double sum_dst = 0.0;
    double sum_zz = 0.0;
    double sum_zdst = 0.0;
    int count = 0;
    for (int y = 0; y < warped_depth.rows; ++y) {
        for (int x = 0; x < warped_depth.cols; ++x) {
            if (calibration_mask.at<unsigned char>(y, x) == 0
                || valid_warp.at<unsigned char>(y, x) == 0
                || canvas.valid.at<unsigned char>(y, x) == 0) {
                continue;
            }
            const float z = warped_depth.at<float>(y, x);
            const float dst = canvas.depth.at<float>(y, x);
            const float conf = warped_conf.at<float>(y, x);
            if (!std::isfinite(z) || !std::isfinite(dst) || conf < static_cast<float>(args.min_conf)) {
                continue;
            }
            sum_z += z;
            sum_dst += dst;
            sum_zz += static_cast<double>(z) * z;
            sum_zdst += static_cast<double>(z) * dst;
            ++count;
        }
    }
    if (count < 128 && use_anchor) {
        sum_z = 0.0;
        sum_dst = 0.0;
        sum_zz = 0.0;
        sum_zdst = 0.0;
        count = 0;
        for (int y = 0; y < warped_depth.rows; ++y) {
            for (int x = 0; x < warped_depth.cols; ++x) {
                if (valid_warp.at<unsigned char>(y, x) == 0 || canvas.valid.at<unsigned char>(y, x) == 0) {
                    continue;
                }
                const float z = warped_depth.at<float>(y, x);
                const float dst = canvas.depth.at<float>(y, x);
                const float conf = warped_conf.at<float>(y, x);
                if (!std::isfinite(z) || !std::isfinite(dst) || conf < static_cast<float>(args.min_conf)) {
                    continue;
                }
                sum_z += z;
                sum_dst += dst;
                sum_zz += static_cast<double>(z) * z;
                sum_zdst += static_cast<double>(z) * dst;
                ++count;
            }
        }
    }
    if (count < 128) {
        return warped_depth.clone();
    }
    const double denom = static_cast<double>(count) * sum_zz - sum_z * sum_z;
    double scale = 1.0;
    double bias = 0.0;
    if (std::abs(denom) > 1e-9) {
        scale = (static_cast<double>(count) * sum_zdst - sum_z * sum_dst) / denom;
        bias = (sum_dst - scale * sum_z) / static_cast<double>(count);
    }
    scale = std::clamp(scale, 0.25, 4.0);
    bias = std::clamp(bias, -10.0, 10.0);
    cv::Mat aligned;
    warped_depth.convertTo(aligned, CV_32FC1, scale, bias);
    return aligned;
}

static int fuse(CanvasState& canvas, const cv::Mat& warped_rgb_f, const cv::Mat& aligned_depth, const cv::Mat& warped_conf, const cv::Mat& update_mask, const Args& args) {
    if (!canvas.initialized) {
        canvas.rgb_f = cv::Mat::zeros(aligned_depth.rows, aligned_depth.cols, CV_32FC3);
        canvas.depth = cv::Mat::zeros(aligned_depth.rows, aligned_depth.cols, CV_32FC1);
        canvas.conf = cv::Mat::zeros(aligned_depth.rows, aligned_depth.cols, CV_32FC1);
        canvas.weight = cv::Mat::zeros(aligned_depth.rows, aligned_depth.cols, CV_32FC1);
        canvas.valid = cv::Mat::zeros(aligned_depth.rows, aligned_depth.cols, CV_8UC1);
        canvas.support = cv::Mat::zeros(aligned_depth.rows, aligned_depth.cols, CV_8UC1);
        canvas.initialized = true;
    }

    int fused = 0;
    for (int y = 0; y < aligned_depth.rows; ++y) {
        for (int x = 0; x < aligned_depth.cols; ++x) {
            if (update_mask.at<unsigned char>(y, x) == 0) {
                continue;
            }
            const float z = aligned_depth.at<float>(y, x);
            const float c = warped_conf.at<float>(y, x);
            if (!std::isfinite(z) || c < static_cast<float>(args.min_conf)) {
                continue;
            }
            const float w_old = canvas.weight.at<float>(y, x);
            const float w_obs = std::clamp(c, 1e-4f, 32.0f);
            const float w_new = std::min(w_old + w_obs, 32.0f);
            canvas.depth.at<float>(y, x) = (canvas.depth.at<float>(y, x) * w_old + z * w_obs) / std::max(w_new, 1e-6f);
            canvas.conf.at<float>(y, x) = std::max(canvas.conf.at<float>(y, x), c);
            canvas.weight.at<float>(y, x) = w_new;
            canvas.valid.at<unsigned char>(y, x) = 255;
            canvas.support.at<unsigned char>(y, x) = 255;
            const cv::Vec3f old_rgb = canvas.rgb_f.at<cv::Vec3f>(y, x);
            const cv::Vec3f obs_rgb = warped_rgb_f.at<cv::Vec3f>(y, x);
            canvas.rgb_f.at<cv::Vec3f>(y, x) = (old_rgb * w_old + obs_rgb * w_obs) * (1.0f / std::max(w_new, 1e-6f));
            ++fused;
        }
    }
    return fused;
}

static float percentile(std::vector<float> values, double percent) {
    values.erase(std::remove_if(values.begin(), values.end(), [](float v) { return !std::isfinite(v); }), values.end());
    if (values.empty()) {
        return 0.0f;
    }
    const double clamped = std::max(0.0, std::min(100.0, percent));
    const size_t index = static_cast<size_t>(std::floor((clamped / 100.0) * static_cast<double>(values.size() - 1)));
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(index), values.end());
    return values[index];
}

static int write_canvas_ply(const fs::path& path, const CanvasState& canvas, const Args& args) {
    std::vector<float> conf_values;
    conf_values.reserve(canvas.conf.total());
    for (int y = 0; y < canvas.conf.rows; ++y) {
        for (int x = 0; x < canvas.conf.cols; ++x) {
            if (canvas.valid.at<unsigned char>(y, x) != 0) {
                conf_values.push_back(canvas.conf.at<float>(y, x));
            }
        }
    }
    const float threshold = args.conf_percentile <= 0.0 ? 0.0f : percentile(conf_values, args.conf_percentile);

    struct Vertex {
        float x;
        float y;
        float z;
        unsigned char r;
        unsigned char g;
        unsigned char b;
    };
    std::vector<Vertex> vertices;
    const float scale = static_cast<float>(std::max(canvas.depth.rows, canvas.depth.cols));
    for (int y = 0; y < canvas.depth.rows; ++y) {
        for (int x = 0; x < canvas.depth.cols; ++x) {
            if (canvas.valid.at<unsigned char>(y, x) == 0) {
                continue;
            }
            const float conf = canvas.conf.at<float>(y, x);
            const float z = canvas.depth.at<float>(y, x);
            if (conf < threshold || !std::isfinite(z)) {
                continue;
            }
            const float px = (static_cast<float>(x) - static_cast<float>(canvas.depth.cols) * 0.5f) / scale;
            const float py = -(static_cast<float>(y) - static_cast<float>(canvas.depth.rows) * 0.5f) / scale;
            const cv::Vec3f color = canvas.rgb_f.at<cv::Vec3f>(y, x);
            vertices.push_back(Vertex{
                px,
                py,
                z,
                static_cast<unsigned char>(std::clamp(color[0] * 255.0f, 0.0f, 255.0f)),
                static_cast<unsigned char>(std::clamp(color[1] * 255.0f, 0.0f, 255.0f)),
                static_cast<unsigned char>(std::clamp(color[2] * 255.0f, 0.0f, 255.0f))});
        }
    }

    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to write PLY: " + path.string());
    }
    out << "ply\nformat ascii 1.0\n";
    out << "element vertex " << vertices.size() << "\n";
    out << "property float x\nproperty float y\nproperty float z\n";
    out << "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
    out << std::fixed << std::setprecision(7);
    for (const Vertex& v : vertices) {
        out << v.x << ' ' << v.y << ' ' << v.z << ' '
            << static_cast<int>(v.r) << ' ' << static_cast<int>(v.g) << ' ' << static_cast<int>(v.b) << '\n';
    }
    return static_cast<int>(vertices.size());
}

static void save_debug_images(
    const fs::path& debug_dir,
    const Frame& frame,
    const cv::Mat& warped_rgb_f,
    const cv::Mat& change_mask,
    const cv::Mat& valid_warp) {
    fs::create_directories(debug_dir);
    cv::Mat warped_u8;
    warped_rgb_f.convertTo(warped_u8, CV_8UC3, 255.0);
    cv::Mat warped_bgr;
    cv::cvtColor(warped_u8, warped_bgr, cv::COLOR_RGB2BGR);
    const std::string stem = "frame_" + std::to_string(frame.frame_id);
    cv::imwrite((debug_dir / (stem + "_warped.png")).string(), warped_bgr);
    cv::imwrite((debug_dir / (stem + "_change_mask.png")).string(), change_mask);
    cv::imwrite((debug_dir / (stem + "_support.png")).string(), valid_warp);
}

static void write_timing_report(const fs::path& path, const std::vector<StreamMetrics>& rows) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to write timing report: " + path.string());
    }
    out << "# C++ OmniVGGT aligned stream timing\n\n";
    out << "| frame | image | total_ms | align2d_ms | diff_ms | model_ms | depth_align_ms | fuse_ms | changed_ratio | fused_pixels | points | homography_inliers | fallback | skipped_model |\n";
    out << "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|\n";
    out << std::fixed << std::setprecision(2);
    for (const auto& row : rows) {
        out << "| " << row.frame_id << " | " << row.image
            << " | " << row.total_ms
            << " | " << row.align2d_ms
            << " | " << row.diff_ms
            << " | " << row.model_ms
            << " | " << row.depth_align_ms
            << " | " << row.fuse_ms;
        out << std::setprecision(4) << " | " << row.changed_ratio;
        out << std::setprecision(2)
            << " | " << row.fused_pixels
            << " | " << row.point_count
            << " | " << row.homography_inliers
            << " | " << (row.fallback.empty() ? "None" : row.fallback)
            << " | " << (row.skipped_model ? "yes" : "no") << " |\n";
    }
}

static int count_valid_points(const CanvasState& canvas) {
    if (!canvas.initialized) {
        return 0;
    }
    return cv::countNonZero(canvas.valid);
}

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        fs::create_directories(args.output_dir);

        torch::NoGradGuard no_grad;
        const torch::Device device = parse_device(args.device);
        const torch::ScalarType dtype = parse_dtype(args.dtype);
        torch::jit::script::Module module = torch::jit::load(args.model, device);
        module.eval();

        const std::vector<fs::path> image_paths = list_images(args.image_dir, args.num_images);
        CanvasState canvas;
        std::vector<StreamMetrics> all_metrics;
        all_metrics.reserve(image_paths.size());

        for (size_t i = 0; i < image_paths.size(); ++i) {
            Timer total_timer;
            StreamMetrics metrics;
            metrics.frame_id = static_cast<int>(i);
            metrics.image = image_paths[i].filename().string();

            Timer t_read;
            Frame frame = load_frame(image_paths[i], static_cast<int>(i), args);
            metrics.read_ms = t_read.ms();

            Timer t_align;
            auto estimate = estimate_homography(frame, canvas.anchor_rgb_u8, metrics);
            cv::Mat h = estimate.first;
            metrics = estimate.second;
            metrics.align2d_ms = t_align.ms();

            Timer t_diff;
            cv::Mat warped_rgb_f;
            cv::Mat valid_warp;
            cv::Mat support_change;
            cv::Mat photometric_change;
            cv::Mat change_mask = compute_change_mask(
                frame,
                h,
                canvas,
                args,
                metrics,
                warped_rgb_f,
                valid_warp,
                support_change,
                photometric_change);
            metrics.changed_ratio = static_cast<double>(cv::countNonZero(change_mask)) / static_cast<double>(change_mask.total());
            if (metrics.changed_ratio >= args.scene_jump_ratio && metrics.fallback.empty()) {
                metrics.fallback = "scene_jump";
            }
            metrics.diff_ms = t_diff.ms();

            if (canvas.initialized && !metrics.fallback.empty() && metrics.fallback != "scene_jump") {
                change_mask.setTo(0);
                support_change.setTo(0);
                photometric_change.setTo(0);
                metrics.changed_ratio = 0.0;
            }

            const cv::Mat anchor_ring = canvas.initialized
                ? anchor_ring_mask_stream(change_mask, canvas.valid, valid_warp)
                : cv::Mat::zeros(valid_warp.size(), CV_8UC1);
            cv::Mat fusion_mask;
            if (cv::countNonZero(change_mask) > 0) {
                const cv::Mat close_kernel = cv::Mat::ones(9, 9, CV_8UC1);
                const cv::Mat band_kernel = cv::Mat::ones(17, 17, CV_8UC1);
                cv::Mat closed;
                cv::morphologyEx(change_mask, closed, cv::MORPH_CLOSE, close_kernel);
                cv::dilate(closed, fusion_mask, band_kernel);
                cv::bitwise_and(fusion_mask, valid_warp, fusion_mask);
            } else {
                fusion_mask = cv::Mat::zeros(valid_warp.size(), CV_8UC1);
            }

            const bool skip_model = canvas.initialized && metrics.changed_ratio <= args.no_change_ratio;
            metrics.skipped_model = skip_model;
            if (!skip_model) {
                Timer t_model;
                Frame model_frame = frame;
                if (canvas.initialized) {
                    cv::Mat composite = canvas.rgb_f.clone();
                    warped_rgb_f.copyTo(composite, fusion_mask);
                    model_frame.rgb_f = composite;
                    composite.convertTo(model_frame.rgb_u8, CV_8UC3, 255.0);
                }
                Prediction pred = run_model(module, model_frame, args.height, args.width, device, dtype);
                metrics.model_ms = t_model.ms();

                Timer t_depth;
                const cv::Mat model_to_canvas = canvas.initialized
                    ? cv::Mat::eye(3, 3, CV_32FC1)
                    : h;
                cv::Mat warped_depth = warp_like(pred.depth, model_to_canvas, cv::Size(args.width, args.height), cv::INTER_LINEAR);
                cv::Mat warped_conf = warp_like(pred.conf, model_to_canvas, cv::Size(args.width, args.height), cv::INTER_LINEAR);
                cv::Mat aligned_depth = align_depth_to_canvas(
                    warped_depth,
                    warped_conf,
                    valid_warp,
                    anchor_ring,
                    canvas,
                    args);
                metrics.depth_align_ms = t_depth.ms();

                Timer t_fuse;
                cv::Mat model_valid = stream_model_valid(
                    aligned_depth,
                    warped_conf,
                    valid_warp,
                    static_cast<float>(args.min_conf));
                cv::Mat not_support;
                cv::bitwise_not(support_change, not_support);
                cv::Mat photo_only_existing;
                cv::bitwise_and(photometric_change, not_support, photo_only_existing);
                if (canvas.initialized) {
                    cv::bitwise_and(photo_only_existing, canvas.valid, photo_only_existing);
                }
                cv::Mat update_mask;
                cv::bitwise_and(model_valid, change_mask, update_mask);
                cv::Mat not_photo_only;
                cv::bitwise_not(photo_only_existing, not_photo_only);
                cv::bitwise_and(update_mask, not_photo_only, update_mask);
                cv::Mat not_anchor;
                cv::bitwise_not(anchor_ring, not_anchor);
                cv::bitwise_and(update_mask, not_anchor, update_mask);

                cv::Mat color_bridge_mask = cv::Mat::zeros(update_mask.size(), CV_8UC1);
                cv::Mat color_bridge_mix = cv::Mat::zeros(update_mask.size(), CV_32FC1);
                cv::Mat fused_rgb = warped_rgb_f.clone();
                if (canvas.initialized) {
                    cv::Mat color_apply_mask;
                    const cv::Mat bridge_kernel = cv::getStructuringElement(
                        cv::MORPH_ELLIPSE, cv::Size(65, 65));
                    cv::Mat expanded_support;
                    cv::dilate(support_change, expanded_support, bridge_kernel);
                    cv::bitwise_and(expanded_support, canvas.valid, expanded_support);
                    cv::bitwise_and(expanded_support, valid_warp, expanded_support);
                    cv::bitwise_and(expanded_support, not_anchor, expanded_support);
                    cv::Mat not_update;
                    cv::bitwise_not(update_mask, not_update);
                    cv::bitwise_and(expanded_support, not_update, color_bridge_mask);
                    cv::bitwise_or(update_mask, color_bridge_mask, color_apply_mask);
                    fused_rgb = stream_texture_transfer(
                        warped_rgb_f,
                        canvas.rgb_f,
                        canvas.valid,
                        valid_warp,
                        color_apply_mask,
                        support_change,
                        anchor_ring);

                    cv::Mat non_anchor;
                    cv::bitwise_not(anchor_ring, non_anchor);
                    cv::Mat distance_to_ring;
                    cv::distanceTransform(non_anchor, distance_to_ring, cv::DIST_L2, 3);
                    for (int y = 0; y < color_bridge_mask.rows; ++y) {
                        for (int x = 0; x < color_bridge_mask.cols; ++x) {
                            if (color_bridge_mask.at<unsigned char>(y, x) == 0) {
                                continue;
                            }
                            const float alpha = std::clamp(
                                distance_to_ring.at<float>(y, x) / 8.0f, 0.0f, 1.0f);
                            color_bridge_mix.at<float>(y, x) = alpha;
                            fused_rgb.at<cv::Vec3f>(y, x) =
                                fused_rgb.at<cv::Vec3f>(y, x) * alpha
                                + canvas.rgb_f.at<cv::Vec3f>(y, x) * (1.0f - alpha);
                        }
                    }
                }

                const cv::Mat continuity_mask = [&]() {
                    cv::Mat mask;
                    cv::bitwise_and(update_mask, support_change, mask);
                    return mask;
                }();
                if (canvas.initialized && cv::countNonZero(continuity_mask) > 0) {
                    aligned_depth = stream_depth_continuity(
                        aligned_depth,
                        canvas.depth,
                        canvas.valid,
                        continuity_mask,
                        anchor_ring);
                }
                metrics.fused_pixels = fuse(canvas, fused_rgb, aligned_depth, warped_conf, update_mask, args);
                if (canvas.initialized && cv::countNonZero(color_bridge_mask) > 0) {
                    for (int y = 0; y < color_bridge_mask.rows; ++y) {
                        for (int x = 0; x < color_bridge_mask.cols; ++x) {
                            if (color_bridge_mask.at<unsigned char>(y, x) != 0) {
                                canvas.rgb_f.at<cv::Vec3f>(y, x) = fused_rgb.at<cv::Vec3f>(y, x);
                            }
                        }
                    }
                }
                metrics.fuse_ms = t_fuse.ms();
            }

            if (!canvas.anchor_rgb_u8.data) {
                canvas.anchor_rgb_u8 = frame.rgb_u8.clone();
            }
            if (canvas.initialized) {
                cv::bitwise_or(canvas.support, valid_warp, canvas.support);
            }
            if (args.save_debug) {
                save_debug_images(fs::path(args.output_dir) / "debug", frame, warped_rgb_f, change_mask, valid_warp);
            }

            metrics.point_count = count_valid_points(canvas);
            metrics.total_ms = total_timer.ms();
            all_metrics.push_back(metrics);
            std::cout << "frame=" << metrics.frame_id
                      << " changed=" << std::fixed << std::setprecision(4) << metrics.changed_ratio
                      << " model_ms=" << std::setprecision(2) << metrics.model_ms
                      << " fused=" << metrics.fused_pixels
                      << " points=" << metrics.point_count
                      << " fallback=" << (metrics.fallback.empty() ? "None" : metrics.fallback)
                      << "\n";
        }

        const fs::path out_dir(args.output_dir);
        const int point_count = write_canvas_ply(out_dir / "stream_pointcloud.ply", canvas, args);
        write_timing_report(out_dir / "timings.md", all_metrics);
        if (canvas.initialized) {
            cv::Mat canvas_u8;
            canvas.rgb_f.convertTo(canvas_u8, CV_8UC3, 255.0);
            cv::Mat canvas_bgr;
            cv::cvtColor(canvas_u8, canvas_bgr, cv::COLOR_RGB2BGR);
            cv::imwrite((out_dir / "canvas_rgb.png").string(), canvas_bgr);
        }
        std::cout << "Wrote stream pointcloud with " << point_count << " points to " << (out_dir / "stream_pointcloud.ply") << "\n";
        std::cout << "Done. Outputs are in " << out_dir << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << "\n";
        return 1;
    }
}
