#include "inference_pipeline.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <ATen/autocast_mode.h>
#include <c10/core/InferenceMode.h>
#include <torch/csrc/jit/api/function_impl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace omnivggt::observer {

namespace {

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

int round_to_multiple_14(const double value) {
    return std::max(14, static_cast<int>(std::round(value / 14.0)) * 14);
}

int floor_to_multiple_14(const double value) {
    return std::max(14, static_cast<int>(std::floor(value / 14.0)) * 14);
}

constexpr std::size_t kMaxDynamicPairGpuCache = 2U;

// Python's _bucket_roi_size preserves the ROI aspect ratio and only pads the
// model input through the patch-size floor.  Stretching every ROI to 700x700
// changes the scene geometry before inference and is the main reason the old
// C++ replay produced a vertically discontinuous surface.
std::pair<int, int> bucket_roi_size(
    const int crop_width,
    const int crop_height,
    const int max_width,
    const int max_height) {
    if (crop_width <= 0 || crop_height <= 0 || max_width < 14 || max_height < 14) {
        throw std::runtime_error("invalid ROI/model bucket dimensions");
    }
    const double scale = std::min(
        static_cast<double>(max_width) / static_cast<double>(crop_width),
        static_cast<double>(max_height) / static_cast<double>(crop_height));
    const int target_width = std::clamp(
        floor_to_multiple_14(static_cast<double>(crop_width) * scale), 14, max_width);
    const int target_height = std::clamp(
        floor_to_multiple_14(static_cast<double>(crop_height) * scale), 14, max_height);
    return {target_width, target_height};
}

void configure_torchscript_executor(torch::jit::script::Module& module) {
    // LibTorch's default profiling executor can spend minutes specializing
    // this transformer graph on the first call.  Python eager inference does
    // not use that pass, so use the simple executor for every loaded bucket.
    auto& forward_function = torch::jit::toGraphFunction(
        module.get_method("forward").function());
    forward_function._set_initial_executor_execution_mode(torch::jit::SIMPLE);
}

std::filesystem::path pair_artifact_for_shape(
    const std::filesystem::path& base_path,
    const std::filesystem::path& bucket_dir,
    const int width,
    const int height) {
    if (bucket_dir.empty()) {
        return base_path;
    }
    const std::string filename = base_path.filename().string();
    const std::size_t marker = filename.find("_s2_");
    if (marker == std::string::npos) {
        throw std::runtime_error(
            "cannot derive dynamic pair artifact name from: " + filename);
    }
    const std::size_t dimensions_begin = marker + 4U;
    const std::size_t suffix_begin = filename.find('_', dimensions_begin);
    if (suffix_begin == std::string::npos) {
        throw std::runtime_error(
            "pair artifact name must contain a suffix after WxH: " + filename);
    }
    const std::string candidate_name = filename.substr(0, dimensions_begin)
        + std::to_string(width) + "x" + std::to_string(height)
        + filename.substr(suffix_begin);
    const std::filesystem::path candidate = bucket_dir / candidate_name;
    if (std::filesystem::exists(candidate)) {
        return candidate;
    }
    // The compact observer wrapper and the full pair wrapper have identical
    // depth/camera heads. Keep one dynamic bucket family on disk: when the
    // launcher names the observer artifact, fall back to the corresponding
    // full artifact instead of switching to a fixed-shape graph.
    constexpr const char* observer_prefix = "omnivggt_observer_s2_";
    constexpr const char* full_prefix = "omnivggt_s2_";
    if (candidate_name.rfind(observer_prefix, 0U) == 0U) {
        const std::string full_name = std::string(full_prefix)
            + candidate_name.substr(std::char_traits<char>::length(observer_prefix));
        const std::filesystem::path full_candidate = bucket_dir / full_name;
        if (std::filesystem::exists(full_candidate)) {
            return full_candidate;
        }
    }
    return candidate;
}

std::vector<std::pair<int, int>> discover_dynamic_pair_shapes(
    const std::filesystem::path& bucket_dir) {
    std::vector<std::pair<int, int>> shapes;
    if (bucket_dir.empty() || !std::filesystem::is_directory(bucket_dir)) {
        return shapes;
    }
    for (const auto& entry : std::filesystem::directory_iterator(bucket_dir)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".pt") {
            continue;
        }
        const std::string filename = entry.path().filename().string();
        const std::size_t marker = filename.find("_s2_");
        if (marker == std::string::npos) {
            continue;
        }
        const std::size_t dimensions_begin = marker + 4U;
        const std::size_t separator = filename.find('x', dimensions_begin);
        const std::size_t suffix_begin = filename.find('_', separator + 1U);
        if (separator == std::string::npos || suffix_begin == std::string::npos) {
            continue;
        }
        try {
            const int width = std::stoi(filename.substr(
                dimensions_begin, separator - dimensions_begin));
            const int height = std::stoi(filename.substr(
                separator + 1U, suffix_begin - separator - 1U));
            if (width > 0 && height > 0) {
                shapes.emplace_back(width, height);
            }
        } catch (const std::exception&) {
            // Ignore unrelated .pt files in the artifact directory.
        }
    }
    std::sort(shapes.begin(), shapes.end());
    shapes.erase(std::unique(shapes.begin(), shapes.end()), shapes.end());
    return shapes;
}

cv::Mat quantize_rgb_u8(const cv::Mat& rgb_f) {
    if (rgb_f.type() != CV_32FC3) {
        throw std::runtime_error("quantize_rgb_u8 expects CV_32FC3 RGB input");
    }
    cv::Mat result(rgb_f.size(), CV_8UC3);
    for (int y = 0; y < rgb_f.rows; ++y) {
        for (int x = 0; x < rgb_f.cols; ++x) {
            const cv::Vec3f value = rgb_f.at<cv::Vec3f>(y, x);
            cv::Vec3b& output = result.at<cv::Vec3b>(y, x);
            for (int channel = 0; channel < 3; ++channel) {
                output[channel] = static_cast<std::uint8_t>(std::clamp(
                    static_cast<int>(std::floor(std::clamp(value[channel], 0.0f, 1.0f) * 255.0f)),
                    0,
                    255));
            }
        }
    }
    return result;
}

cv::Mat read_rgb(const std::filesystem::path& path) {
    const cv::Mat input = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
    if (input.empty()) {
        throw std::runtime_error("failed to read image: " + path.string());
    }
    cv::Mat rgb;
    if (input.channels() == 1) {
        cv::cvtColor(input, rgb, cv::COLOR_GRAY2RGB);
    } else if (input.channels() == 3) {
        cv::cvtColor(input, rgb, cv::COLOR_BGR2RGB);
    } else if (input.channels() == 4) {
        cv::cvtColor(input, rgb, cv::COLOR_BGRA2RGB);
    } else {
        throw std::runtime_error("unsupported image channel count: " + path.string());
    }
    return rgb;
}

#ifdef _WIN32
void load_libtorch_cuda_dlls() {
    const std::array<const char*, 2> dlls = {"c10_cuda.dll", "torch_cuda.dll"};
    for (const char* dll : dlls) {
        if (::LoadLibraryA(dll) == nullptr) {
            const DWORD error = ::GetLastError();
            throw std::runtime_error(
                std::string("failed to load ") + dll + " before CUDA initialization, GetLastError="
                + std::to_string(error));
        }
    }
}
#endif

torch::Device parse_device(const std::string& value) {
    if (value != "cuda") {
        throw std::runtime_error("observer inference currently requires --device cuda");
    }
#ifdef _WIN32
    load_libtorch_cuda_dlls();
#endif
    if (!torch::cuda::is_available()) {
        throw std::runtime_error("CUDA requested but torch::cuda::is_available() is false");
    }
    return torch::Device(torch::kCUDA);
}

torch::ScalarType parse_dtype(const std::string& value) {
    if (value == "float32") {
        return torch::kFloat32;
    }
    if (value == "float16") {
        return torch::kFloat16;
    }
    if (value == "bfloat16" || value == "bf16") {
        return torch::kBFloat16;
    }
    throw std::runtime_error("unsupported --dtype: " + value);
}

torch::Tensor to_cpu_float(torch::Tensor tensor) {
    return tensor.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32)).contiguous();
}

torch::Tensor make_image_tensor(const cv::Mat& rgb_u8) {
    const int height = rgb_u8.rows;
    const int width = rgb_u8.cols;
    std::vector<float> data(3U * static_cast<std::size_t>(height) * static_cast<std::size_t>(width), 0.0f);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const cv::Vec3f pixel = rgb_u8.type() == CV_32FC3
                ? rgb_u8.at<cv::Vec3f>(y, x)
                : cv::Vec3f(
                    static_cast<float>(rgb_u8.at<cv::Vec3b>(y, x)[0]) / 255.0f,
                    static_cast<float>(rgb_u8.at<cv::Vec3b>(y, x)[1]) / 255.0f,
                    static_cast<float>(rgb_u8.at<cv::Vec3b>(y, x)[2]) / 255.0f);
            for (int channel = 0; channel < 3; ++channel) {
                const std::size_t index =
                    (static_cast<std::size_t>(channel) * static_cast<std::size_t>(height)
                     + static_cast<std::size_t>(y))
                    * static_cast<std::size_t>(width)
                    + static_cast<std::size_t>(x);
                // The current Python stream path constructs model batches as
                // ``normalize_rgb((rgb_f * 255).astype(np.uint8))``.  That
                // truncation is part of the model input contract; passing the
                // pre-quantized float directly changes the transformer input
                // even though both values are nominally in [0, 1].
                const float normalized = std::clamp(pixel[channel], 0.0f, 1.0f);
                data[index] = rgb_u8.type() == CV_32FC3
                    ? std::floor(normalized * 255.0f) / 255.0f
                    : normalized;
            }
        }
    }
    return torch::from_blob(
               data.data(),
               {1, 1, 3, static_cast<int64_t>(height), static_cast<int64_t>(width)},
               torch::kFloat32)
        .clone();
}

torch::Tensor make_image_tensor_batch(const std::vector<cv::Mat>& images) {
    if (images.empty()) {
        throw std::runtime_error("OmniVGGT image batch is empty");
    }
    std::vector<torch::Tensor> tensors;
    tensors.reserve(images.size());
    for (const cv::Mat& image : images) {
        tensors.push_back(make_image_tensor(image));
    }
    return torch::cat(tensors, 1);
}

torch::Tensor make_image_tensor_independent_batch(const std::vector<cv::Mat>& images) {
    if (images.empty()) {
        throw std::runtime_error("OmniVGGT independent batch is empty");
    }
    std::vector<torch::Tensor> tensors;
    tensors.reserve(images.size());
    for (const cv::Mat& image : images) {
        tensors.push_back(make_image_tensor(image));
    }
    // make_image_tensor returns [1,1,3,H,W].  Concatenating on dim 0 is the
    // deliberate B=3,S=1 contract; concatenating on dim 1 would recreate the
    // native S=3 sequence path that produces independent height/color layers.
    return torch::cat(tensors, 0);
}

class CudaAutocastGuard {
public:
    explicit CudaAutocastGuard(const torch::ScalarType dtype)
        : active_(dtype == torch::kFloat16 || dtype == torch::kBFloat16),
          previous_enabled_(false),
          previous_dtype_(torch::kFloat16) {
        if (!active_) {
            return;
        }
        previous_enabled_ = at::autocast::is_autocast_enabled(at::kCUDA);
        previous_dtype_ = at::autocast::get_autocast_dtype(at::kCUDA);
        at::autocast::set_autocast_dtype(at::kCUDA, dtype);
        at::autocast::set_autocast_enabled(at::kCUDA, true);
    }

    CudaAutocastGuard(const CudaAutocastGuard&) = delete;
    CudaAutocastGuard& operator=(const CudaAutocastGuard&) = delete;

    ~CudaAutocastGuard() {
        if (active_) {
            at::autocast::set_autocast_dtype(at::kCUDA, previous_dtype_);
            at::autocast::set_autocast_enabled(at::kCUDA, previous_enabled_);
        }
    }

private:
    bool active_;
    bool previous_enabled_;
    torch::ScalarType previous_dtype_;
};

cv::Mat finite_mask(const cv::Mat& source) {
    cv::Mat result(source.rows, source.cols, CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < source.rows; ++y) {
        for (int x = 0; x < source.cols; ++x) {
            const float value = source.at<float>(y, x);
            result.at<std::uint8_t>(y, x) = std::isfinite(value) ? 255U : 0U;
        }
    }
    return result;
}

cv::Mat foreground_mask(const cv::Mat& rgb_f) {
    std::vector<cv::Mat> channels;
    cv::split(rgb_f, channels);
    cv::Mat gray = (channels[0] + channels[1] + channels[2]) / 3.0f;
    cv::Mat gray_u8(gray.rows, gray.cols, CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < gray.rows; ++y) {
        for (int x = 0; x < gray.cols; ++x) {
            gray_u8.at<std::uint8_t>(y, x) = static_cast<std::uint8_t>(std::clamp(
                static_cast<int>(gray.at<float>(y, x) * 255.0f), 0, 255));
        }
    }
    double otsu_threshold = 0.0;
    cv::Mat mask;
    otsu_threshold = cv::threshold(
        gray_u8, mask, 0.0, 255.0, cv::THRESH_BINARY | cv::THRESH_OTSU);
    const double threshold_value = std::max(0.10 * 255.0, otsu_threshold);
    cv::threshold(gray_u8, mask, threshold_value, 255.0, cv::THRESH_BINARY);
    if (static_cast<double>(cv::countNonZero(mask)) / static_cast<double>(mask.total()) > 0.94) {
        mask.setTo(255U);
        return mask;
    }
    const cv::Mat open_kernel = cv::Mat::ones(3, 3, CV_8UC1);
    const cv::Mat close_kernel = cv::Mat::ones(9, 9, CV_8UC1);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, open_kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, close_kernel);
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int count = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8, CV_32S);
    if (count <= 1) {
        return mask;
    }
    int largest = 1;
    for (int component = 2; component < count; ++component) {
        if (stats.at<int>(component, cv::CC_STAT_AREA)
            > stats.at<int>(largest, cv::CC_STAT_AREA)) {
            largest = component;
        }
    }
    if (stats.at<int>(largest, cv::CC_STAT_AREA)
        < static_cast<int>(0.08 * static_cast<double>(mask.total()))) {
        return mask;
    }
    return labels == largest;
}

cv::Mat anchor_ring_mask(
    const cv::Mat& change_mask,
    const cv::Mat& old_valid,
    const cv::Mat& current_valid,
    const int inner_radius = 8,
    const int outer_radius = 32) {
    cv::Mat result = cv::Mat::zeros(change_mask.size(), CV_8UC1);
    if (change_mask.empty() || cv::countNonZero(change_mask) == 0
        || cv::countNonZero(old_valid) == 0 || cv::countNonZero(current_valid) == 0) {
        return result;
    }
    const cv::Mat inner_kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(inner_radius * 2 + 1, inner_radius * 2 + 1));
    const cv::Mat outer_kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(outer_radius * 2 + 1, outer_radius * 2 + 1));
    cv::Mat inner;
    cv::Mat outer;
    cv::dilate(change_mask, inner, inner_kernel);
    cv::dilate(change_mask, outer, outer_kernel);
    cv::subtract(outer, inner, result);
    cv::bitwise_and(result, old_valid, result);
    cv::bitwise_and(result, current_valid, result);
    return result;
}

cv::Mat seam_fusion_mask(const cv::Mat& change_mask, const cv::Mat& valid_warp, const int radius = 8) {
    if (change_mask.empty() || cv::countNonZero(change_mask) == 0) {
        return cv::Mat::zeros(change_mask.size(), CV_8UC1);
    }
    const cv::Mat close_kernel = cv::Mat::ones(radius + 1, radius + 1, CV_8UC1);
    const cv::Mat band_kernel = cv::Mat::ones(radius * 2 + 1, radius * 2 + 1, CV_8UC1);
    cv::Mat closed;
    cv::Mat expanded;
    cv::morphologyEx(change_mask, closed, cv::MORPH_CLOSE, close_kernel);
    cv::dilate(closed, expanded, band_kernel);
    cv::bitwise_and(expanded, valid_warp, expanded);
    return expanded;
}

cv::Mat filter_components(const cv::Mat& mask, const int min_pixels) {
    if (mask.empty() || cv::countNonZero(mask) == 0 || min_pixels <= 1) {
        return mask.clone();
    }
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int component_count = cv::connectedComponentsWithStats(
        mask, labels, stats, centroids, 8, CV_32S);
    cv::Mat result = cv::Mat::zeros(mask.size(), CV_8UC1);
    for (int component = 1; component < component_count; ++component) {
        if (stats.at<int>(component, cv::CC_STAT_AREA) < min_pixels) {
            continue;
        }
        result.setTo(255U, labels == component);
    }
    return result;
}

cv::Mat model_valid_mask(
    const cv::Mat& depth,
    const cv::Mat& confidence,
    const cv::Mat& valid_warp,
    const float min_confidence) {
    cv::Mat result = cv::Mat::zeros(valid_warp.size(), CV_8UC1);
    for (int y = 0; y < valid_warp.rows; ++y) {
        for (int x = 0; x < valid_warp.cols; ++x) {
            const float z = depth.at<float>(y, x);
            const float c = confidence.at<float>(y, x);
            if (valid_warp.at<std::uint8_t>(y, x) != 0U
                && std::isfinite(z) && z > 0.0f
                && std::isfinite(c) && c >= min_confidence) {
                result.at<std::uint8_t>(y, x) = 255U;
            }
        }
    }
    return result;
}

float median_value(std::vector<float> values) {
    values.erase(
        std::remove_if(values.begin(), values.end(), [](const float value) {
            return !std::isfinite(value);
        }),
        values.end());
    if (values.empty()) {
        return 0.0f;
    }
    const std::size_t middle_index = values.size() / 2U;
    auto middle = values.begin() + static_cast<std::ptrdiff_t>(middle_index);
    std::nth_element(values.begin(), middle, values.end());
    const float upper = *middle;
    if ((values.size() & 1U) != 0U) {
        return upper;
    }
    auto lower = values.begin() + static_cast<std::ptrdiff_t>(middle_index - 1U);
    std::nth_element(values.begin(), lower, middle);
    return (*lower + upper) * 0.5f;
}

cv::Mat nearest_fill_values(const cv::Mat& values, const cv::Mat& source_mask) {
    cv::Mat result = cv::Mat::zeros(values.size(), CV_32FC1);
    if (values.empty() || source_mask.empty() || cv::countNonZero(source_mask) == 0) {
        return result;
    }
    const int width = values.cols;
    const int height = values.rows;
    std::vector<std::int32_t> nearest(
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height), -1);
    std::deque<int> queue;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (source_mask.at<std::uint8_t>(y, x) == 0U) {
                continue;
            }
            const int index = y * width + x;
            const float value = values.at<float>(y, x);
            if (!std::isfinite(value)) {
                continue;
            }
            nearest[static_cast<std::size_t>(index)] = index;
            result.at<float>(y, x) = value;
            queue.push_back(index);
        }
    }
    constexpr std::array<int, 8> dx = {-1, 0, 1, -1, 1, -1, 0, 1};
    constexpr std::array<int, 8> dy = {-1, -1, -1, 0, 0, 1, 1, 1};
    while (!queue.empty()) {
        const int index = queue.front();
        queue.pop_front();
        const int x = index % width;
        const int y = index / width;
        const int source_index = nearest[static_cast<std::size_t>(index)];
        for (int direction = 0; direction < 8; ++direction) {
            const int nx = x + dx[static_cast<std::size_t>(direction)];
            const int ny = y + dy[static_cast<std::size_t>(direction)];
            if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
                continue;
            }
            const int next = ny * width + nx;
            if (nearest[static_cast<std::size_t>(next)] >= 0) {
                continue;
            }
            nearest[static_cast<std::size_t>(next)] = source_index;
            result.at<float>(ny, nx) = values.at<float>(
                source_index / width, source_index % width);
            queue.push_back(next);
        }
    }
    return result;
}

cv::Mat propagate_seam_residual(
    const cv::Mat& residual,
    const cv::Mat& seam_mask,
    const cv::Mat& target_mask,
    const double sigma,
    const float max_abs) {
    cv::Mat correction = cv::Mat::zeros(residual.size(), CV_32FC1);
    if (seam_mask.empty() || target_mask.empty()
        || cv::countNonZero(seam_mask) == 0 || cv::countNonZero(target_mask) == 0) {
        return correction;
    }
    std::vector<float> seam_values;
    seam_values.reserve(static_cast<std::size_t>(cv::countNonZero(seam_mask)));
    for (int y = 0; y < residual.rows; ++y) {
        for (int x = 0; x < residual.cols; ++x) {
            if (seam_mask.at<std::uint8_t>(y, x) != 0U
                && std::isfinite(residual.at<float>(y, x))) {
                seam_values.push_back(std::clamp(residual.at<float>(y, x), -max_abs, max_abs));
            }
        }
    }
    if (seam_values.size() < 32U) {
        return correction;
    }
    const float center = median_value(seam_values);
    std::vector<float> deviations;
    deviations.reserve(seam_values.size());
    for (const float value : seam_values) {
        deviations.push_back(std::abs(value - center));
    }
    const float mad = std::max(median_value(deviations), 1e-6f);
    const float limit = 3.0f * 1.4826f * mad;
    cv::Mat robust_mask = cv::Mat::zeros(seam_mask.size(), CV_8UC1);
    cv::Mat clipped = cv::Mat::zeros(residual.size(), CV_32FC1);
    for (int y = 0; y < residual.rows; ++y) {
        for (int x = 0; x < residual.cols; ++x) {
            if (seam_mask.at<std::uint8_t>(y, x) == 0U) {
                continue;
            }
            const float value = residual.at<float>(y, x);
            if (std::isfinite(value) && std::abs(value - center) <= limit) {
                robust_mask.at<std::uint8_t>(y, x) = 255U;
                clipped.at<float>(y, x) = std::clamp(value, -max_abs, max_abs);
            }
        }
    }
    if (cv::countNonZero(robust_mask) < 32) {
        robust_mask = seam_mask.clone();
        for (int y = 0; y < residual.rows; ++y) {
            for (int x = 0; x < residual.cols; ++x) {
                if (robust_mask.at<std::uint8_t>(y, x) != 0U) {
                    clipped.at<float>(y, x) = std::clamp(residual.at<float>(y, x), -max_abs, max_abs);
                }
            }
        }
    }
    cv::Mat nearest = nearest_fill_values(clipped, robust_mask);
    cv::Mat target_f;
    target_mask.convertTo(target_f, CV_32FC1, 1.0 / 255.0);
    cv::Mat weighted_nearest;
    cv::multiply(nearest, target_f, weighted_nearest);
    cv::Mat smooth;
    cv::Mat weight;
    cv::GaussianBlur(weighted_nearest, smooth, cv::Size(), sigma, sigma);
    cv::GaussianBlur(target_f, weight, cv::Size(), sigma, sigma);
    cv::divide(smooth, weight + 1e-6f, correction);
    for (int y = 0; y < correction.rows; ++y) {
        for (int x = 0; x < correction.cols; ++x) {
            if (robust_mask.at<std::uint8_t>(y, x) != 0U) {
                correction.at<float>(y, x) = clipped.at<float>(y, x);
            } else if (target_mask.at<std::uint8_t>(y, x) == 0U) {
                correction.at<float>(y, x) = 0.0f;
            } else {
                correction.at<float>(y, x) = std::clamp(
                    correction.at<float>(y, x), -max_abs, max_abs);
            }
        }
    }
    return correction;
}

cv::Mat fit_spatial_seam_residual(
    const cv::Mat& residual,
    const cv::Mat& seam_mask,
    const cv::Mat& target_mask,
    const float max_abs) {
    cv::Mat correction = cv::Mat::zeros(residual.size(), CV_32FC1);
    if (seam_mask.empty() || target_mask.empty()
        || cv::countNonZero(seam_mask) == 0 || cv::countNonZero(target_mask) == 0) {
        return correction;
    }
    std::vector<double> xs;
    std::vector<double> ys;
    std::vector<double> values;
    const std::size_t total = static_cast<std::size_t>(cv::countNonZero(seam_mask));
    const std::size_t step = std::max<std::size_t>(1U, total / 100000U);
    std::size_t ordinal = 0;
    for (int y = 0; y < residual.rows; ++y) {
        for (int x = 0; x < residual.cols; ++x) {
            if (seam_mask.at<std::uint8_t>(y, x) == 0U || (ordinal++ % step) != 0U) {
                continue;
            }
            const float value = residual.at<float>(y, x);
            if (!std::isfinite(value)) {
                continue;
            }
            xs.push_back(static_cast<double>(x));
            ys.push_back(static_cast<double>(y));
            values.push_back(static_cast<double>(std::clamp(value, -max_abs, max_abs)));
        }
    }
    if (values.size() < 64U) {
        return correction;
    }
    std::vector<float> value_f;
    value_f.reserve(values.size());
    for (const double value : values) {
        value_f.push_back(static_cast<float>(value));
    }
    std::vector<double> sorted_x = xs;
    std::vector<double> sorted_y = ys;
    std::nth_element(sorted_x.begin(), sorted_x.begin() + static_cast<std::ptrdiff_t>(sorted_x.size() / 2U), sorted_x.end());
    std::nth_element(sorted_y.begin(), sorted_y.begin() + static_cast<std::ptrdiff_t>(sorted_y.size() / 2U), sorted_y.end());
    const double center_x = sorted_x[sorted_x.size() / 2U];
    const double center_y = sorted_y[sorted_y.size() / 2U];
    const double coordinate_scale = std::max(
        static_cast<double>(std::max(residual.rows, residual.cols)) * 0.25, 32.0);
    std::vector<std::uint8_t> keep(values.size(), 1U);
    std::array<double, 3> coeff = {static_cast<double>(median_value(value_f)), 0.0, 0.0};
    for (int iteration = 0; iteration < 5; ++iteration) {
        cv::Mat normal = cv::Mat::zeros(3, 3, CV_64F);
        cv::Mat rhs = cv::Mat::zeros(3, 1, CV_64F);
        std::size_t kept = 0;
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (keep[i] == 0U) {
                continue;
            }
            const double basis[3] = {
                1.0, (xs[i] - center_x) / coordinate_scale,
                (ys[i] - center_y) / coordinate_scale};
            for (int row = 0; row < 3; ++row) {
                rhs.at<double>(row, 0) += basis[row] * values[i];
                for (int column = 0; column < 3; ++column) {
                    normal.at<double>(row, column) += basis[row] * basis[column];
                }
            }
            ++kept;
        }
        cv::Mat solution;
        if (kept < 32U || !cv::solve(normal, rhs, solution, cv::DECOMP_SVD)
            || solution.rows != 3) {
            break;
        }
        for (int index = 0; index < 3; ++index) {
            coeff[static_cast<std::size_t>(index)] = solution.at<double>(index, 0);
        }
        std::vector<float> residuals;
        residuals.reserve(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) {
            const double fit = coeff[0]
                + coeff[1] * (xs[i] - center_x) / coordinate_scale
                + coeff[2] * (ys[i] - center_y) / coordinate_scale;
            residuals.push_back(static_cast<float>(values[i] - fit));
        }
        const float residual_center = median_value(residuals);
        std::vector<float> deviations;
        deviations.reserve(residuals.size());
        for (const float value : residuals) {
            deviations.push_back(std::abs(value - residual_center));
        }
        const float limit = 3.0f * 1.4826f * std::max(median_value(deviations), 1e-6f);
        for (std::size_t i = 0; i < residuals.size(); ++i) {
            keep[i] = std::abs(residuals[i] - residual_center) <= limit ? 1U : 0U;
        }
    }
    for (int y = 0; y < residual.rows; ++y) {
        for (int x = 0; x < residual.cols; ++x) {
            if (target_mask.at<std::uint8_t>(y, x) == 0U) {
                continue;
            }
            const double field = coeff[0]
                + coeff[1] * (static_cast<double>(x) - center_x) / coordinate_scale
                + coeff[2] * (static_cast<double>(y) - center_y) / coordinate_scale;
            correction.at<float>(y, x) = static_cast<float>(std::clamp(field, -static_cast<double>(max_abs), static_cast<double>(max_abs)));
        }
    }
    return correction;
}

cv::Mat anchor_depth_continuity(
    const cv::Mat& model_depth,
    const cv::Mat& canvas_depth,
    const cv::Mat& canvas_valid,
    const cv::Mat& apply_mask,
    const cv::Mat& anchor_ring) {
    cv::Mat result = model_depth.clone();
    cv::Mat delta = cv::Mat::zeros(model_depth.size(), CV_32FC1);
    if (apply_mask.empty() || cv::countNonZero(apply_mask) == 0
        || cv::countNonZero(anchor_ring) < 64) {
        return result;
    }

    cv::Mat old_valid;
    cv::bitwise_and(canvas_valid, finite_mask(canvas_depth), old_valid);
    cv::Mat anchor;
    cv::bitwise_and(anchor_ring, old_valid, anchor);
    if (cv::countNonZero(anchor) < 64) {
        return result;
    }

    // Keep the same global fallback field as Python's current
    // _anchor_depth_continuity.  It is intentionally only a fallback: a
    // single polynomial across a partial border ring can bend the newly
    // exposed side onto another depth layer, so the local field below has
    // priority wherever it has support.
    const int height = canvas_depth.rows;
    const int width = canvas_depth.cols;
    std::vector<cv::Point> anchor_points;
    std::vector<float> anchor_x;
    std::vector<float> anchor_y;
    anchor_points.reserve(static_cast<std::size_t>(cv::countNonZero(anchor)));
    anchor_x.reserve(anchor_points.capacity());
    anchor_y.reserve(anchor_points.capacity());
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (anchor.at<std::uint8_t>(y, x) != 0U) {
                anchor_points.emplace_back(x, y);
                anchor_x.push_back(static_cast<float>(x));
                anchor_y.push_back(static_cast<float>(y));
            }
        }
    }
    const float center_x = median_value(anchor_x);
    const float center_y = median_value(anchor_y);
    const double coordinate_scale = std::max(
        static_cast<double>(std::max(height, width)) * 0.25, 32.0);

    std::vector<std::uint8_t> keep(anchor_points.size(), 1U);
    std::array<double, 6> coefficients{};
    for (int iteration = 0; iteration < 4; ++iteration) {
        std::size_t kept_count = 0;
        for (const std::uint8_t value : keep) {
            kept_count += value != 0U ? 1U : 0U;
        }
        if (kept_count < 32U) {
            break;
        }
        cv::Mat fit_design(static_cast<int>(kept_count), 6, CV_64F);
        cv::Mat fit_values(static_cast<int>(kept_count), 1, CV_64F);
        int row = 0;
        for (std::size_t index = 0; index < anchor_points.size(); ++index) {
            if (keep[index] == 0U) {
                continue;
            }
            const double xn = (static_cast<double>(anchor_points[index].x) - center_x)
                / coordinate_scale;
            const double yn = (static_cast<double>(anchor_points[index].y) - center_y)
                / coordinate_scale;
            fit_design.at<double>(row, 0) = 1.0;
            fit_design.at<double>(row, 1) = xn;
            fit_design.at<double>(row, 2) = yn;
            fit_design.at<double>(row, 3) = xn * xn;
            fit_design.at<double>(row, 4) = xn * yn;
            fit_design.at<double>(row, 5) = yn * yn;
            fit_values.at<double>(row, 0) = static_cast<double>(
                canvas_depth.at<float>(anchor_points[index].y, anchor_points[index].x));
            ++row;
        }
        cv::Mat solution;
        if (!cv::solve(fit_design, fit_values, solution, cv::DECOMP_SVD)
            || solution.rows != 6) {
            return result;
        }
        for (int index = 0; index < 6; ++index) {
            coefficients[static_cast<std::size_t>(index)] = solution.at<double>(index, 0);
        }

        std::vector<float> residuals;
        residuals.reserve(anchor_points.size());
        for (const cv::Point& point : anchor_points) {
            const double xn = (static_cast<double>(point.x) - center_x) / coordinate_scale;
            const double yn = (static_cast<double>(point.y) - center_y) / coordinate_scale;
            const double fitted = coefficients[0]
                + coefficients[1] * xn
                + coefficients[2] * yn
                + coefficients[3] * xn * xn
                + coefficients[4] * xn * yn
                + coefficients[5] * yn * yn;
            residuals.push_back(static_cast<float>(
                static_cast<double>(canvas_depth.at<float>(point.y, point.x)) - fitted));
        }
        const float center = median_value(residuals);
        std::vector<float> deviations;
        deviations.reserve(residuals.size());
        for (const float value : residuals) {
            deviations.push_back(std::abs(value - center));
        }
        const float mad = std::max(median_value(deviations), 1e-6f);
        const float limit = 3.0f * 1.4826f * mad;
        for (std::size_t index = 0; index < residuals.size(); ++index) {
            keep[index] = std::abs(residuals[index] - center) <= limit ? 1U : 0U;
        }
    }

    cv::Mat fallback(height, width, CV_32FC1);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const double xn = (static_cast<double>(x) - center_x) / coordinate_scale;
            const double yn = (static_cast<double>(y) - center_y) / coordinate_scale;
            fallback.at<float>(y, x) = static_cast<float>(
                coefficients[0]
                + coefficients[1] * xn
                + coefficients[2] * yn
                + coefficients[3] * xn * xn
                + coefficients[4] * xn * yn
                + coefficients[5] * yn * yn);
        }
    }

    // This is the same normalized-convolution field used by Python's
    // _local_anchor_depth_field.  Notice that the boundary is measured from
    // newly exposed support (apply_mask & ~old_valid), not from all target
    // pixels; using the latter makes an old-overlap strip act like a second
    // boundary and is the source of the visible parallel band.
    cv::Mat old_f;
    old_valid.convertTo(old_f, CV_32FC1, 1.0 / 255.0);
    cv::Mat source = cv::Mat::zeros(canvas_depth.size(), CV_32FC1);
    canvas_depth.copyTo(source, old_valid);
    cv::Mat numerator;
    cv::Mat denominator;
    cv::GaussianBlur(source, numerator, cv::Size(), 6.0, 6.0);
    cv::GaussianBlur(old_f, denominator, cv::Size(), 6.0, 6.0);
    cv::Mat smooth;
    cv::divide(numerator, denominator + 1e-6f, smooth);

    cv::Mat not_old;
    cv::bitwise_not(old_valid, not_old);
    cv::Mat new_target;
    cv::bitwise_and(apply_mask, not_old, new_target);
    cv::Mat not_new_target;
    cv::bitwise_not(new_target, not_new_target);
    cv::Mat distance_to_target;
    cv::distanceTransform(not_new_target, distance_to_target, cv::DIST_L2, 3);
    cv::Mat boundary;
    cv::compare(distance_to_target, 2.5, boundary, cv::CMP_LE);
    cv::bitwise_and(boundary, old_valid, boundary);

    cv::Mat boundary_f;
    boundary.convertTo(boundary_f, CV_32FC1, 1.0 / 255.0);
    cv::Mat residual = canvas_depth - smooth;
    cv::Mat boundary_residual;
    cv::multiply(residual, boundary_f, boundary_residual);
    cv::Mat boundary_numerator;
    cv::Mat boundary_denominator;
    const double boundary_sigma = 3.0;
    cv::GaussianBlur(boundary_residual, boundary_numerator, cv::Size(), boundary_sigma, boundary_sigma);
    cv::GaussianBlur(boundary_f, boundary_denominator, cv::Size(), boundary_sigma, boundary_sigma);
    cv::Mat boundary_correction;
    cv::divide(boundary_numerator, boundary_denominator + 1e-6f, boundary_correction);
    cv::Mat local = smooth + boundary_correction;

    cv::Mat gaussian_support;
    cv::compare(denominator, 0.02, gaussian_support, cv::CMP_GT);
    cv::Mat boundary_support;
    cv::compare(boundary_denominator, 1e-4, boundary_support, cv::CMP_GT);
    cv::Mat local_support;
    cv::bitwise_or(gaussian_support, boundary_support, local_support);
    cv::Mat nearest = nearest_fill_values(canvas_depth, old_valid);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (local_support.at<std::uint8_t>(y, x) == 0U) {
                local.at<float>(y, x) = nearest.at<float>(y, x);
            }
        }
    }

    cv::Mat finite_continuation = finite_mask(local);
    cv::Mat finite_model = finite_mask(result);
    cv::Mat new_only;
    cv::bitwise_and(apply_mask, not_old, new_only);
    cv::bitwise_and(new_only, finite_continuation, new_only);
    cv::bitwise_and(new_only, finite_model, new_only);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (new_only.at<std::uint8_t>(y, x) != 0U) {
                const float before = result.at<float>(y, x);
                result.at<float>(y, x) = local.at<float>(y, x);
                delta.at<float>(y, x) = result.at<float>(y, x) - before;
            }
        }
    }

    cv::Mat finite_canvas = finite_mask(canvas_depth);
    cv::Mat old_overlap;
    cv::bitwise_and(apply_mask, old_valid, old_overlap);
    cv::bitwise_and(old_overlap, finite_canvas, old_overlap);
    cv::bitwise_and(old_overlap, finite_model, old_overlap);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (old_overlap.at<std::uint8_t>(y, x) != 0U) {
                const float before = result.at<float>(y, x);
                result.at<float>(y, x) = canvas_depth.at<float>(y, x);
                delta.at<float>(y, x) = result.at<float>(y, x) - before;
            }
        }
    }
    return result;
}

cv::Mat anchor_texture_transfer(
    const cv::Mat& current_rgb,
    const cv::Mat& canvas_rgb,
    const cv::Mat& canvas_valid,
    const cv::Mat& current_valid,
    const cv::Mat& apply_mask,
    const cv::Mat& support_change,
    const cv::Mat& anchor_ring) {
    cv::Mat result = current_rgb.clone();
    if (apply_mask.empty() || cv::countNonZero(apply_mask) == 0
        || cv::countNonZero(canvas_valid) == 0 || cv::countNonZero(anchor_ring) == 0) {
        return result;
    }

    // Canvas RGB values are bounded when they enter CanvasState; the state
    // validity mask is therefore the authoritative RGB support mask here.
    const cv::Mat old_valid = canvas_valid.clone();
    const cv::Mat new_valid = current_valid;
    cv::Mat old_f;
    cv::Mat new_f;
    old_valid.convertTo(old_f, CV_32FC1, 1.0 / 255.0);
    new_valid.convertTo(new_f, CV_32FC1, 1.0 / 255.0);
    const double sigma = 24.0;
    cv::Mat old_den;
    cv::Mat new_den;
    cv::GaussianBlur(old_f, old_den, cv::Size(), sigma, sigma);
    cv::GaussianBlur(new_f, new_den, cv::Size(), sigma, sigma);

    std::vector<cv::Mat> old_channels;
    std::vector<cv::Mat> new_channels;
    cv::split(canvas_rgb, old_channels);
    cv::split(current_rgb, new_channels);
    std::vector<cv::Mat> old_fields(3);
    std::vector<cv::Mat> new_fields(3);
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
        cv::divide(old_num, safe_old_den, old_fields[channel]);
        cv::divide(new_num, safe_new_den, new_fields[channel]);
    }

    cv::Mat stable;
    cv::bitwise_and(anchor_ring, old_valid, stable);
    cv::bitwise_and(stable, new_valid, stable);
    if (cv::countNonZero(stable) < 128) {
        cv::Mat not_support;
        cv::bitwise_not(support_change, not_support);
        cv::bitwise_and(old_valid, new_valid, stable);
        cv::bitwise_and(stable, not_support, stable);
    }
    if (cv::countNonZero(stable) < 128) {
        return result;
    }

    std::vector<cv::Mat> correction_channels(3, cv::Mat::zeros(current_rgb.size(), CV_32FC1));
    for (int channel = 0; channel < 3; ++channel) {
        const cv::Mat residual = old_fields[channel] - new_fields[channel];
        std::vector<float> values;
        values.reserve(static_cast<std::size_t>(cv::countNonZero(stable)));
        for (int y = 0; y < residual.rows; ++y) {
            for (int x = 0; x < residual.cols; ++x) {
                if (stable.at<std::uint8_t>(y, x) != 0U) {
                    values.push_back(residual.at<float>(y, x));
                }
            }
        }
        const float center = median_value(values);
        std::vector<float> deviations;
        deviations.reserve(values.size());
        for (const float value : values) {
            deviations.push_back(std::abs(value - center));
        }
        const float mad = std::max(median_value(deviations), 1e-4f);
        const float limit = std::max(0.04f, 3.0f * 1.4826f * mad);
        cv::Mat clipped = cv::Mat::zeros(residual.size(), CV_32FC1);
        for (int y = 0; y < residual.rows; ++y) {
            for (int x = 0; x < residual.cols; ++x) {
                if (stable.at<std::uint8_t>(y, x) == 0U) {
                    continue;
                }
                clipped.at<float>(y, x) = std::clamp(
                    residual.at<float>(y, x), center - limit, center + limit);
            }
        }
        cv::Mat stable_f;
        stable.convertTo(stable_f, CV_32FC1, 1.0 / 255.0);
        cv::Mat numerator;
        cv::Mat denominator;
        cv::Mat product;
        cv::multiply(clipped, stable_f, product);
        cv::GaussianBlur(product, numerator, cv::Size(), sigma, sigma);
        cv::GaussianBlur(stable_f, denominator, cv::Size(), sigma, sigma);
        cv::Mat safe_denominator;
        cv::add(denominator, cv::Scalar(1e-5), safe_denominator);
        cv::divide(numerator, safe_denominator, correction_channels[channel]);
        for (int y = 0; y < correction_channels[channel].rows; ++y) {
            for (int x = 0; x < correction_channels[channel].cols; ++x) {
                correction_channels[channel].at<float>(y, x) = std::clamp(
                    correction_channels[channel].at<float>(y, x), -0.05f, 0.05f);
            }
        }
    }

    cv::Mat correction;
    cv::merge(correction_channels, correction);
    cv::Mat transferred = current_rgb.clone();
    for (int y = 0; y < transferred.rows; ++y) {
        for (int x = 0; x < transferred.cols; ++x) {
            if (apply_mask.at<std::uint8_t>(y, x) == 0U) {
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
    cv::Mat support_overlap;
    cv::bitwise_and(apply_mask, support_change, support_overlap);
    cv::bitwise_and(support_overlap, old_valid, support_overlap);
    cv::bitwise_and(support_overlap, new_valid, support_overlap);
    for (int y = 0; y < transferred.rows; ++y) {
        for (int x = 0; x < transferred.cols; ++x) {
            if (support_overlap.at<std::uint8_t>(y, x) == 0U) {
                continue;
            }
            const float old_mix = std::clamp(
                1.0f - distance_to_ring.at<float>(y, x) / 16.0f, 0.65f, 0.95f);
            const cv::Vec3f old_color = canvas_rgb.at<cv::Vec3f>(y, x);
            const cv::Vec3f new_color = transferred.at<cv::Vec3f>(y, x);
            transferred.at<cv::Vec3f>(y, x) =
                new_color * (1.0f - old_mix) + old_color * old_mix;
        }
    }

    for (int y = 0; y < result.rows; ++y) {
        for (int x = 0; x < result.cols; ++x) {
            if (apply_mask.at<std::uint8_t>(y, x) != 0U
                && new_valid.at<std::uint8_t>(y, x) != 0U) {
                result.at<cv::Vec3f>(y, x) = transferred.at<cv::Vec3f>(y, x);
            }
        }
    }
    return result;
}

}  // namespace

std::string InferenceMetrics::csv_line() const {
    std::ostringstream line;
    line << std::fixed << std::setprecision(3)
         << frame_seq << ',' << image << ',' << total_ms << ',' << read_ms << ',' << align2d_ms << ',' << diff_ms
         << ',' << model_ms << ',' << depth_align_ms << ',' << patch_ms << ',' << changed_ratio << ','
         << changed_point_count << ',' << valid_point_count << ',' << homography_inliers << ','
         << homography_error_px << ',' << roi_width << ',' << roi_height << ','
         << model_input_width << ',' << model_input_height << ','
         << photometric_changed_ratio << ',' << support_changed_ratio << ','
         << (skipped_model ? "yes" : "no") << ','
         << (fallback.empty() ? "None" : fallback) << ','
         << group_size << ',' << group_stride << ',' << group_anchor_index << ','
         << forward_calls << ',' << forward_batch_size << ',' << forward_sequence_size << ','
         << group_fused_sources << ',' << group_rejected_sources << ','
         << group_max_depth_residual;
    return line.str();
}

InferenceEngine::InferenceEngine(InferenceOptions options) : options_(std::move(options)) {
    if (options_.group_mode) {
        if (options_.group_model.empty()) {
            throw std::invalid_argument("group mode requires a B=3,S=1 TorchScript model");
        }
        if (options_.group_width <= 0 || options_.group_height <= 0
            || options_.group_width % 14 != 0 || options_.group_height % 14 != 0) {
            throw std::invalid_argument("group model dimensions must be positive multiples of 14");
        }
    } else if (options_.model.empty()) {
        throw std::invalid_argument("inference model path is empty");
    }
    if (options_.width <= 0 || options_.height <= 0) {
        throw std::invalid_argument("inference dimensions must be positive");
    }
    if (options_.canvas_width <= 0) {
        options_.canvas_width = options_.width;
    }
    if (options_.canvas_height <= 0) {
        options_.canvas_height = options_.height;
    }
    if (options_.first_model_width <= 0) {
        options_.first_model_width = options_.width;
    }
    if (options_.first_model_height <= 0) {
        options_.first_model_height = options_.height;
    }
    device_ = parse_device(options_.device);
    dtype_ = parse_dtype(options_.dtype);
    if (!options_.model.empty()) {
        module_ = torch::jit::load(options_.model, device_);
        module_.eval();
        configure_torchscript_executor(module_);
    }

    if (options_.group_mode) {
        group_module_ = torch::jit::load(options_.group_model, device_);
        group_module_.eval();
        configure_torchscript_executor(group_module_);
        has_group_module_ = true;
    }

    pair_model_dir_ = options_.pair_model_dir;
    dynamic_pair_shapes_ = discover_dynamic_pair_shapes(pair_model_dir_);
    if (!options_.group_mode && !options_.pair_model.empty()) {
        // Loading every possible dynamic ROI graph at startup would keep
        // several multi-gigabyte copies on the GPU.  With a bucket directory,
        // activate only the shapes requested by the stream and retain a small
        // LRU of already uploaded graphs.
        if (pair_model_dir_.empty() || options_.pair_letterbox) {
            pair_module_ = torch::jit::load(options_.pair_model, device_);
            pair_module_.eval();
            configure_torchscript_executor(pair_module_);
            pair_module_loaded_ = true;
        }
        has_pair_module_ = true;
    }

    // Move the fixed-shape compilation cost to startup.  The first frame and
    // later frames intentionally use the same one-frame/two-frame split as
    // Python's _single_frame_batch/_two_frame_batch.
    if (device_.is_cuda()) {
        if (options_.group_mode) {
            const cv::Mat group_warmup = cv::Mat::zeros(
                options_.group_height, options_.group_width, CV_8UC3);
            (void)run_group_model({group_warmup, group_warmup, group_warmup});
            return;
        }
        const cv::Mat first_warmup = cv::Mat::zeros(
            options_.first_model_height, options_.first_model_width, CV_8UC3);
        (void)run_model({first_warmup}, 0);
        if (has_pair_module_ && (pair_model_dir_.empty() || options_.pair_letterbox)) {
            const cv::Mat pair_warmup = cv::Mat::zeros(options_.height, options_.width, CV_8UC3);
            (void)run_model({pair_warmup, pair_warmup}, 1);
        } else if (has_pair_module_ && !pair_model_dir_.empty()
            && dynamic_pair_shapes_.size() == 1U) {
            // A single enclosing bucket is the practical low-memory deployment
            // mode: warm it before the stream starts, then release the
            // single-frame graph after frame 0. The common 490x700/518x700
            // replay therefore pays no graph-load cost inside frame timings.
            const auto shape = dynamic_pair_shapes_.front();
            const std::filesystem::path bucket_path = pair_artifact_for_shape(
                options_.pair_model, pair_model_dir_, shape.first, shape.second);
            activate_dynamic_pair_module(bucket_path, shape);
            const cv::Mat pair_warmup = cv::Mat::zeros(shape.second, shape.first, CV_8UC3);
            (void)run_model({pair_warmup, pair_warmup}, 1);
        }
    }
}

std::pair<int, int> InferenceEngine::dynamic_pair_shape_for_target(
    const std::pair<int, int>& target) const {
    if (pair_model_dir_.empty() || options_.pair_letterbox) {
        return target;
    }
    const auto fits = [&target](const std::pair<int, int>& shape) {
        return shape.first >= target.first && shape.second >= target.second;
    };
    // Prefer a resident bucket so a later ROI does not trigger a graph swap.
    for (auto it = pair_module_lru_.rbegin(); it != pair_module_lru_.rend(); ++it) {
        if (fits(*it)) {
            return *it;
        }
    }
    std::pair<int, int> selected = target;
    std::int64_t selected_area = std::numeric_limits<std::int64_t>::max();
    for (const auto& shape : dynamic_pair_shapes_) {
        if (!fits(shape)) {
            continue;
        }
        const std::int64_t area = static_cast<std::int64_t>(shape.first)
            * static_cast<std::int64_t>(shape.second);
        if (area < selected_area) {
            selected = shape;
            selected_area = area;
        }
    }
    return selected;
}

void InferenceEngine::activate_dynamic_pair_module(
    const std::filesystem::path& bucket_path,
    const std::pair<int, int>& shape) {
    const auto cached = pair_module_cache_.find(shape);
    if (cached == pair_module_cache_.end()) {
        // Release the least-recently-used GPU graph before loading a new one;
        // otherwise the old graph and the new graph coexist during CUDA
        // deserialization and can exceed the 8 GB deployment budget.
        while (pair_module_cache_.size() >= kMaxDynamicPairGpuCache
            && !pair_module_lru_.empty()) {
            const std::pair<int, int> victim = pair_module_lru_.front();
            pair_module_lru_.erase(pair_module_lru_.begin());
            if (pair_module_loaded_ && pair_module_shape_ == victim) {
                pair_module_ = torch::jit::script::Module();
                pair_module_loaded_ = false;
                pair_module_shape_ = {-1, -1};
            }
            const auto victim_it = pair_module_cache_.find(victim);
            if (victim_it != pair_module_cache_.end()) {
                victim_it->second.to(torch::Device(torch::kCPU));
                pair_module_cache_.erase(victim_it);
            }
        }

        auto loaded = torch::jit::load(bucket_path.string(), device_);
        loaded.eval();
        configure_torchscript_executor(loaded);
        pair_module_cache_.emplace(shape, std::move(loaded));
    }

    pair_module_ = pair_module_cache_.at(shape);
    pair_module_shape_ = shape;
    pair_module_loaded_ = true;

    const auto lru_it = std::find(pair_module_lru_.begin(), pair_module_lru_.end(), shape);
    if (lru_it != pair_module_lru_.end()) {
        pair_module_lru_.erase(lru_it);
    }
    pair_module_lru_.push_back(shape);
}

void InferenceEngine::release_single_model_after_first_frame() {
    if (options_.group_mode || !device_.is_cuda() || single_model_released_ || options_.model.empty()) {
        return;
    }
    // The single-frame graph is used only for the initial frame. Moving it to
    // CPU after that call frees enough VRAM for the second dynamic pair
    // bucket to remain resident instead of forcing a reload/paging cycle.
    module_.to(torch::Device(torch::kCPU));
    module_ = torch::jit::script::Module();
    single_model_released_ = true;
}

InferenceEngine::FrameImage InferenceEngine::load_frame(const std::filesystem::path& path) const {
    FrameImage frame;
    frame.path = path;
    const cv::Mat original = read_rgb(path);
    const double scale_x = static_cast<double>(options_.width) / static_cast<double>(original.cols);
    // Python's matching canvas preserves the aspect ratio and only rounds the
    // matching image to the nearest pixel.  The model bucket is rounded later;
    // rounding this image to a patch multiple was the source of a different
    // image geometry and of the clipped top edge in the C++ path.
    const int resized_height = std::max(
        1, static_cast<int>(std::round(static_cast<double>(original.rows) * scale_x)));
    cv::Mat original_f;
    original.convertTo(original_f, CV_32FC3, 1.0 / 255.0);
    cv::resize(
        original_f,
        frame.match_rgb_f,
        cv::Size(options_.width, resized_height),
        0.0,
        0.0,
        cv::INTER_AREA);
    // Python's model/matcher path uses ``astype(np.uint8)`` (floor), while
    // cv::Mat::convertTo rounds. The one-byte difference changes SIFT
    // keypoint selection at the moving boundary and therefore changes the
    // ROI support.
    frame.match_rgb_u8 = quantize_rgb_u8(frame.match_rgb_f);

    frame.rgb_f = cv::Mat::zeros(options_.canvas_height, options_.canvas_width, CV_32FC3);
    frame.rgb_u8 = cv::Mat::zeros(options_.canvas_height, options_.canvas_width, CV_8UC3);
    const int pad_left = std::max(32, static_cast<int>(std::round(options_.width * 0.05)));
    const int pad_top = std::max(128, static_cast<int>(std::round(options_.width * 0.18)));
    const int copy_x = pad_left;
    const int copy_y = pad_top;
    if (copy_x >= 0 && copy_y >= 0
        && copy_x + frame.match_rgb_u8.cols <= frame.rgb_u8.cols
        && copy_y + frame.match_rgb_u8.rows <= frame.rgb_u8.rows) {
        frame.match_rgb_u8.copyTo(frame.rgb_u8(cv::Rect(
            copy_x, copy_y, frame.match_rgb_u8.cols, frame.match_rgb_u8.rows)));
        frame.match_rgb_f.copyTo(frame.rgb_f(cv::Rect(
            copy_x, copy_y, frame.match_rgb_f.cols, frame.match_rgb_f.rows)));
    } else {
        // Keep the original fixed-canvas behavior for callers that do not
        // request Python's padded canvas.  The live replay launcher supplies
        // a large enough canvas and takes the branch above.
        const int copy_width = std::min(frame.match_rgb_u8.cols, frame.rgb_u8.cols);
        const int copy_height = std::min(frame.match_rgb_u8.rows, frame.rgb_u8.rows);
        const int source_x = std::max(0, (frame.match_rgb_u8.cols - copy_width) / 2);
        const int source_y = std::max(0, (frame.match_rgb_u8.rows - copy_height) / 2);
        const int target_x = std::max(0, (frame.rgb_u8.cols - copy_width) / 2);
        const int target_y = std::max(0, (frame.rgb_u8.rows - copy_height) / 2);
        frame.match_rgb_u8(
            cv::Rect(source_x, source_y, copy_width, copy_height)).copyTo(
                frame.rgb_u8(cv::Rect(target_x, target_y, copy_width, copy_height)));
        frame.match_rgb_f(
            cv::Rect(source_x, source_y, copy_width, copy_height)).copyTo(
                frame.rgb_f(cv::Rect(target_x, target_y, copy_width, copy_height)));
    }
    frame.support = foreground_mask(frame.rgb_f);
    return frame;
}

InferenceEngine::Prediction InferenceEngine::run_model(
    const std::vector<cv::Mat>& rgb_u8,
    const int output_frame_index) {
    if (rgb_u8.empty()) {
        throw std::runtime_error("cannot run OmniVGGT on an empty image batch");
    }
    const int frame_count = static_cast<int>(rgb_u8.size());
    const int height = rgb_u8.front().rows;
    const int width = rgb_u8.front().cols;
    for (const cv::Mat& image : rgb_u8) {
        if (image.empty() || image.rows != height || image.cols != width
            || (image.type() != CV_8UC3 && image.type() != CV_32FC3)) {
            throw std::runtime_error(
                "OmniVGGT image batch entries must have one matching CV_8UC3/CV_32FC3 shape");
        }
    }
    // InferenceMode is the C++ equivalent of Python's inference_mode().
    // It also removes version-counter/view tracking that NoGradGuard leaves
    // enabled.  This matters for the large transformer graph.
    c10::InferenceMode inference_mode;
    // Keep the artifact loadable as the compact FP32 TorchScript file while
    // executing CUDA kernels in the same BF16 autocast regime as Python.
    // This avoids materializing the full FP32 activation graph on an 8 GB
    // GPU.  The guard is a no-op for an explicitly requested float32 run.
    CudaAutocastGuard autocast(dtype_);
    const torch::Tensor images = make_image_tensor_batch(rgb_u8).to(device_, dtype_);
    // Match stream_omnivggt's _single_frame_batch/_two_frame_batch: camera
    // tensors stay FP32 and use the identity camera, while image/depth/mask
    // tensors use the selected CUDA input dtype.  Keeping these tensors in
    // BF16 (and filling them with zeros) changes the model path and is not
    // equivalent to the Python backend.
    const auto float_options = torch::TensorOptions().device(device_).dtype(torch::kFloat32);
    const torch::Tensor extrinsics = torch::eye(4, float_options)
        .slice(0, 0, 3)
        .reshape({1, 1, 3, 4})
        .repeat({1, frame_count, 1, 1});
    const torch::Tensor intrinsics = torch::eye(3, float_options)
        .reshape({1, 1, 3, 3})
        .repeat({1, frame_count, 1, 1});
    const torch::Tensor depth_input = torch::zeros(
        {1, frame_count, height, width, 1}, torch::TensorOptions().dtype(torch::kFloat32)).to(device_, dtype_);
    const torch::Tensor mask = torch::zeros(
        {1, frame_count, height, width}, torch::TensorOptions().dtype(torch::kFloat32)).to(device_, dtype_);

    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(images);
    inputs.emplace_back(extrinsics);
    inputs.emplace_back(intrinsics);
    inputs.emplace_back(depth_input);
    inputs.emplace_back(mask);
    torch::jit::script::Module* module = &module_;
    if (frame_count == 2 && has_pair_module_) {
        if (!pair_model_dir_.empty() && !options_.pair_letterbox) {
            const std::filesystem::path bucket_path = pair_artifact_for_shape(
                options_.pair_model, pair_model_dir_, width, height);
            if (!std::filesystem::exists(bucket_path)) {
                throw std::runtime_error(
                    "missing TorchScript pair bucket for input "
                    + std::to_string(width) + "x" + std::to_string(height)
                    + ": " + bucket_path.string());
            }
            if (!pair_module_loaded_ || pair_module_shape_ != std::pair<int, int>{width, height}) {
                activate_dynamic_pair_module(
                    bucket_path,
                    {width, height});
            }
        }
        module = &pair_module_;
    }
    if (frame_count == 2 && !has_pair_module_) {
        throw std::runtime_error(
            "a two-frame stream update requires --model-pair with a num_images=2 TorchScript artifact");
    }
    const auto output_tuple = module->forward(inputs).toTuple();
    if (output_tuple->elements().size() < 3U) {
        throw std::runtime_error(
            "TorchScript observer model must return pose, depth and depth confidence");
    }

    const torch::Tensor pose = to_cpu_float(output_tuple->elements()[0].toTensor());
    const torch::Tensor depth = to_cpu_float(output_tuple->elements()[1].toTensor());
    const torch::Tensor confidence = to_cpu_float(output_tuple->elements()[2].toTensor());
    const bool has_world_points = output_tuple->elements().size() >= 5U;
    torch::Tensor world_points;
    torch::Tensor world_points_confidence;
    if (has_world_points) {
        world_points = to_cpu_float(output_tuple->elements()[3].toTensor());
        world_points_confidence = to_cpu_float(output_tuple->elements()[4].toTensor());
    }
    if (depth.dim() != 5 || confidence.dim() != 4
        || (has_world_points && (world_points.dim() != 5 || world_points_confidence.dim() != 4))) {
        throw std::runtime_error("unexpected complete OmniVGGT output dimensions");
    }
    if (depth.size(2) != height || depth.size(3) != width
        || confidence.size(2) != height || confidence.size(3) != width
        || (has_world_points && (world_points.size(2) != height || world_points.size(3) != width
            || world_points_confidence.size(2) != height
            || world_points_confidence.size(3) != width))) {
        throw std::runtime_error(
            "TorchScript output shape does not match the requested ROI input "
            + std::to_string(width) + "x" + std::to_string(height));
    }
    const int frame_index = std::clamp(output_frame_index, 0, frame_count - 1);
    if (depth.size(1) <= frame_index || confidence.size(1) <= frame_index
        || (has_world_points && (world_points.size(1) <= frame_index
            || world_points_confidence.size(1) <= frame_index))) {
        throw std::runtime_error("TorchScript output does not contain the requested frame index");
    }
    const int output_height = static_cast<int>(depth.size(2));
    const int output_width = static_cast<int>(depth.size(3));

    Prediction prediction;
    prediction.depth = cv::Mat(output_height, output_width, CV_32FC1);
    prediction.confidence = cv::Mat(output_height, output_width, CV_32FC1);
    if (has_world_points) {
        prediction.world_points = cv::Mat(output_height, output_width, CV_32FC3);
        prediction.world_points_confidence = cv::Mat(output_height, output_width, CV_32FC1);
    }
    const auto depth_access = depth.accessor<float, 5>();
    const auto confidence_access = confidence.accessor<float, 4>();
    if (has_world_points) {
        const auto world_points_access = world_points.accessor<float, 5>();
        const auto world_points_confidence_access = world_points_confidence.accessor<float, 4>();
        for (int y = 0; y < output_height; ++y) {
            for (int x = 0; x < output_width; ++x) {
                prediction.depth.at<float>(y, x) = depth_access[0][frame_index][y][x][0];
                prediction.confidence.at<float>(y, x) = confidence_access[0][frame_index][y][x];
                prediction.world_points.at<cv::Vec3f>(y, x) = cv::Vec3f(
                    world_points_access[0][frame_index][y][x][0],
                    world_points_access[0][frame_index][y][x][1],
                    world_points_access[0][frame_index][y][x][2]);
                prediction.world_points_confidence.at<float>(y, x) =
                    world_points_confidence_access[0][frame_index][y][x];
            }
        }
    } else {
        for (int y = 0; y < output_height; ++y) {
            for (int x = 0; x < output_width; ++x) {
                prediction.depth.at<float>(y, x) = depth_access[0][frame_index][y][x][0];
                prediction.confidence.at<float>(y, x) = confidence_access[0][frame_index][y][x];
            }
        }
    }

    if (pose.numel() >= 9) {
        const auto pose_access = pose.accessor<float, 3>();
        prediction.fov_h = pose_access[0][frame_index][7];
        prediction.fov_w = pose_access[0][frame_index][8];
    }
    if (!std::isfinite(prediction.fov_h) || prediction.fov_h <= 0.01f || prediction.fov_h >= 3.1f) {
        prediction.fov_h = 1.2f;
    }
    if (!std::isfinite(prediction.fov_w) || prediction.fov_w <= 0.01f || prediction.fov_w >= 3.1f) {
        prediction.fov_w = 1.2f;
    }
    return prediction;
}

std::vector<InferenceEngine::Prediction> InferenceEngine::run_group_model(
    const std::vector<cv::Mat>& rgb_u8) {
    if (!has_group_module_ || rgb_u8.size() != 3U) {
        throw std::runtime_error("B=3,S=1 group inference requires exactly three images");
    }
    const int height = rgb_u8.front().rows;
    const int width = rgb_u8.front().cols;
    for (const cv::Mat& image : rgb_u8) {
        if (image.empty() || image.rows != height || image.cols != width
            || (image.type() != CV_8UC3 && image.type() != CV_32FC3)) {
            throw std::runtime_error("group image batch entries must share one CV_8UC3/CV_32FC3 shape");
        }
    }
    c10::InferenceMode inference_mode;
    CudaAutocastGuard autocast(dtype_);
    const torch::Tensor images = make_image_tensor_independent_batch(rgb_u8).to(device_, dtype_);
    const auto float_options = torch::TensorOptions().device(device_).dtype(torch::kFloat32);
    const torch::Tensor extrinsics = torch::eye(4, float_options)
        .slice(0, 0, 3)
        .reshape({1, 1, 3, 4})
        .repeat({3, 1, 1, 1});
    const torch::Tensor intrinsics = torch::eye(3, float_options)
        .reshape({1, 1, 3, 3})
        .repeat({3, 1, 1, 1});
    const torch::Tensor depth_input = torch::zeros(
        {3, 1, height, width, 1}, torch::TensorOptions().dtype(torch::kFloat32)).to(device_, dtype_);
    const torch::Tensor mask = torch::zeros(
        {3, 1, height, width}, torch::TensorOptions().dtype(torch::kFloat32)).to(device_, dtype_);

    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(images);
    inputs.emplace_back(extrinsics);
    inputs.emplace_back(intrinsics);
    inputs.emplace_back(depth_input);
    inputs.emplace_back(mask);
    const auto output_tuple = group_module_.forward(inputs).toTuple();
    if (output_tuple->elements().size() < 3U) {
        throw std::runtime_error("group TorchScript model must return pose, depth and confidence");
    }
    const torch::Tensor pose = to_cpu_float(output_tuple->elements()[0].toTensor());
    const torch::Tensor depth = to_cpu_float(output_tuple->elements()[1].toTensor());
    const torch::Tensor confidence = to_cpu_float(output_tuple->elements()[2].toTensor());
    const bool has_world_points = output_tuple->elements().size() >= 5U;
    torch::Tensor world_points;
    torch::Tensor world_points_confidence;
    if (has_world_points) {
        world_points = to_cpu_float(output_tuple->elements()[3].toTensor());
        world_points_confidence = to_cpu_float(output_tuple->elements()[4].toTensor());
    }
    if (depth.dim() != 5 || confidence.dim() != 4
        || depth.size(0) != 3 || depth.size(1) != 1
        || confidence.size(0) != 3 || confidence.size(1) != 1
        || depth.size(2) != height || depth.size(3) != width
        || confidence.size(2) != height || confidence.size(3) != width
        || (has_world_points && (world_points.dim() != 5 || world_points_confidence.dim() != 4
            || world_points.size(0) != 3 || world_points.size(1) != 1
            || world_points_confidence.size(0) != 3 || world_points_confidence.size(1) != 1
            || world_points.size(2) != height || world_points.size(3) != width
            || world_points_confidence.size(2) != height
            || world_points_confidence.size(3) != width))) {
        throw std::runtime_error("group TorchScript output must have shape [3,1,H,W,*]");
    }

    std::vector<Prediction> predictions;
    predictions.reserve(3U);
    const auto depth_access = depth.accessor<float, 5>();
    const auto confidence_access = confidence.accessor<float, 4>();
    for (int batch = 0; batch < 3; ++batch) {
        Prediction prediction;
        prediction.depth = cv::Mat(height, width, CV_32FC1);
        prediction.confidence = cv::Mat(height, width, CV_32FC1);
        if (has_world_points) {
            prediction.world_points = cv::Mat(height, width, CV_32FC3);
            prediction.world_points_confidence = cv::Mat(height, width, CV_32FC1);
        }
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                prediction.depth.at<float>(y, x) = depth_access[batch][0][y][x][0];
                prediction.confidence.at<float>(y, x) = confidence_access[batch][0][y][x];
                if (has_world_points) {
                    const auto points = world_points.accessor<float, 5>();
                    const auto point_conf = world_points_confidence.accessor<float, 4>();
                    prediction.world_points.at<cv::Vec3f>(y, x) = cv::Vec3f(
                        points[batch][0][y][x][0], points[batch][0][y][x][1], points[batch][0][y][x][2]);
                    prediction.world_points_confidence.at<float>(y, x) = point_conf[batch][0][y][x];
                }
            }
        }
        if (pose.dim() == 3 && pose.size(0) >= 3 && pose.size(1) >= 1 && pose.size(2) >= 9) {
            const auto pose_values = pose.accessor<float, 3>();
            prediction.fov_h = pose_values[batch][0][7];
            prediction.fov_w = pose_values[batch][0][8];
        }
        if (!std::isfinite(prediction.fov_h) || prediction.fov_h <= 0.01f || prediction.fov_h >= 3.1f) {
            prediction.fov_h = 1.2f;
        }
        if (!std::isfinite(prediction.fov_w) || prediction.fov_w <= 0.01f || prediction.fov_w >= 3.1f) {
            prediction.fov_w = 1.2f;
        }
        predictions.push_back(std::move(prediction));
    }
    return predictions;
}

InferenceEngine::Prediction InferenceEngine::fuse_group_predictions(
    const std::vector<Prediction>& predictions,
    const int anchor_index,
    InferenceMetrics& metrics) const {
    if (predictions.size() != 3U || anchor_index < 0 || anchor_index >= 3) {
        throw std::runtime_error("group prediction fusion expects three predictions");
    }
    Prediction fused = predictions[static_cast<std::size_t>(anchor_index)];
    const cv::Mat anchor_valid = model_valid_mask(
        fused.depth, fused.confidence,
        cv::Mat(fused.depth.size(), CV_8UC1, cv::Scalar(255)),
        static_cast<float>(options_.min_conf));
    for (int source_index = 0; source_index < 3; ++source_index) {
        if (source_index == anchor_index) {
            continue;
        }
        const Prediction& source = predictions[static_cast<std::size_t>(source_index)];
        std::vector<float> source_values;
        std::vector<float> anchor_values;
        const int step = std::max(1, fused.depth.rows * fused.depth.cols / 60000);
        int ordinal = 0;
        for (int y = 0; y < fused.depth.rows; ++y) {
            for (int x = 0; x < fused.depth.cols; ++x) {
                if ((ordinal++ % step) != 0
                    || anchor_valid.at<std::uint8_t>(y, x) == 0U) {
                    continue;
                }
                const float side = source.depth.at<float>(y, x);
                const float anchor = fused.depth.at<float>(y, x);
                if (std::isfinite(side) && side > 1e-6f && std::isfinite(anchor)
                    && anchor > 1e-6f
                    && source.confidence.at<float>(y, x) >= static_cast<float>(options_.min_conf)) {
                    source_values.push_back(side);
                    anchor_values.push_back(anchor);
                }
            }
        }
        if (source_values.size() < 128U) {
            ++metrics.group_rejected_sources;
            continue;
        }
        double scale = 1.0;
        double bias = static_cast<double>(median_value(anchor_values))
            - static_cast<double>(median_value(source_values));
        std::vector<std::uint8_t> keep(source_values.size(), 1U);
        for (int iteration = 0; iteration < 4; ++iteration) {
            double sw = 0.0;
            double sz = 0.0;
            double sd = 0.0;
            double szz = 0.0;
            double szd = 0.0;
            for (std::size_t i = 0; i < source_values.size(); ++i) {
                if (keep[i] == 0U) {
                    continue;
                }
                const double weight = 1.0;
                const double z = source_values[i];
                const double dst = anchor_values[i];
                sw += weight;
                sz += weight * z;
                sd += weight * dst;
                szz += weight * z * z;
                szd += weight * z * dst;
            }
            const double denominator = sw * szz - sz * sz;
            if (std::abs(denominator) > 1e-9) {
                scale = (sw * szd - sz * sd) / denominator;
                bias = (sd - scale * sz) / sw;
            }
            std::vector<float> residuals;
            residuals.reserve(source_values.size());
            for (std::size_t i = 0; i < source_values.size(); ++i) {
                residuals.push_back(anchor_values[i]
                    - static_cast<float>(scale * source_values[i] + bias));
            }
            const float center = median_value(residuals);
            std::vector<float> deviations;
            deviations.reserve(residuals.size());
            for (const float value : residuals) {
                deviations.push_back(std::abs(value - center));
            }
            const float limit = std::max(0.08f, 3.0f * 1.4826f
                * std::max(median_value(deviations), 1e-6f));
            for (std::size_t i = 0; i < residuals.size(); ++i) {
                keep[i] = std::abs(residuals[i] - center) <= limit ? 1U : 0U;
            }
        }
        if (!std::isfinite(scale) || !std::isfinite(bias) || scale <= 0.0 || scale > 8.0) {
            ++metrics.group_rejected_sources;
            continue;
        }
        scale = std::clamp(scale, 0.25, 4.0);
        bias = std::clamp(bias, -10.0, 10.0);
        std::vector<float> residuals;
        residuals.reserve(source_values.size());
        for (std::size_t i = 0; i < source_values.size(); ++i) {
            residuals.push_back(std::abs(anchor_values[i]
                - static_cast<float>(scale * source_values[i] + bias)));
        }
        const float median_residual = median_value(residuals);
        metrics.group_max_depth_residual = std::max(
            metrics.group_max_depth_residual, static_cast<double>(median_residual));
        if (!std::isfinite(median_residual) || median_residual > 0.25f) {
            ++metrics.group_rejected_sources;
            continue;
        }
        ++metrics.group_fused_sources;
        for (int y = 0; y < fused.depth.rows; ++y) {
            for (int x = 0; x < fused.depth.cols; ++x) {
                if (anchor_valid.at<std::uint8_t>(y, x) != 0U) {
                    // The anchor owns every overlap pixel.  This is the key
                    // single-layer rule: side predictions can only fill a
                    // genuinely invalid anchor pixel after calibration.
                    continue;
                }
                const float side = source.depth.at<float>(y, x);
                const float confidence = source.confidence.at<float>(y, x);
                if (std::isfinite(side) && side > 1e-6f
                    && std::isfinite(confidence)
                    && confidence >= static_cast<float>(options_.min_conf)) {
                    fused.depth.at<float>(y, x) = static_cast<float>(scale * side + bias);
                    fused.confidence.at<float>(y, x) = confidence * 0.75f;
                }
            }
        }
    }
    return fused;
}

cv::Mat InferenceEngine::gray_u8(const cv::Mat& rgb_u8) {
    // Match Python's channel-mean matcher.  OpenCV's RGB2GRAY uses
    // luminance weights and changes feature/phase matches on this data.
    std::vector<cv::Mat> channels;
    cv::split(rgb_u8, channels);
    cv::Mat c0;
    cv::Mat c1;
    cv::Mat c2;
    channels[0].convertTo(c0, CV_32FC1);
    channels[1].convertTo(c1, CV_32FC1);
    channels[2].convertTo(c2, CV_32FC1);
    cv::Mat gray = (c0 + c1 + c2) / 3.0f;
    cv::Mat gray_u8(gray.rows, gray.cols, CV_8UC1, cv::Scalar(0));
    const float scale = rgb_u8.type() == CV_32FC3 ? 255.0f : 1.0f;
    for (int y = 0; y < gray.rows; ++y) {
        for (int x = 0; x < gray.cols; ++x) {
            gray_u8.at<std::uint8_t>(y, x) = static_cast<std::uint8_t>(std::clamp(
                static_cast<int>(gray.at<float>(y, x) * scale), 0, 255));
        }
    }
    return gray_u8;
}

cv::Mat InferenceEngine::translation_h(const double dx, const double dy) {
    cv::Mat homography = cv::Mat::eye(3, 3, CV_32FC1);
    homography.at<float>(0, 2) = static_cast<float>(dx);
    homography.at<float>(1, 2) = static_cast<float>(dy);
    return homography;
}

cv::Mat InferenceEngine::warp_like(
    const cv::Mat& source,
    const cv::Mat& homography,
    const cv::Size size,
    const int interpolation) {
    cv::Mat result;
    cv::warpPerspective(source, result, homography, size, interpolation, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    return result;
}

cv::Mat InferenceEngine::dilate_mask(const cv::Mat& mask, const int kernel_size) {
    if (kernel_size <= 1 || cv::countNonZero(mask) == 0) {
        return mask.clone();
    }
    cv::Mat result;
    const cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
    cv::dilate(mask, result, kernel);
    return result;
}

cv::Mat InferenceEngine::state_rgb_float(const CanvasState& state) {
    cv::Mat result(state.height, state.width, CV_32FC3, cv::Scalar::all(0));
    for (int y = 0; y < state.height; ++y) {
        for (int x = 0; x < state.width; ++x) {
            const auto rgba = unpack_rgba(state.rgba[static_cast<std::size_t>(y) * state.width + x]);
            result.at<cv::Vec3f>(y, x) = cv::Vec3f(
                static_cast<float>(rgba[0]) / 255.0f,
                static_cast<float>(rgba[1]) / 255.0f,
                static_cast<float>(rgba[2]) / 255.0f);
        }
    }
    return result;
}

cv::Mat InferenceEngine::anchor_rgb_float(const CanvasState& state) {
    cv::Mat result(state.height, state.width, CV_32FC3, cv::Scalar::all(0));
    if (state.anchor_rgba.size() != state.slot_count()) {
        return result;
    }
    for (int y = 0; y < state.height; ++y) {
        for (int x = 0; x < state.width; ++x) {
            const auto rgba = unpack_rgba(state.anchor_rgba[static_cast<std::size_t>(y) * state.width + x]);
            result.at<cv::Vec3f>(y, x) = cv::Vec3f(
                static_cast<float>(rgba[0]) / 255.0f,
                static_cast<float>(rgba[1]) / 255.0f,
                static_cast<float>(rgba[2]) / 255.0f);
        }
    }
    return result;
}

cv::Mat InferenceEngine::state_mask(
    const std::vector<std::uint8_t>& values,
    const int width,
    const int height) {
    if (values.size() != static_cast<std::size_t>(width) * static_cast<std::size_t>(height)) {
        throw std::runtime_error("CanvasState mask shape mismatch");
    }
    cv::Mat result(height, width, CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            result.at<std::uint8_t>(y, x) = values[static_cast<std::size_t>(y) * width + x] != 0U ? 255U : 0U;
        }
    }
    return result;
}

std::uint32_t InferenceEngine::color_at(const cv::Mat& rgb_u8, const int x, const int y) {
    const cv::Vec3b color = rgb_u8.at<cv::Vec3b>(y, x);
    return pack_rgba(color[0], color[1], color[2]);
}

std::vector<std::uint32_t> InferenceEngine::pack_image(const cv::Mat& rgb_u8) {
    std::vector<std::uint32_t> result(
        static_cast<std::size_t>(rgb_u8.rows) * static_cast<std::size_t>(rgb_u8.cols), 0U);
    for (int y = 0; y < rgb_u8.rows; ++y) {
        for (int x = 0; x < rgb_u8.cols; ++x) {
            result[static_cast<std::size_t>(y) * rgb_u8.cols + x] = color_at(rgb_u8, x, y);
        }
    }
    return result;
}

cv::Mat InferenceEngine::estimate_homography(
    const FrameImage& current,
    const CanvasState& state,
    InferenceMetrics& metrics) const {
    if (!state.initialized) {
        return cv::Mat::eye(3, 3, CV_32FC1);
    }
    const cv::Mat anchor_rgb = !anchor_rgb_float_.empty()
        && anchor_rgb_float_.size() == cv::Size(state.width, state.height)
        ? anchor_rgb_float_
        : anchor_rgb_float(state);
    const cv::Mat current_gray = gray_u8(current.rgb_f);
    const cv::Mat anchor_gray = gray_u8(anchor_rgb);
    cv::Ptr<cv::Feature2D> detector;
    int norm = cv::NORM_HAMMING;
    try {
        detector = cv::SIFT::create(1500);
        norm = cv::NORM_L2;
    } catch (const cv::Exception&) {
        detector = cv::ORB::create(1500);
    }
    std::vector<cv::KeyPoint> current_keypoints;
    std::vector<cv::KeyPoint> anchor_keypoints;
    cv::Mat current_descriptors;
    cv::Mat anchor_descriptors;
    detector->detectAndCompute(current_gray, cv::noArray(), current_keypoints, current_descriptors);
    detector->detectAndCompute(anchor_gray, cv::noArray(), anchor_keypoints, anchor_descriptors);

    const auto phase_fallback = [&](const std::string& reason) {
        cv::Mat current_float;
        cv::Mat anchor_float;
        current_gray.convertTo(current_float, CV_32FC1, 1.0 / 255.0);
        anchor_gray.convertTo(anchor_float, CV_32FC1, 1.0 / 255.0);
        const cv::Point2d shift = cv::phaseCorrelate(current_float, anchor_float);
        metrics.fallback = reason;
        return translation_h(shift.x, shift.y);
    };

    if (current_descriptors.empty() || anchor_descriptors.empty()
        || current_keypoints.size() < 8U || anchor_keypoints.size() < 8U) {
        return phase_fallback("few_features");
    }
    cv::BFMatcher matcher(norm);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(current_descriptors, anchor_descriptors, knn, 2);
    std::vector<cv::DMatch> good;
    for (const auto& pair : knn) {
        if (pair.size() == 2U && pair[0].distance < 0.75f * pair[1].distance) {
            good.push_back(pair[0]);
        }
    }
    if (good.size() < 8U) {
        return phase_fallback("few_matches");
    }
    std::vector<cv::Point2f> current_points;
    std::vector<cv::Point2f> anchor_points;
    current_points.reserve(good.size());
    anchor_points.reserve(good.size());
    for (const cv::DMatch& match : good) {
        current_points.push_back(current_keypoints[static_cast<std::size_t>(match.queryIdx)].pt);
        anchor_points.push_back(anchor_keypoints[static_cast<std::size_t>(match.trainIdx)].pt);
    }
    cv::Mat inlier_mask;
    cv::Mat homography = cv::findHomography(current_points, anchor_points, cv::RANSAC, 3.0, inlier_mask);
    if (homography.empty() || inlier_mask.empty() || cv::countNonZero(inlier_mask) < 8) {
        return phase_fallback("bad_homography");
    }
    homography.convertTo(homography, CV_32FC1);
    metrics.homography_inliers = cv::countNonZero(inlier_mask);
    std::vector<cv::Point2f> projected;
    cv::perspectiveTransform(current_points, projected, homography);
    std::vector<double> errors;
    for (std::size_t i = 0; i < projected.size(); ++i) {
        if (inlier_mask.at<std::uint8_t>(static_cast<int>(i), 0) == 0U) {
            continue;
        }
        const cv::Point2f difference = projected[i] - anchor_points[i];
        errors.push_back(std::sqrt(static_cast<double>(difference.x * difference.x + difference.y * difference.y)));
    }
    if (!errors.empty()) {
        std::nth_element(errors.begin(), errors.begin() + static_cast<std::ptrdiff_t>(errors.size() / 2), errors.end());
        metrics.homography_error_px = errors[errors.size() / 2];
    }
    return homography;
}

cv::Mat InferenceEngine::estimate_pair_homography(
    const FrameImage& source,
    const FrameImage& target) const {
    if (source.match_rgb_u8.empty() || target.match_rgb_u8.empty()) {
        return cv::Mat::eye(3, 3, CV_32FC1);
    }
    const cv::Mat source_gray = gray_u8(source.match_rgb_u8);
    const cv::Mat target_gray = gray_u8(target.match_rgb_u8);
    cv::Ptr<cv::Feature2D> detector;
    int norm = cv::NORM_HAMMING;
    try {
        detector = cv::SIFT::create(1200);
        norm = cv::NORM_L2;
    } catch (const cv::Exception&) {
        detector = cv::ORB::create(1200);
    }
    std::vector<cv::KeyPoint> source_keypoints;
    std::vector<cv::KeyPoint> target_keypoints;
    cv::Mat source_descriptors;
    cv::Mat target_descriptors;
    detector->detectAndCompute(source_gray, cv::noArray(), source_keypoints, source_descriptors);
    detector->detectAndCompute(target_gray, cv::noArray(), target_keypoints, target_descriptors);
    if (source_descriptors.empty() || target_descriptors.empty()
        || source_keypoints.size() < 8U || target_keypoints.size() < 8U) {
        return cv::Mat::eye(3, 3, CV_32FC1);
    }
    cv::BFMatcher matcher(norm);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(source_descriptors, target_descriptors, knn, 2);
    std::vector<cv::DMatch> good;
    for (const auto& pair : knn) {
        if (pair.size() == 2U && pair[0].distance < 0.75f * pair[1].distance) {
            good.push_back(pair[0]);
        }
    }
    if (good.size() < 8U) {
        return cv::Mat::eye(3, 3, CV_32FC1);
    }
    std::vector<cv::Point2f> source_points;
    std::vector<cv::Point2f> target_points;
    source_points.reserve(good.size());
    target_points.reserve(good.size());
    for (const cv::DMatch& match : good) {
        source_points.push_back(source_keypoints[static_cast<std::size_t>(match.queryIdx)].pt);
        target_points.push_back(target_keypoints[static_cast<std::size_t>(match.trainIdx)].pt);
    }
    cv::Mat inlier_mask;
    cv::Mat homography = cv::findHomography(
        source_points, target_points, cv::RANSAC, 3.0, inlier_mask);
    if (homography.empty() || inlier_mask.empty() || cv::countNonZero(inlier_mask) < 8) {
        return cv::Mat::eye(3, 3, CV_32FC1);
    }
    homography.convertTo(homography, CV_32FC1);
    return homography;
}

cv::Mat InferenceEngine::compute_change_mask(
    const FrameImage& current,
    const cv::Mat& homography,
    const CanvasState& state,
    InferenceMetrics& metrics,
    cv::Mat& warped_rgb_f,
    cv::Mat& valid_warp,
    cv::Mat& support_change,
    cv::Mat& photometric_change) const {
    if (!state.initialized) {
        warped_rgb_f = current.rgb_f.clone();
        valid_warp = current.support.clone();
        support_change = valid_warp.clone();
        photometric_change = cv::Mat::zeros(valid_warp.size(), CV_8UC1);
        metrics.support_changed_ratio = static_cast<double>(cv::countNonZero(valid_warp))
            / static_cast<double>(valid_warp.total());
        return valid_warp.clone();
    }
    warped_rgb_f = warp_like(current.rgb_f, homography, cv::Size(state.width, state.height), cv::INTER_LINEAR);
    const cv::Mat warped_support = warp_like(
        current.support, homography, cv::Size(state.width, state.height), cv::INTER_NEAREST);
    cv::threshold(warped_support, valid_warp, 127.0, 255.0, cv::THRESH_BINARY);
    const cv::Mat canvas_support = state_mask(state.support, state.width, state.height);
    const cv::Mat canvas_valid = state_mask(state.valid, state.width, state.height);
    cv::Mat overlap;
    cv::bitwise_and(valid_warp, canvas_support, overlap);
    cv::bitwise_and(overlap, canvas_valid, overlap);

    const cv::Mat canvas_rgb = !live_rgb_float_.empty()
        && live_rgb_float_.size() == cv::Size(state.width, state.height)
        ? live_rgb_float_
        : state_rgb_float(state);
    cv::Mat absolute_difference;
    cv::absdiff(warped_rgb_f, canvas_rgb, absolute_difference);
    std::vector<cv::Mat> channels;
    cv::split(absolute_difference, channels);
    cv::Mat difference = (channels[0] + channels[1] + channels[2]) / 3.0f;
    std::vector<float> overlap_values;
    overlap_values.reserve(static_cast<std::size_t>(cv::countNonZero(overlap)));
    for (int y = 0; y < difference.rows; ++y) {
        for (int x = 0; x < difference.cols; ++x) {
            if (overlap.at<std::uint8_t>(y, x) != 0U) {
                overlap_values.push_back(difference.at<float>(y, x));
            }
        }
    }
    const float overlap_center = median_value(overlap_values);
    std::vector<float> overlap_deviations;
    overlap_deviations.reserve(overlap_values.size());
    for (const float value : overlap_values) {
        overlap_deviations.push_back(std::abs(value - overlap_center));
    }
    const float overlap_mad = std::max(median_value(overlap_deviations), 1e-6f);
    const float robust_threshold = std::clamp(
        std::max(static_cast<float>(options_.image_l1_thr * 2.0),
            overlap_center + 3.0f * 1.4826f * overlap_mad),
        0.08f,
        0.22f);
    cv::Mat photo_mask;
    cv::threshold(difference, photo_mask, robust_threshold, 255.0, cv::THRESH_BINARY);
    photo_mask.convertTo(photo_mask, CV_8UC1);
    cv::bitwise_and(photo_mask, overlap, photo_mask);
    photo_mask = filter_components(photo_mask, 128);

    double photo_ratio = static_cast<double>(cv::countNonZero(photo_mask))
        / static_cast<double>(photo_mask.total());

    cv::Mat inverse_support;
    cv::bitwise_not(canvas_support, inverse_support);
    cv::bitwise_and(valid_warp, inverse_support, support_change);
    const double support_ratio = static_cast<double>(cv::countNonZero(support_change))
        / static_cast<double>(support_change.total());
    if (photo_ratio > options_.scene_jump_ratio
        || (photo_ratio > 0.15 && support_ratio < 0.05)) {
        photo_mask.setTo(0);
        photo_ratio = 0.0;
    }
    cv::Mat change;
    cv::bitwise_or(photo_mask, support_change, change);
    change = dilate_mask(change, options_.dilate_ksize);
    cv::bitwise_and(change, valid_warp, change);
    metrics.photometric_changed_ratio = photo_ratio;
    metrics.support_changed_ratio = static_cast<double>(cv::countNonZero(support_change))
        / static_cast<double>(support_change.total());
    // Python returns the committed masks after the same 3-pixel dilation
    // used by the fusion mask.  The old C++ path exposed the raw masks here,
    // so update_mask was one pixel narrower than the Python path and the
    // depth/color seam was written on different support.
    cv::Mat support_commit = dilate_mask(support_change, options_.dilate_ksize);
    cv::bitwise_and(support_commit, valid_warp, support_commit);
    cv::Mat photo_commit = dilate_mask(photo_mask, options_.dilate_ksize);
    cv::bitwise_and(photo_commit, overlap, photo_commit);
    support_change = std::move(support_commit);
    photometric_change = std::move(photo_commit);
    return change;
}

cv::Mat InferenceEngine::align_depth_to_canvas(
    const cv::Mat& depth,
    const cv::Mat& confidence,
    const cv::Mat& valid_warp,
    const CanvasState& state,
    const cv::Mat& anchor_mask) const {
    if (!state.initialized) {
        return depth.clone();
    }

    const cv::Mat canvas_valid = state_mask(state.valid, state.width, state.height);
    const bool anchor_requested = !anchor_mask.empty();
    const bool use_anchor = anchor_requested && cv::countNonZero(anchor_mask) >= 128;
    const cv::Mat& calibration_mask = use_anchor ? anchor_mask : valid_warp;
    std::vector<float> source_values;
    std::vector<float> destination_values;
    std::vector<float> confidence_values;
    source_values.reserve(static_cast<std::size_t>(depth.total()));
    destination_values.reserve(static_cast<std::size_t>(depth.total()));
    confidence_values.reserve(static_cast<std::size_t>(depth.total()));

    const auto collect = [&](const cv::Mat& mask) {
        source_values.clear();
        destination_values.clear();
        confidence_values.clear();
        for (int y = 0; y < depth.rows; ++y) {
            for (int x = 0; x < depth.cols; ++x) {
                const std::size_t index = static_cast<std::size_t>(y) * state.width + x;
                if (mask.at<std::uint8_t>(y, x) == 0U
                    || valid_warp.at<std::uint8_t>(y, x) == 0U
                    || canvas_valid.at<std::uint8_t>(y, x) == 0U) {
                    continue;
                }
                const float z = depth.at<float>(y, x);
                const float destination = state.depth[index];
                const float confidence_value = confidence.at<float>(y, x);
                if (!std::isfinite(z) || !std::isfinite(destination)
                    || confidence_value <= static_cast<float>(options_.min_conf)) {
                    continue;
                }
                source_values.push_back(z);
                destination_values.push_back(destination);
                confidence_values.push_back(std::max(confidence_value, 1e-4f));
            }
        }
    };

    collect(calibration_mask);
    if (source_values.size() < 128 && anchor_requested) {
        // A ring clipped by an image border is not enough to calibrate depth;
        // use the broad valid overlap only as a fallback.
        collect(valid_warp);
    }
    if (source_values.size() < 128) {
        return depth.clone();
    }

    // NumPy's default percentile method is linear interpolation between the
    // two surrounding order statistics.  Selecting one lower element with
    // nth_element (the previous C++ implementation) changes the inlier set
    // at the model/ROI boundary and can leave a thin depth step.
    std::vector<float> quantiles = source_values;
    std::sort(quantiles.begin(), quantiles.end());
    const auto linear_percentile = [&quantiles](const double fraction) {
        const double position = fraction * static_cast<double>(quantiles.size() - 1U);
        const std::size_t lower = static_cast<std::size_t>(std::floor(position));
        const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
        const float weight = static_cast<float>(position - static_cast<double>(lower));
        return quantiles[lower] + (quantiles[upper] - quantiles[lower]) * weight;
    };
    const float low = linear_percentile(0.02);
    const float high = linear_percentile(0.98);
    std::vector<float> filtered_source;
    std::vector<float> filtered_destination;
    std::vector<float> filtered_confidence;
    filtered_source.reserve(source_values.size());
    filtered_destination.reserve(destination_values.size());
    filtered_confidence.reserve(confidence_values.size());
    for (std::size_t i = 0; i < source_values.size(); ++i) {
        if (source_values[i] >= low && source_values[i] <= high) {
            filtered_source.push_back(source_values[i]);
            filtered_destination.push_back(destination_values[i]);
            filtered_confidence.push_back(confidence_values[i]);
        }
    }
    source_values = std::move(filtered_source);
    destination_values = std::move(filtered_destination);
    confidence_values = std::move(filtered_confidence);
    if (source_values.size() < 128) {
        return depth.clone();
    }

    std::vector<std::uint8_t> keep(source_values.size(), 1U);
    double scale = 1.0;
    double bias = static_cast<double>(median_value(destination_values))
        - static_cast<double>(median_value(source_values));
    for (int iteration = 0; iteration < 4; ++iteration) {
        double sum_w = 0.0;
        double sum_z = 0.0;
        double sum_dst = 0.0;
        double sum_zz = 0.0;
        double sum_zdst = 0.0;
        for (std::size_t i = 0; i < source_values.size(); ++i) {
            if (keep[i] == 0U) {
                continue;
            }
            // Python solves ``lstsq(design * sqrt(conf), dst * sqrt(conf))``.
            // Its normal equations therefore accumulate ``conf`` (not
            // sqrt(conf)).  Using sqrt(conf) here overweights low-confidence
            // ROI-edge samples and shifts the reconstructed height at the
            // anchor seam.
            const double weight = static_cast<double>(confidence_values[i]);
            const double z = source_values[i];
            const double destination = destination_values[i];
            sum_w += weight;
            sum_z += weight * z;
            sum_dst += weight * destination;
            sum_zz += weight * z * z;
            sum_zdst += weight * z * destination;
        }
        const double denominator = sum_w * sum_zz - sum_z * sum_z;
        if (sum_w > 1e-9 && std::abs(denominator) > 1e-9) {
            scale = (sum_w * sum_zdst - sum_z * sum_dst) / denominator;
            bias = (sum_dst - scale * sum_z) / sum_w;
        }
        std::vector<float> residuals;
        residuals.reserve(source_values.size());
        for (std::size_t i = 0; i < source_values.size(); ++i) {
            residuals.push_back(destination_values[i]
                - static_cast<float>(scale * source_values[i] + bias));
        }
        const float center = median_value(residuals);
        std::vector<float> deviations;
        deviations.reserve(residuals.size());
        for (const float residual : residuals) {
            deviations.push_back(std::abs(residual - center));
        }
        const float mad = std::max(median_value(deviations), 1e-6f);
        const float limit = 3.0f * 1.4826f * mad;
        for (std::size_t i = 0; i < residuals.size(); ++i) {
            keep[i] = std::abs(residuals[i] - center) <= limit ? 1U : 0U;
        }
    }
    if (!std::isfinite(scale) || !std::isfinite(bias) || scale <= 0.0 || scale > 8.0) {
        scale = 1.0;
        bias = static_cast<double>(median_value(destination_values))
            - static_cast<double>(median_value(source_values));
    }
    scale = std::clamp(scale, 0.25, 4.0);
    bias = std::clamp(bias, -10.0, 10.0);
    cv::Mat result;
    depth.convertTo(result, CV_32FC1, scale, bias);
    // Match Python's post-affine seam correction.  A global scale/bias can
    // agree on the ring while still leaving a local tilt at the newly exposed
    // boundary; carrying that residual into the target ROI prevents the two
    // depth layers seen in the C++ viewer.
    cv::Mat seam = calibration_mask.clone();
    cv::bitwise_and(seam, canvas_valid, seam);
    cv::bitwise_and(seam, finite_mask(result), seam);
    cv::Mat target = valid_warp.clone();
    cv::bitwise_and(target, finite_mask(result), target);
    if (cv::countNonZero(seam) >= 64 && cv::countNonZero(target) > 0) {
        cv::Mat canvas_depth(state.height, state.width, CV_32FC1);
        for (int y = 0; y < state.height; ++y) {
            for (int x = 0; x < state.width; ++x) {
                canvas_depth.at<float>(y, x) = state.depth[
                    static_cast<std::size_t>(y) * state.width + x];
            }
        }
        cv::Mat residual = canvas_depth - result;
        const cv::Mat spatial = fit_spatial_seam_residual(
            residual, seam, target, 0.08f);
        const cv::Mat propagated = propagate_seam_residual(
            residual, seam, target, 24.0, 0.08f);
        cv::Mat correction = 0.78f * spatial + 0.22f * propagated;
        cv::min(correction, cv::Scalar(0.08f), correction);
        cv::max(correction, cv::Scalar(-0.08f), correction);
        result += correction;
    }
    return result;
}

void InferenceEngine::save_debug_images(
    const FrameImage& frame,
    const cv::Mat& warped_rgb_f,
    const cv::Mat& change_mask,
    const cv::Mat& valid_warp,
    const cv::Mat& anchor_ring,
    const cv::Mat& update_mask,
    const cv::Mat& color_bridge_mask,
    const cv::Mat& fused_rgb) const {
    if (!options_.save_debug || options_.debug_dir.empty()) {
        return;
    }
    std::filesystem::create_directories(options_.debug_dir);
    cv::Mat warped_u8;
    warped_rgb_f.convertTo(warped_u8, CV_8UC3, 255.0);
    cv::Mat warped_bgr;
    cv::cvtColor(warped_u8, warped_bgr, cv::COLOR_RGB2BGR);
    cv::Mat fused_u8 = quantize_rgb_u8(fused_rgb);
    cv::Mat fused_bgr;
    cv::cvtColor(fused_u8, fused_bgr, cv::COLOR_RGB2BGR);
    const std::string stem = frame.path.stem().string();
    cv::imwrite((options_.debug_dir / (stem + "_warped.png")).string(), warped_bgr);
    cv::imwrite((options_.debug_dir / (stem + "_fused_rgb.png")).string(), fused_bgr);
    cv::imwrite((options_.debug_dir / (stem + "_change_mask.png")).string(), change_mask);
    cv::imwrite((options_.debug_dir / (stem + "_support.png")).string(), valid_warp);
    cv::imwrite((options_.debug_dir / (stem + "_anchor_ring.png")).string(), anchor_ring);
    cv::imwrite((options_.debug_dir / (stem + "_update_mask.png")).string(), update_mask);
    cv::imwrite((options_.debug_dir / (stem + "_color_bridge.png")).string(), color_bridge_mask);
}

CandidateCommit InferenceEngine::process(const RawFrame& raw, const CanvasState& state) {
    if (options_.group_mode && raw.group_paths.size() == 3U) {
        return process_group(raw, state);
    }
    Timer read_timer;
    const FrameImage frame = load_frame(raw.path);
    return process_impl(raw, state, frame, nullptr, read_timer.ms());
}

CandidateCommit InferenceEngine::process_impl(
    const RawFrame& raw,
    const CanvasState& state,
    const FrameImage& frame,
    const PreparedGroup* prepared_group,
    const double read_ms) {
    Timer total_timer;
    CandidateCommit result;
    result.frame.frame_seq = raw.frame_seq;
    result.frame.base_version = state.version;
    result.frame.commit_version = state.version;
    result.frame.image_name = raw.path.filename().string();
    result.metrics.frame_seq = raw.frame_seq;
    result.metrics.image = result.frame.image_name;

    result.metrics.read_ms = read_ms;
    result.metrics.group_size = prepared_group == nullptr ? 1 : 3;
    result.metrics.group_stride = prepared_group == nullptr ? 1 : options_.group_stride;
    result.metrics.group_anchor_index = prepared_group == nullptr ? 0 : raw.group_anchor_index;
    result.metrics.forward_batch_size = prepared_group == nullptr ? 1 : 3;
    result.metrics.forward_sequence_size = 1;
    if (!state.initialized) {
        anchor_rgb_float_ = frame.rgb_f.clone();
        live_rgb_float_ = frame.rgb_f.clone();
    } else {
        if (anchor_rgb_float_.empty()
            || anchor_rgb_float_.size() != cv::Size(state.width, state.height)) {
            anchor_rgb_float_ = anchor_rgb_float(state);
        }
        if (live_rgb_float_.empty()
            || live_rgb_float_.size() != cv::Size(state.width, state.height)) {
            live_rgb_float_ = state_rgb_float(state);
        }
    }

    Timer align_timer;
    InferenceMetrics metrics = result.metrics;
    cv::Mat homography = estimate_homography(frame, state, metrics);
    if (state.initialized && !metrics.fallback.empty() && !last_homography_.empty()) {
        // Python's feature build accepted this frame in the reference replay,
        // while the native OpenCV build can fall below the eight-point RANSAC
        // minimum on a late view. Reuse the last accepted current-to-canvas
        // transform and let the same dense phase refinement correct the small
        // inter-frame translation. This preserves the frame instead of
        // publishing a false jump or silently skipping its model call.
        homography = last_homography_.clone();
        metrics.fallback.clear();
        metrics.homography_inliers = 0;
        metrics.homography_error_px = -1.0;
    }
    if (state.initialized && metrics.fallback.empty()) {
        // Python performs a second, small phase-correlation correction on the
        // unchanged overlap after feature RANSAC.  Without it a sub-pixel
        // homography error is converted into a one-pixel support/photo strip.
        const cv::Size canvas_size(state.width, state.height);
        const cv::Mat coarse_warp = warp_like(frame.rgb_f, homography, canvas_size, cv::INTER_LINEAR);
        const cv::Mat coarse_support = warp_like(frame.support, homography, canvas_size, cv::INTER_NEAREST);
        cv::Mat overlap;
        cv::bitwise_and(coarse_support, state_mask(state.valid, state.width, state.height), overlap);
        if (cv::countNonZero(overlap) >= 2048) {
            cv::Mat canvas_rgb = !live_rgb_float_.empty()
                && live_rgb_float_.size() == cv::Size(state.width, state.height)
                ? live_rgb_float_
                : state_rgb_float(state);
            cv::Mat warped_gray;
            cv::Mat canvas_gray;
            cv::cvtColor(coarse_warp, warped_gray, cv::COLOR_RGB2GRAY);
            cv::cvtColor(canvas_rgb, canvas_gray, cv::COLOR_RGB2GRAY);
            cv::Mat difference;
            cv::absdiff(coarse_warp, canvas_rgb, difference);
            std::vector<cv::Mat> difference_channels;
            cv::split(difference, difference_channels);
            cv::Mat difference_gray =
                (difference_channels[0] + difference_channels[1] + difference_channels[2]) / 3.0f;
            std::vector<float> values;
            values.reserve(static_cast<std::size_t>(cv::countNonZero(overlap)));
            for (int y = 0; y < overlap.rows; ++y) {
                for (int x = 0; x < overlap.cols; ++x) {
                    if (overlap.at<std::uint8_t>(y, x) != 0U) {
                        values.push_back(difference_gray.at<float>(y, x));
                    }
                }
            }
            const float center = median_value(values);
            std::vector<float> deviations;
            deviations.reserve(values.size());
            for (const float value : values) {
                deviations.push_back(std::abs(value - center));
            }
            const float mad = std::max(median_value(deviations), 1e-6f);
            cv::Mat stable;
            cv::compare(difference_gray, std::max(0.08f, center + 3.0f * 1.4826f * mad), stable, cv::CMP_LE);
            cv::bitwise_and(stable, overlap, stable);
            if (cv::countNonZero(stable) >= 2048) {
                cv::Rect box = cv::boundingRect(stable);
                const int pad = 24;
                const int x0 = std::max(0, box.x - pad);
                const int y0 = std::max(0, box.y - pad);
                const int x1 = std::min(state.width, box.x + box.width + pad);
                const int y1 = std::min(state.height, box.y + box.height + pad);
                if (x1 - x0 >= 64 && y1 - y0 >= 64) {
                    const cv::Rect crop(x0, y0, x1 - x0, y1 - y0);
                    cv::Mat a = warped_gray(crop).clone();
                    cv::Mat b = canvas_gray(crop).clone();
                    cv::Mat mask = stable(crop);
                    const cv::Scalar a_mean = cv::mean(a, mask);
                    const cv::Scalar b_mean = cv::mean(b, mask);
                    a -= a_mean[0];
                    b -= b_mean[0];
                    cv::Mat inverse_mask;
                    cv::bitwise_not(mask, inverse_mask);
                    a.setTo(0.0f, inverse_mask);
                    b.setTo(0.0f, inverse_mask);
                    cv::Mat window;
                    cv::createHanningWindow(window, cv::Size(crop.width, crop.height), CV_32F);
                    double response = 0.0;
                    const cv::Mat windowed_a = a.mul(window);
                    const cv::Mat windowed_b = b.mul(window);
                    // Python applies the Hanning window once before calling
                    // phaseCorrelate.  Passing it as the third argument too
                    // applies it twice and biases the edge translation.
                    const cv::Point2d shift = cv::phaseCorrelate(
                        windowed_a, windowed_b, cv::noArray(), &response);
                    const double dx = std::clamp(shift.x, -8.0, 8.0);
                    const double dy = std::clamp(shift.y, -8.0, 8.0);
                    if (std::isfinite(dx) && std::isfinite(dy) && response >= 0.02) {
                        homography = translation_h(dx, dy) * homography;
                    }
                }
            }
        }
    }
    result.metrics = metrics;
    result.metrics.align2d_ms = align_timer.ms();
    if (options_.save_debug && !options_.debug_dir.empty()) {
        std::filesystem::create_directories(options_.debug_dir);
        std::ofstream transform_file(
            options_.debug_dir / (frame.path.stem().string() + "_homography.txt"));
        if (transform_file) {
            transform_file << std::setprecision(9);
            for (int row = 0; row < 3; ++row) {
                transform_file << homography.at<float>(row, 0) << ' '
                    << homography.at<float>(row, 1) << ' '
                    << homography.at<float>(row, 2) << '\n';
            }
        }
    }
    if (metrics.fallback.empty()) {
        last_homography_ = homography.clone();
    }

    Timer diff_timer;
    cv::Mat warped_rgb_f;
    cv::Mat valid_warp;
    cv::Mat support_change;
    cv::Mat photometric_change;
    cv::Mat change_mask = compute_change_mask(
        frame,
        homography,
        state,
        result.metrics,
        warped_rgb_f,
        valid_warp,
        support_change,
        photometric_change);
    std::vector<cv::Mat> grouped_warped_rgb_f;
    std::vector<cv::Mat> grouped_valid_warp;
    if (prepared_group != nullptr) {
        const cv::Size canvas_size(state.width, state.height);
        grouped_warped_rgb_f.reserve(prepared_group->warped_rgb_f.size());
        grouped_valid_warp.reserve(prepared_group->valid_warp.size());
        for (std::size_t index = 0; index < prepared_group->warped_rgb_f.size(); ++index) {
            grouped_warped_rgb_f.push_back(warp_like(
                prepared_group->warped_rgb_f[index], homography, canvas_size, cv::INTER_LINEAR));
            cv::Mat support = warp_like(
                prepared_group->valid_warp[index], homography, canvas_size, cv::INTER_NEAREST);
            cv::threshold(support, support, 127.0, 255.0, cv::THRESH_BINARY);
            support.convertTo(support, CV_8UC1);
            grouped_valid_warp.push_back(std::move(support));
        }
        if (grouped_warped_rgb_f.size() != 3U || grouped_valid_warp.size() != 3U) {
            throw std::runtime_error("prepared group must contain exactly three aligned images");
        }
        cv::Mat union_valid = cv::Mat::zeros(canvas_size, CV_8UC1);
        for (const cv::Mat& support : grouped_valid_warp) {
            cv::bitwise_or(union_valid, support, union_valid);
        }
        const int anchor_index = std::clamp(raw.group_anchor_index, 0, 2);
        warped_rgb_f = grouped_warped_rgb_f[static_cast<std::size_t>(anchor_index)].clone();
        valid_warp = union_valid.clone();
        // The anchor owns overlap RGB.  Side RGB is only allowed to fill a
        // pixel that the anchor does not observe; this makes the fused color
        // map single-source in all overlap regions.
        for (std::size_t source_index = 0; source_index < grouped_warped_rgb_f.size(); ++source_index) {
            if (static_cast<int>(source_index) == anchor_index) {
                continue;
            }
            for (int y = 0; y < warped_rgb_f.rows; ++y) {
                for (int x = 0; x < warped_rgb_f.cols; ++x) {
                    if (grouped_valid_warp[static_cast<std::size_t>(anchor_index)].at<std::uint8_t>(y, x) != 0U
                        || grouped_valid_warp[source_index].at<std::uint8_t>(y, x) == 0U) {
                        continue;
                    }
                    warped_rgb_f.at<cv::Vec3f>(y, x) =
                        grouped_warped_rgb_f[source_index].at<cv::Vec3f>(y, x);
                }
            }
        }
        const cv::Mat canvas_support = state.initialized
            ? state_mask(state.support, state.width, state.height)
            : cv::Mat::zeros(canvas_size, CV_8UC1);
        cv::Mat group_support_change;
        cv::bitwise_not(canvas_support, group_support_change);
        cv::bitwise_and(union_valid, group_support_change, group_support_change);
        if (state.initialized) {
            cv::bitwise_or(support_change, group_support_change, support_change);
        } else {
            support_change = union_valid.clone();
        }
        cv::bitwise_or(change_mask, support_change, change_mask);
        cv::bitwise_and(change_mask, valid_warp, change_mask);
    }
    result.metrics.changed_ratio = static_cast<double>(cv::countNonZero(change_mask))
        / static_cast<double>(change_mask.total());
    result.metrics.diff_ms = diff_timer.ms();
    result.frame.changed_ratio = static_cast<float>(result.metrics.changed_ratio);
    result.frame.align_error_px = static_cast<float>(result.metrics.homography_error_px);
    // A fallback homography is not trusted for a depth commit.  It may still
    // be useful for diagnostics, but publishing its geometry would create a
    // second sheet at exactly the frame boundary that the anchor ring is
    // intended to protect.
    if (state.initialized && !result.metrics.fallback.empty()) {
        change_mask.setTo(0);
        support_change.setTo(0);
        photometric_change.setTo(0);
        result.metrics.changed_ratio = 0.0;
        result.metrics.photometric_changed_ratio = 0.0;
        result.metrics.support_changed_ratio = 0.0;
    }

    const cv::Mat canvas_valid = state.initialized
        ? state_mask(state.valid, state.width, state.height)
        : cv::Mat::zeros(valid_warp.size(), CV_8UC1);
    const cv::Mat anchor_ring = state.initialized
        ? anchor_ring_mask(change_mask, canvas_valid, valid_warp)
        : cv::Mat::zeros(valid_warp.size(), CV_8UC1);
    const cv::Mat fusion_mask = filter_components(
        seam_fusion_mask(change_mask, valid_warp), 256);

    const bool skip_model = state.initialized && result.metrics.changed_ratio <= options_.no_change_ratio;
    result.metrics.skipped_model = skip_model;
    if (skip_model) {
        // Python updates the accumulated support before taking the no-change
        // fast path.  Keep that state change in the versioned C++ stream as
        // an observation-only patch; otherwise later ROIs are cropped from
        // stale support and the first visible geometry update is displaced.
        if (state.initialized && result.metrics.fallback.empty()) {
            CandidatePatch support_patch;
            support_patch.frame_seq = raw.frame_seq;
            support_patch.base_version = state.version;
            support_patch.width = state.width;
            support_patch.height = state.height;
            support_patch.changed_ratio = 0.0f;
            support_patch.initialize_canvas = false;
            support_patch.observed_slots.reserve(static_cast<std::size_t>(cv::countNonZero(valid_warp)));
            for (int y = 0; y < valid_warp.rows; ++y) {
                for (int x = 0; x < valid_warp.cols; ++x) {
                    if (valid_warp.at<std::uint8_t>(y, x) != 0U) {
                        support_patch.observed_slots.push_back(
                            static_cast<std::uint32_t>(y * state.width + x));
                    }
                }
            }
            if (!support_patch.observed_slots.empty()) {
                result.patch = std::move(support_patch);
                result.has_patch = true;
            }
        }
        result.frame.status = FrameStatus::NoChange;
        result.frame.changed_point_count = 0;
        result.frame.valid_point_count = static_cast<std::uint32_t>(
            std::count(state.valid.begin(), state.valid.end(), static_cast<std::uint8_t>(1U)));
        result.metrics.total_ms = total_timer.ms();
        save_debug_images(
            frame,
            warped_rgb_f,
            change_mask,
            valid_warp,
            anchor_ring,
            cv::Mat::zeros(change_mask.size(), CV_8UC1),
            cv::Mat::zeros(change_mask.size(), CV_8UC1),
            warped_rgb_f);
        return result;
    }

    // Match Python's model inputs: the first frame uses the unpadded matching
    // image, while every later frame uses an aligned anchor/current ROI.  The
    // ROI is expanded for context but is never written back as geometry.
    std::vector<cv::Mat> model_inputs;
    cv::Mat model_to_canvas = cv::Mat::eye(3, 3, CV_32FC1);
    cv::Mat model_support;
    if (prepared_group != nullptr) {
        cv::Rect roi(0, 0, state.width, state.height);
        int source_width = frame.match_rgb_f.cols;
        int source_height = frame.match_rgb_f.rows;
        if (!state.initialized) {
            const int pad_left = std::max(32, static_cast<int>(std::round(options_.width * 0.05)));
            const int pad_top = std::max(128, static_cast<int>(std::round(options_.width * 0.18)));
            roi = cv::Rect(
                std::clamp(pad_left, 0, std::max(0, state.width - 1)),
                std::clamp(pad_top, 0, std::max(0, state.height - 1)),
                std::min(source_width, state.width - std::clamp(pad_left, 0, std::max(0, state.width - 1))),
                std::min(source_height, state.height - std::clamp(pad_top, 0, std::max(0, state.height - 1))));
        } else if (cv::countNonZero(fusion_mask) > 0) {
            roi = cv::boundingRect(fusion_mask);
            const int context = 32;
            const int x0 = std::max(0, roi.x - context);
            const int y0 = std::max(0, roi.y - context);
            const int x1 = std::min(state.width, roi.x + roi.width + context);
            const int y1 = std::min(state.height, roi.y + roi.height + context);
            roi = cv::Rect(x0, y0, std::max(1, x1 - x0), std::max(1, y1 - y0));
        }
        const auto [target_width, target_height] = bucket_roi_size(
            roi.width, roi.height, options_.group_width, options_.group_height);
        const int model_width = options_.group_width;
        const int model_height = options_.group_height;
        const int pad_x = (model_width - target_width) / 2;
        const int pad_y = (model_height - target_height) / 2;
        result.metrics.roi_width = target_width;
        result.metrics.roi_height = target_height;
        result.metrics.model_input_width = model_width;
        result.metrics.model_input_height = model_height;
        for (const cv::Mat& group_image : grouped_warped_rgb_f) {
            cv::Rect safe_roi = roi & cv::Rect(0, 0, group_image.cols, group_image.rows);
            if (safe_roi.width <= 0 || safe_roi.height <= 0) {
                throw std::runtime_error("group ROI lies outside aligned canvas");
            }
            cv::Mat resized;
            cv::resize(
                group_image(safe_roi), resized,
                cv::Size(target_width, target_height), 0.0, 0.0, cv::INTER_AREA);
            cv::Mat padded;
            cv::copyMakeBorder(
                resized, padded,
                pad_y, model_height - target_height - pad_y,
                pad_x, model_width - target_width - pad_x,
                cv::BORDER_REPLICATE);
            model_inputs.push_back(std::move(padded));
        }
        const float scale_x = static_cast<float>(roi.width) / static_cast<float>(target_width);
        const float scale_y = static_cast<float>(roi.height) / static_cast<float>(target_height);
        model_to_canvas.at<float>(0, 0) = scale_x;
        model_to_canvas.at<float>(1, 1) = scale_y;
        model_to_canvas.at<float>(0, 2) = static_cast<float>(roi.x) - static_cast<float>(pad_x) * scale_x;
        model_to_canvas.at<float>(1, 2) = static_cast<float>(roi.y) - static_cast<float>(pad_y) * scale_y;
        model_support = cv::Mat::zeros(model_height, model_width, CV_32FC1);
        cv::Mat content_support = cv::Mat::ones(target_height, target_width, CV_32FC1);
        const int margin = (target_width > 18 && target_height > 18) ? 8 : 0;
        if (margin > 0) {
            content_support.rowRange(0, margin).setTo(0.0f);
            content_support.rowRange(target_height - margin, target_height).setTo(0.0f);
            content_support.colRange(0, margin).setTo(0.0f);
            content_support.colRange(target_width - margin, target_width).setTo(0.0f);
        }
        content_support.copyTo(model_support(cv::Rect(pad_x, pad_y, target_width, target_height)));
    } else if (!state.initialized) {
        const int first_width = options_.first_model_width;
        const int first_height = options_.first_model_height;
        cv::Mat first_model;
        cv::resize(
            frame.match_rgb_f,
            first_model,
            cv::Size(first_width, first_height),
            0.0,
            0.0,
            cv::INTER_AREA);
        model_inputs.push_back(std::move(first_model));
        result.metrics.roi_width = first_width;
        result.metrics.roi_height = first_height;
        result.metrics.model_input_width = first_width;
        result.metrics.model_input_height = first_height;
        const int pad_left = std::max(32, static_cast<int>(std::round(options_.width * 0.05)));
        const int pad_top = std::max(128, static_cast<int>(std::round(options_.width * 0.18)));
        model_to_canvas.at<float>(0, 0) = static_cast<float>(frame.match_rgb_u8.cols)
            / static_cast<float>(first_width);
        model_to_canvas.at<float>(1, 1) = static_cast<float>(frame.match_rgb_u8.rows)
            / static_cast<float>(first_height);
        model_to_canvas.at<float>(0, 2) = static_cast<float>(pad_left);
        model_to_canvas.at<float>(1, 2) = static_cast<float>(pad_top);
        model_support = cv::Mat::ones(first_height, first_width, CV_32FC1);
    } else {
        const cv::Mat canvas_rgb = !anchor_rgb_float_.empty()
            && anchor_rgb_float_.size() == cv::Size(state.width, state.height)
            ? anchor_rgb_float_
            : anchor_rgb_float(state);
        cv::Rect roi(0, 0, state.width, state.height);
        if (cv::countNonZero(fusion_mask) > 0) {
            roi = cv::boundingRect(fusion_mask);
            const int context = 32;
            const int x0 = std::max(0, roi.x - context);
            const int y0 = std::max(0, roi.y - context);
            const int x1 = std::min(state.width, roi.x + roi.width + context);
            const int y1 = std::min(state.height, roi.y + roi.height + context);
            roi = cv::Rect(x0, y0, std::max(1, x1 - x0), std::max(1, y1 - y0));
        }
        cv::Mat anchor_crop = canvas_rgb(roi).clone();
        cv::Mat current_crop = warped_rgb_f(roi).clone();
        const auto [target_width, target_height] = bucket_roi_size(
            roi.width, roi.height, options_.width, options_.height);
        const auto [bucket_width, bucket_height] = dynamic_pair_shape_for_target(
            {target_width, target_height});
        cv::Mat anchor_roi;
        cv::Mat current_roi;
        // Keep the Python float path: resize [0,1] RGB directly instead of
        // quantizing the ROI to uint8 before the model sees it.
        cv::resize(anchor_crop, anchor_roi, cv::Size(target_width, target_height), 0.0, 0.0, cv::INTER_AREA);
        cv::resize(current_crop, current_roi, cv::Size(target_width, target_height), 0.0, 0.0, cv::INTER_AREA);
        // Python keeps the crop aspect ratio.  In strict mode the TorchScript
        // graph is traced once per exact bucket; in fast letterbox mode the
        // same content rectangle is copied into the fixed 700x700 graph with
        // replicated edge pixels.  Mapping and support are defined from the
        // content rectangle, so the padding cannot become a second surface.
        const int model_width = options_.pair_letterbox ? options_.width : bucket_width;
        const int model_height = options_.pair_letterbox ? options_.height : bucket_height;
        result.metrics.roi_width = target_width;
        result.metrics.roi_height = target_height;
        result.metrics.model_input_width = model_width;
        result.metrics.model_input_height = model_height;
        const bool needs_padding = options_.pair_letterbox
            || model_width != target_width || model_height != target_height;
        const int pad_x = needs_padding ? (model_width - target_width) / 2 : 0;
        const int pad_y = needs_padding ? (model_height - target_height) / 2 : 0;
        if (model_width < target_width || model_height < target_height) {
            throw std::runtime_error("letterbox model bucket is smaller than the ROI bucket");
        }
        if (needs_padding) {
            cv::Mat anchor_padded;
            cv::Mat current_padded;
            cv::copyMakeBorder(
                anchor_roi,
                anchor_padded,
                pad_y,
                model_height - target_height - pad_y,
                pad_x,
                model_width - target_width - pad_x,
                cv::BORDER_REPLICATE);
            cv::copyMakeBorder(
                current_roi,
                current_padded,
                pad_y,
                model_height - target_height - pad_y,
                pad_x,
                model_width - target_width - pad_x,
                cv::BORDER_REPLICATE);
            model_inputs.push_back(std::move(anchor_padded));
            model_inputs.push_back(std::move(current_padded));
        } else {
            model_inputs.push_back(std::move(anchor_roi));
            model_inputs.push_back(std::move(current_roi));
        }
        const float scale_x = static_cast<float>(roi.width) / static_cast<float>(target_width);
        const float scale_y = static_cast<float>(roi.height) / static_cast<float>(target_height);
        model_to_canvas.at<float>(0, 0) = scale_x;
        model_to_canvas.at<float>(1, 1) = scale_y;
        model_to_canvas.at<float>(0, 2) = static_cast<float>(roi.x) - static_cast<float>(pad_x) * scale_x;
        model_to_canvas.at<float>(1, 2) = static_cast<float>(roi.y) - static_cast<float>(pad_y) * scale_y;
        model_support = cv::Mat::zeros(model_height, model_width, CV_32FC1);
        cv::Mat content_support = cv::Mat::ones(target_height, target_width, CV_32FC1);
        // Python's _model_roi_support removes an 8-pixel model-space border
        // explicitly.  In letterbox mode the margin is in the unpadded
        // content rectangle, not in the replicated border.
        const int margin = (target_width > 18 && target_height > 18) ? 8 : 0;
        if (margin > 0) {
            content_support.rowRange(0, margin).setTo(0.0f);
            content_support.rowRange(target_height - margin, target_height).setTo(0.0f);
            content_support.colRange(0, margin).setTo(0.0f);
            content_support.colRange(target_width - margin, target_width).setTo(0.0f);
        }
        content_support.copyTo(model_support(cv::Rect(pad_x, pad_y, target_width, target_height)));
    }

    Timer model_timer;
    const int output_frame_index = state.initialized ? 1 : 0;
    Prediction prediction;
    if (prepared_group != nullptr) {
        const std::vector<Prediction> predictions = run_group_model(model_inputs);
        prediction = fuse_group_predictions(
            predictions,
            std::clamp(raw.group_anchor_index, 0, 2),
            result.metrics);
        result.metrics.forward_calls = 1;
        result.metrics.forward_batch_size = 3;
        result.metrics.forward_sequence_size = 1;
        result.metrics.group_size = 3;
    } else {
        prediction = run_model(model_inputs, output_frame_index);
        result.metrics.forward_calls = 1;
        result.metrics.forward_batch_size = 1;
        result.metrics.forward_sequence_size = static_cast<int>(model_inputs.size());
    }
    result.metrics.model_ms = model_timer.ms();
    if (!state.initialized) {
        release_single_model_after_first_frame();
    }

    const cv::Size canvas_size(state.width, state.height);
    const cv::Mat model_depth_canvas = warp_like(
        prediction.depth, model_to_canvas, canvas_size, cv::INTER_LINEAR);
    const cv::Mat model_confidence_canvas = warp_like(
        prediction.confidence, model_to_canvas, canvas_size, cv::INTER_LINEAR);
    const cv::Mat warped_roi_valid_f = warp_like(
        model_support, model_to_canvas, canvas_size, cv::INTER_NEAREST);
    cv::Mat warped_roi_valid;
    cv::threshold(warped_roi_valid_f, warped_roi_valid, 0.5, 255.0, cv::THRESH_BINARY);
    warped_roi_valid.convertTo(warped_roi_valid, CV_8UC1);

    cv::Mat candidate_valid = model_valid_mask(
        model_depth_canvas,
        model_confidence_canvas,
        valid_warp,
        static_cast<float>(options_.min_conf));
    cv::bitwise_and(candidate_valid, warped_roi_valid, candidate_valid);

    // Python computes a frame-local confidence percentile after projecting
    // the ROI back to the canvas.  A fixed 0.1 threshold lets low-confidence
    // crop borders write geometry in C++, which is another direct source of
    // the visible discontinuous height band.
    cv::Mat quality_valid = candidate_valid.clone();
    float confidence_threshold = static_cast<float>(options_.min_conf);
    if (state.initialized) {
        std::vector<float> confidence_values;
        confidence_values.reserve(static_cast<std::size_t>(cv::countNonZero(candidate_valid)));
        for (int y = 0; y < candidate_valid.rows; ++y) {
            for (int x = 0; x < candidate_valid.cols; ++x) {
                if (candidate_valid.at<std::uint8_t>(y, x) != 0U) {
                    confidence_values.push_back(model_confidence_canvas.at<float>(y, x));
                }
            }
        }
        if (confidence_values.size() >= 256U) {
            std::sort(confidence_values.begin(), confidence_values.end());
            const double position = 0.20 * static_cast<double>(confidence_values.size() - 1U);
            const std::size_t lower = static_cast<std::size_t>(std::floor(position));
            const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
            const float fraction = static_cast<float>(position - static_cast<double>(lower));
            const float percentile = confidence_values[lower]
                + (confidence_values[upper] - confidence_values[lower]) * fraction;
            confidence_threshold = std::max(
                confidence_threshold, percentile);
            for (int y = 0; y < quality_valid.rows; ++y) {
                for (int x = 0; x < quality_valid.cols; ++x) {
                    if (quality_valid.at<std::uint8_t>(y, x) != 0U
                        && model_confidence_canvas.at<float>(y, x) < confidence_threshold) {
                        quality_valid.at<std::uint8_t>(y, x) = 0U;
                    }
                }
            }
        }
    }

    cv::Mat commit_confidence = model_confidence_canvas.clone();
    if (state.initialized) {
        cv::Mat distance;
        cv::distanceTransform(candidate_valid, distance, cv::DIST_L2, 3);
        std::vector<float> distances;
        distances.reserve(static_cast<std::size_t>(cv::countNonZero(candidate_valid)));
        for (int y = 0; y < candidate_valid.rows; ++y) {
            for (int x = 0; x < candidate_valid.cols; ++x) {
                if (candidate_valid.at<std::uint8_t>(y, x) != 0U) {
                    distances.push_back(distance.at<float>(y, x));
                }
            }
        }
        const float p85 = distances.empty() ? 1.0f : [&]() {
            std::sort(distances.begin(), distances.end());
            const double position = 0.85 * static_cast<double>(distances.size() - 1U);
            const std::size_t lower = static_cast<std::size_t>(std::floor(position));
            const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
            const float fraction = static_cast<float>(position - static_cast<double>(lower));
            return distances[lower] + (distances[upper] - distances[lower]) * fraction;
        }();
        const float feather_scale = std::max(1.0f, p85 * 0.35f);
        for (int y = 0; y < commit_confidence.rows; ++y) {
            for (int x = 0; x < commit_confidence.cols; ++x) {
                if (candidate_valid.at<std::uint8_t>(y, x) == 0U) {
                    commit_confidence.at<float>(y, x) = 0.0f;
                    continue;
                }
                const float feather = std::clamp(
                    distance.at<float>(y, x) / feather_scale, 0.0f, 1.0f);
                commit_confidence.at<float>(y, x) = std::max(
                    model_confidence_canvas.at<float>(y, x) * feather,
                    static_cast<float>(options_.min_conf));
            }
        }
    }
    cv::Mat model_valid = candidate_valid.clone();
    Timer depth_timer;
    cv::Mat aligned_depth = align_depth_to_canvas(
        model_depth_canvas,
        model_confidence_canvas,
        model_valid,
        state,
        [&]() {
            cv::Mat gated_anchor;
            cv::bitwise_and(anchor_ring, quality_valid, gated_anchor);
            return gated_anchor;
        }());
    result.metrics.depth_align_ms = depth_timer.ms();

    Timer patch_timer;
    CandidatePatch patch;
    patch.frame_seq = raw.frame_seq;
    patch.base_version = state.version;
    patch.width = state.width;
    patch.height = state.height;
    patch.changed_ratio = static_cast<float>(result.metrics.changed_ratio);
    patch.scene_jump = result.metrics.changed_ratio >= options_.scene_jump_ratio;
    patch.initialize_canvas = !state.initialized;
    patch.anchor_camera.fx = (static_cast<float>(state.width) * 0.5f)
        / std::tan(prediction.fov_w * 0.5f);
    patch.anchor_camera.fy = (static_cast<float>(state.height) * 0.5f)
        / std::tan(prediction.fov_h * 0.5f);
    patch.anchor_camera.cx = static_cast<float>(state.width) * 0.5f;
    patch.anchor_camera.cy = static_cast<float>(state.height) * 0.5f;
    if (patch.initialize_canvas) {
        patch.anchor_rgba = pack_image(frame.rgb_u8);
    }

    cv::Mat not_support;
    cv::bitwise_not(support_change, not_support);
    cv::Mat photo_only_existing;
    cv::bitwise_and(photometric_change, not_support, photo_only_existing);
    if (state.initialized) {
        cv::bitwise_and(photo_only_existing, canvas_valid, photo_only_existing);
    }

    cv::Mat update_mask;
    cv::bitwise_and(model_valid, change_mask, update_mask);
    cv::Mat not_photo_only;
    cv::bitwise_not(photo_only_existing, not_photo_only);
    cv::bitwise_and(update_mask, not_photo_only, update_mask);
    cv::Mat not_anchor;
    cv::bitwise_not(anchor_ring, not_anchor);
    cv::bitwise_and(update_mask, not_anchor, update_mask);

    // RGB-only bridge: keep the current aligned source authoritative through
    // the old overlap, then fade to the untouched old canvas only in the last
    // few pixels before the anchor ring.  The bridge never changes depth.
    cv::Mat color_bridge_mask = cv::Mat::zeros(update_mask.size(), CV_8UC1);
    cv::Mat color_bridge_mix = cv::Mat::zeros(update_mask.size(), CV_32FC1);
    cv::Mat canvas_rgb;
    if (state.initialized && cv::countNonZero(support_change) > 0
        && cv::countNonZero(anchor_ring) > 0) {
        canvas_rgb = !live_rgb_float_.empty()
            && live_rgb_float_.size() == cv::Size(state.width, state.height)
            ? live_rgb_float_
            : state_rgb_float(state);
        const cv::Mat bridge_kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, cv::Size(65, 65));
        cv::Mat expanded_support;
        cv::dilate(support_change, expanded_support, bridge_kernel);
        cv::bitwise_and(expanded_support, canvas_valid, expanded_support);
        cv::bitwise_and(expanded_support, valid_warp, expanded_support);
        cv::bitwise_and(expanded_support, not_anchor, expanded_support);
        cv::Mat not_update;
        cv::bitwise_not(update_mask, not_update);
        cv::bitwise_and(expanded_support, not_update, color_bridge_mask);

        cv::Mat non_anchor;
        cv::bitwise_not(anchor_ring, non_anchor);
        cv::Mat distance_to_ring;
        cv::distanceTransform(non_anchor, distance_to_ring, cv::DIST_L2, 3);
        for (int y = 0; y < color_bridge_mask.rows; ++y) {
            for (int x = 0; x < color_bridge_mask.cols; ++x) {
                if (color_bridge_mask.at<std::uint8_t>(y, x) != 0U) {
                    color_bridge_mix.at<float>(y, x) = std::clamp(
                        distance_to_ring.at<float>(y, x) / 8.0f, 0.0f, 1.0f);
                }
            }
        }
    } else if (state.initialized) {
        canvas_rgb = !live_rgb_float_.empty()
            && live_rgb_float_.size() == cv::Size(state.width, state.height)
            ? live_rgb_float_
            : state_rgb_float(state);
    }

    cv::Mat color_apply_mask;
    cv::bitwise_or(update_mask, color_bridge_mask, color_apply_mask);
    cv::Mat fused_rgb = state.initialized
        ? anchor_texture_transfer(
            warped_rgb_f,
            canvas_rgb,
            canvas_valid,
            valid_warp,
            color_apply_mask,
            support_change,
            anchor_ring)
        : warped_rgb_f.clone();

    if (state.initialized && cv::countNonZero(color_bridge_mask) > 0) {
        for (int y = 0; y < color_bridge_mask.rows; ++y) {
            for (int x = 0; x < color_bridge_mask.cols; ++x) {
                if (color_bridge_mask.at<std::uint8_t>(y, x) == 0U) {
                    continue;
                }
                const float alpha = color_bridge_mix.at<float>(y, x);
                fused_rgb.at<cv::Vec3f>(y, x) =
                    fused_rgb.at<cv::Vec3f>(y, x) * alpha
                    + canvas_rgb.at<cv::Vec3f>(y, x) * (1.0f - alpha);
            }
        }
    }

    const cv::Mat continuity_mask = [&]() {
        cv::Mat mask;
        cv::bitwise_and(update_mask, support_change, mask);
        return mask;
    }();
    if (state.initialized && cv::countNonZero(continuity_mask) > 0) {
        cv::Mat canvas_depth(state.height, state.width, CV_32FC1);
        for (int y = 0; y < state.height; ++y) {
            for (int x = 0; x < state.width; ++x) {
                canvas_depth.at<float>(y, x) = state.depth[static_cast<std::size_t>(y) * state.width + x];
            }
        }
        aligned_depth = anchor_depth_continuity(
            aligned_depth,
            canvas_depth,
            canvas_valid,
            continuity_mask,
            anchor_ring);
    }

    cv::Mat fused_u8;
    fused_rgb.convertTo(fused_u8, CV_8UC3, 255.0);

    const std::size_t slot_count = static_cast<std::size_t>(state.width) * static_cast<std::size_t>(state.height);
    patch.observed_slots.reserve(slot_count / 2U);
    for (int y = 0; y < state.height; ++y) {
        for (int x = 0; x < state.width; ++x) {
            if (valid_warp.at<std::uint8_t>(y, x) != 0U) {
                patch.observed_slots.push_back(static_cast<std::uint32_t>(y * state.width + x));
            }
            const bool geometry_update = update_mask.at<std::uint8_t>(y, x) != 0U;
            const bool color_bridge = color_bridge_mask.at<std::uint8_t>(y, x) != 0U;
            if (!geometry_update && !color_bridge) {
                continue;
            }
            SlotValue after;
            const std::size_t index = static_cast<std::size_t>(y) * state.width + x;
            if (geometry_update) {
                const float depth = aligned_depth.at<float>(y, x);
                const float confidence = commit_confidence.at<float>(y, x);
                if (!std::isfinite(depth) || !std::isfinite(confidence)
                    || depth <= 1e-6f || confidence < static_cast<float>(options_.min_conf)) {
                    continue;
                }
                after.depth = depth;
                after.confidence = confidence;
                after.last_update_frame = static_cast<std::uint32_t>(raw.frame_seq);
                after.valid = 1U;
            } else {
                // A bridge cell already has valid geometry.  Only its RGB is
                // replaced, so replay cannot introduce a duplicate depth layer.
                if (!state.initialized || state.valid[index] == 0U) {
                    continue;
                }
                after.depth = state.depth[index];
                after.confidence = state.confidence[index];
                after.last_update_frame = state.last_update_frame[index];
                after.valid = state.valid[index];
            }
            after.rgba = color_at(fused_u8, x, y);
            patch.updates.push_back(SlotUpdate{static_cast<std::uint32_t>(y * state.width + x), after});
        }
    }
    if (!state.initialized) {
        live_rgb_float_ = frame.rgb_f.clone();
    } else if (!live_rgb_float_.empty()
        && live_rgb_float_.size() == cv::Size(state.width, state.height)) {
        // Keep the same float canvas that Python retains between pushes.  The
        // versioned protocol stores the committed byte color separately, so
        // this buffer is only an in-process inference/alignment reference.
        for (const SlotUpdate& update : patch.updates) {
            const int x = static_cast<int>(update.slot_id % static_cast<std::uint32_t>(state.width));
            const int y = static_cast<int>(update.slot_id / static_cast<std::uint32_t>(state.width));
            if (x >= 0 && x < state.width && y >= 0 && y < state.height) {
                live_rgb_float_.at<cv::Vec3f>(y, x) = fused_rgb.at<cv::Vec3f>(y, x);
            }
        }
    }
    result.metrics.patch_ms = patch_timer.ms();
    result.metrics.changed_point_count = static_cast<std::uint32_t>(patch.updates.size());
    std::size_t valid_point_count = static_cast<std::size_t>(
        std::count(state.valid.begin(), state.valid.end(), static_cast<std::uint8_t>(1U)));
    for (const SlotUpdate& update : patch.updates) {
        if (update.slot_id >= state.valid.size() || state.valid[update.slot_id] == 0U) {
            ++valid_point_count;
        }
    }
    result.metrics.valid_point_count = static_cast<std::uint32_t>(valid_point_count);
    result.frame.status = FrameStatus::Committed;
    result.frame.changed_point_count = result.metrics.changed_point_count;
    result.frame.valid_point_count = result.metrics.valid_point_count;
    result.patch = std::move(patch);
    result.has_patch = true;
    result.metrics.total_ms = total_timer.ms();
    save_debug_images(
        frame,
        warped_rgb_f,
        change_mask,
        valid_warp,
        anchor_ring,
        update_mask,
        color_bridge_mask,
        fused_rgb);
    return result;
}

CandidateCommit InferenceEngine::process_group(
    const RawFrame& raw,
    const CanvasState& state) {
    if (raw.group_paths.size() != 3U) {
        throw std::runtime_error("three-image mode received an incomplete input group");
    }
    Timer read_timer;
    std::vector<FrameImage> frames;
    frames.reserve(3U);
    for (const auto& path : raw.group_paths) {
        frames.push_back(load_frame(path));
    }
    const int anchor_index = std::clamp(raw.group_anchor_index, 0, 2);
    PreparedGroup prepared;
    prepared.warped_rgb_f.reserve(3U);
    prepared.valid_warp.reserve(3U);
    const int pad_left = std::max(32, static_cast<int>(std::round(options_.width * 0.05)));
    const int pad_top = std::max(128, static_cast<int>(std::round(options_.width * 0.18)));
    const cv::Mat pad_translation = [&]() {
        cv::Mat result = cv::Mat::eye(3, 3, CV_32FC1);
        result.at<float>(0, 2) = static_cast<float>(pad_left);
        result.at<float>(1, 2) = static_cast<float>(pad_top);
        return result;
    }();
    const cv::Mat inverse_pad_translation = [&]() {
        cv::Mat result = cv::Mat::eye(3, 3, CV_32FC1);
        result.at<float>(0, 2) = -static_cast<float>(pad_left);
        result.at<float>(1, 2) = -static_cast<float>(pad_top);
        return result;
    }();
    for (int index = 0; index < 3; ++index) {
        cv::Mat aligned_rgb = frames[static_cast<std::size_t>(index)].rgb_f.clone();
        cv::Mat aligned_support = frames[static_cast<std::size_t>(index)].support.clone();
        if (index != anchor_index) {
            // Feature coordinates are measured on the unpadded matching
            // image.  Conjugating by the fixed canvas pad applies the same
            // transform to the padded RGB/support buffers.
            const cv::Mat match_homography = estimate_pair_homography(
                frames[static_cast<std::size_t>(index)],
                frames[static_cast<std::size_t>(anchor_index)]);
            const cv::Mat canvas_homography = pad_translation
                * match_homography * inverse_pad_translation;
            aligned_rgb = warp_like(
                frames[static_cast<std::size_t>(index)].rgb_f,
                canvas_homography,
                cv::Size(options_.canvas_width, options_.canvas_height),
                cv::INTER_LINEAR);
            aligned_support = warp_like(
                frames[static_cast<std::size_t>(index)].support,
                canvas_homography,
                cv::Size(options_.canvas_width, options_.canvas_height),
                cv::INTER_NEAREST);
            cv::threshold(aligned_support, aligned_support, 127.0, 255.0, cv::THRESH_BINARY);
            aligned_support.convertTo(aligned_support, CV_8UC1);
        }
        prepared.warped_rgb_f.push_back(std::move(aligned_rgb));
        prepared.valid_warp.push_back(std::move(aligned_support));
    }
    prepared.fused_rgb_f = prepared.warped_rgb_f[static_cast<std::size_t>(anchor_index)].clone();
    prepared.union_valid = cv::Mat::zeros(
        options_.canvas_height, options_.canvas_width, CV_8UC1);
    for (const cv::Mat& support : prepared.valid_warp) {
        cv::bitwise_or(prepared.union_valid, support, prepared.union_valid);
    }
    return process_impl(
        raw,
        state,
        frames[static_cast<std::size_t>(anchor_index)],
        &prepared,
        read_timer.ms());
}

}  // namespace omnivggt::observer
