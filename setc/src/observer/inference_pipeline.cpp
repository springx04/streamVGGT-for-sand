#include "inference_pipeline.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <ATen/autocast_mode.h>
#include <c10/core/InferenceMode.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

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

cv::Point content_origin(
    const cv::Mat& support,
    const int fallback_x,
    const int fallback_y) {
    if (!support.empty() && cv::countNonZero(support) > 0) {
        return cv::boundingRect(support).tl();
    }
    return cv::Point(fallback_x, fallback_y);
}

bool plausible_planar_homography(
    const cv::Mat& homography,
    const cv::Size source_size,
    const cv::Size target_size) {
    if (homography.empty() || source_size.width <= 1 || source_size.height <= 1
        || target_size.width <= 1 || target_size.height <= 1
        || !cv::checkRange(homography)) {
        return false;
    }
    std::vector<cv::Point2f> corners{
        {0.0f, 0.0f},
        {static_cast<float>(source_size.width - 1), 0.0f},
        {static_cast<float>(source_size.width - 1),
            static_cast<float>(source_size.height - 1)},
        {0.0f, static_cast<float>(source_size.height - 1)}};
    std::vector<cv::Point2f> projected;
    cv::perspectiveTransform(corners, projected, homography);
    if (projected.size() != 4U) {
        return false;
    }
    cv::Point2f center(0.0f, 0.0f);
    for (const cv::Point2f& point : projected) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
            return false;
        }
        center += point;
    }
    center *= 0.25f;
    const double source_area = static_cast<double>(source_size.area());
    const double target_area = static_cast<double>(target_size.area());
    const double projected_area = std::abs(cv::contourArea(projected));
    if (!cv::isContourConvex(projected)
        || projected_area < 0.08 * source_area
        || projected_area > 12.0 * target_area
        || center.x < -2.0f * target_size.width
        || center.x > 3.0f * target_size.width
        || center.y < -2.0f * target_size.height
        || center.y > 3.0f * target_size.height) {
        return false;
    }
    return true;
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
    // Avoid the minutes-long global graph optimizer pass on the first CUDA forward.
    torch::jit::setGraphExecutorOptimize(false);
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

cv::Mat bright_nonplanar_mask(const cv::Mat& rgb_u8) {
    cv::Mat mask(rgb_u8.rows, rgb_u8.cols, CV_8UC1, cv::Scalar(0));
    if (rgb_u8.empty() || rgb_u8.type() != CV_8UC3) {
        return mask;
    }
    for (int y = 0; y < rgb_u8.rows; ++y) {
        for (int x = 0; x < rgb_u8.cols; ++x) {
            const cv::Vec3b color = rgb_u8.at<cv::Vec3b>(y, x);
            const int minimum = std::min({
                static_cast<int>(color[0]),
                static_cast<int>(color[1]),
                static_cast<int>(color[2])});
            const int maximum = std::max({
                static_cast<int>(color[0]),
                static_cast<int>(color[1]),
                static_cast<int>(color[2])});
            // The robot is the large bright, nearly achromatic foreground in
            // all three real views. Keep coloured green floor reflections:
            // they are useful planar features and must remain reconstructable.
            if (minimum >= 142 && maximum - minimum <= 72) {
                mask.at<std::uint8_t>(y, x) = 255U;
            }
        }
    }
    cv::morphologyEx(
        mask,
        mask,
        cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11)));
    cv::dilate(
        mask,
        mask,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(25, 25)));
    return mask;
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

void fit_quadratic_surface(
    const std::vector<float>& x,
    const std::vector<float>& y,
    const std::vector<float>& z,
    const std::vector<std::uint8_t>& keep,
    std::array<double, 6>& coefficients) {
    cv::Mat normal = cv::Mat::zeros(6, 6, CV_64F);
    cv::Mat rhs = cv::Mat::zeros(6, 1, CV_64F);
    for (std::size_t index = 0; index < z.size(); ++index) {
        if (keep[index] == 0U) {
            continue;
        }
        const double basis[6] = {
            x[index], y[index],
            static_cast<double>(x[index]) * x[index],
            static_cast<double>(y[index]) * y[index],
            static_cast<double>(x[index]) * y[index],
            1.0};
        for (int row = 0; row < 6; ++row) {
            rhs.at<double>(row, 0) += basis[row] * static_cast<double>(z[index]);
            for (int column = 0; column < 6; ++column) {
                normal.at<double>(row, column) += basis[row] * basis[column];
            }
        }
    }
    cv::Mat solution;
    if (!cv::solve(normal, rhs, solution, cv::DECOMP_SVD) || solution.rows != 6) {
        coefficients.fill(0.0);
        return;
    }
    for (int index = 0; index < 6; ++index) {
        coefficients[static_cast<std::size_t>(index)] = solution.at<double>(index, 0);
    }
}

double quadratic_surface_value(
    const std::array<double, 6>& coefficients,
    const float x,
    const float y) {
    return coefficients[0] * x + coefficients[1] * y
        + coefficients[2] * static_cast<double>(x) * x
        + coefficients[3] * static_cast<double>(y) * y
        + coefficients[4] * static_cast<double>(x) * y
        + coefficients[5];
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

void interpolate_group_gap_scalar(
    cv::Mat& values,
    const cv::Mat& fill,
    const cv::Mat& source_mask) {
    if (values.empty() || fill.empty() || source_mask.empty()
        || cv::countNonZero(fill) == 0 || cv::countNonZero(source_mask) == 0) {
        return;
    }
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int component_count = cv::connectedComponentsWithStats(
        fill, labels, stats, centroids, 8, CV_32S);
    for (int component = 1; component < component_count; ++component) {
        const int left = stats.at<int>(component, cv::CC_STAT_LEFT);
        const int top = stats.at<int>(component, cv::CC_STAT_TOP);
        const int width = stats.at<int>(component, cv::CC_STAT_WIDTH);
        const int height = stats.at<int>(component, cv::CC_STAT_HEIGHT);
        const bool horizontal = width >= height;
        const int first = horizontal ? left : top;
        const int last = horizontal ? left + width : top + height;
        int before_probe = horizontal ? top - 1 : left - 1;
        int after_probe = horizontal ? top + height : left + width;
        const int before_limit = 0;
        const int after_limit = horizontal ? values.rows - 1 : values.cols - 1;
        const auto coverage = [&](const int coordinate) {
            int count = 0;
            for (int offset = 0; offset < last - first; ++offset) {
                const int along = first + offset;
                const int x = horizontal ? along : coordinate;
                const int y = horizontal ? coordinate : along;
                count += source_mask.at<std::uint8_t>(y, x) != 0U ? 1 : 0;
            }
            return count;
        };
        while (before_probe >= before_limit && coverage(before_probe) < (last - first) / 2) {
            --before_probe;
        }
        while (after_probe <= after_limit && coverage(after_probe) < (last - first) / 2) {
            ++after_probe;
        }
        auto row_mean = [&](const int coordinate, float& mean) {
            double sum = 0.0;
            int count = 0;
            for (int offset = 0; offset < last - first; ++offset) {
                const int along = first + offset;
                const int x = horizontal ? along : coordinate;
                const int y = horizontal ? coordinate : along;
                if (source_mask.at<std::uint8_t>(y, x) == 0U
                    || !std::isfinite(values.at<float>(y, x))) {
                    continue;
                }
                sum += values.at<float>(y, x);
                ++count;
            }
            if (count == 0) {
                return false;
            }
            mean = static_cast<float>(sum / static_cast<double>(count));
            return true;
        };
        float before_mean = 0.0f;
        float after_mean = 0.0f;
        const bool have_before_mean = before_probe >= before_limit
            && row_mean(before_probe, before_mean);
        const bool have_after_mean = after_probe <= after_limit
            && row_mean(after_probe, after_mean);
        for (int y = top; y < top + height; ++y) {
            for (int x = left; x < left + width; ++x) {
                if (labels.at<int>(y, x) != component) {
                    continue;
                }
                int before = horizontal ? y - 1 : x - 1;
                int after = horizontal ? y + 1 : x + 1;
                const int limit_before = horizontal ? 0 : 0;
                const int limit_after = horizontal
                    ? values.rows - 1
                    : values.cols - 1;
                while (before >= limit_before) {
                    const int sx = horizontal ? x : before;
                    const int sy = horizontal ? before : y;
                    if (source_mask.at<std::uint8_t>(sy, sx) != 0U
                        && std::isfinite(values.at<float>(sy, sx))) {
                        break;
                    }
                    --before;
                }
                while (after <= limit_after) {
                    const int sx = horizontal ? x : after;
                    const int sy = horizontal ? after : y;
                    if (source_mask.at<std::uint8_t>(sy, sx) != 0U
                        && std::isfinite(values.at<float>(sy, sx))) {
                        break;
                    }
                    ++after;
                }
                if (before < limit_before || after > limit_after) {
                    if (have_before_mean && have_after_mean) {
                        const float alpha = horizontal
                            ? static_cast<float>(y - top)
                                / static_cast<float>(std::max(1, height - 1))
                            : static_cast<float>(x - left)
                                / static_cast<float>(std::max(1, width - 1));
                        values.at<float>(y, x) = before_mean * (1.0f - alpha)
                            + after_mean * alpha;
                    } else if (have_before_mean) {
                        values.at<float>(y, x) = before_mean;
                    } else if (have_after_mean) {
                        values.at<float>(y, x) = after_mean;
                    }
                    continue;
                }
                const int before_x = horizontal ? x : before;
                const int before_y = horizontal ? before : y;
                const int after_x = horizontal ? x : after;
                const int after_y = horizontal ? after : y;
                const float denominator = static_cast<float>(
                    (horizontal ? after_y - before_y : after_x - before_x));
                if (denominator <= 0.0f) {
                    continue;
                }
                const float numerator = static_cast<float>(
                    (horizontal ? y - before_y : x - before_x));
                const float alpha = std::clamp(numerator / denominator, 0.0f, 1.0f);
                values.at<float>(y, x) = values.at<float>(before_y, before_x)
                    * (1.0f - alpha)
                    + values.at<float>(after_y, after_x) * alpha;
            }
        }
    }
}

// A grouped model can predict a valid strip while the previous canvas has a
// one-sided support boundary immediately next to it.  Nearest/model-only fill
// then leaves the strip on the model's depth layer and creates a visible step
// against the already committed canvas.  Interpolate the repaired strip from
// the nearest finite model sample on one side to the nearest committed canvas
// sample on the other side.  The operation is restricted to group_gap_fill;
// ordinary grouped and single-image geometry is untouched.
void interpolate_group_gap_depth_to_canvas(
    cv::Mat& depth,
    const cv::Mat& fill,
    const cv::Mat& canvas_depth,
    const cv::Mat& canvas_valid) {
    if (depth.empty() || fill.empty() || canvas_depth.empty() || canvas_valid.empty()
        || cv::countNonZero(fill) == 0 || cv::countNonZero(canvas_valid) == 0) {
        return;
    }
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int component_count = cv::connectedComponentsWithStats(
        fill, labels, stats, centroids, 8, CV_32S);
    for (int component = 1; component < component_count; ++component) {
        const int left = stats.at<int>(component, cv::CC_STAT_LEFT);
        const int top = stats.at<int>(component, cv::CC_STAT_TOP);
        const int width = stats.at<int>(component, cv::CC_STAT_WIDTH);
        const int height = stats.at<int>(component, cv::CC_STAT_HEIGHT);
        const bool horizontal = width >= height;
        const int first = horizontal ? left : top;
        const int last = horizontal ? left + width : top + height;
        const int before_start = horizontal ? top - 1 : left - 1;
        const int after_start = horizontal ? top + height : left + width;
        const int before_limit = 0;
        const int after_limit = horizontal ? depth.rows - 1 : depth.cols - 1;
        for (int along = first; along < last; ++along) {
            const int x = horizontal ? along : 0;
            const int y = horizontal ? 0 : along;
            int before = before_start;
            bool have_before = false;
            while (before >= before_limit) {
                const int sx = horizontal ? x : before;
                const int sy = horizontal ? before : y;
                if (std::isfinite(depth.at<float>(sy, sx))
                    && depth.at<float>(sy, sx) > 1e-6f) {
                    have_before = true;
                    break;
                }
                --before;
            }

            int after = after_start;
            bool have_after = false;
            bool after_from_canvas = false;
            while (after <= after_limit) {
                const int sx = horizontal ? x : after;
                const int sy = horizontal ? after : y;
                if (canvas_valid.at<std::uint8_t>(sy, sx) != 0U
                    && std::isfinite(canvas_depth.at<float>(sy, sx))
                    && canvas_depth.at<float>(sy, sx) > 1e-6f) {
                    have_after = true;
                    after_from_canvas = true;
                    break;
                }
                if (!after_from_canvas
                    && std::isfinite(depth.at<float>(sy, sx))
                    && depth.at<float>(sy, sx) > 1e-6f) {
                    have_after = true;
                    break;
                }
                ++after;
            }
            if (!have_before && !have_after) {
                continue;
            }
            const int cross_first = horizontal ? top : left;
            const int cross_last = horizontal ? top + height : left + width;
            for (int coordinate = cross_first; coordinate < cross_last; ++coordinate) {
                const int px = horizontal ? along : coordinate;
                const int py = horizontal ? coordinate : along;
                if (labels.at<int>(py, px) != component) {
                    continue;
                }
                float value = depth.at<float>(py, px);
                if (have_before && have_after && after > before) {
                    const int before_x = horizontal ? x : before;
                    const int before_y = horizontal ? before : y;
                    const int after_x = horizontal ? x : after;
                    const int after_y = horizontal ? after : y;
                    const float before_value = depth.at<float>(before_y, before_x);
                    const float after_value = after_from_canvas
                        ? canvas_depth.at<float>(after_y, after_x)
                        : depth.at<float>(after_y, after_x);
                    const float alpha = std::clamp(
                        static_cast<float>(coordinate - before)
                            / static_cast<float>(after - before),
                        0.0f,
                        1.0f);
                    value = before_value * (1.0f - alpha) + after_value * alpha;
                } else if (have_before) {
                    const int before_x = horizontal ? x : before;
                    const int before_y = horizontal ? before : y;
                    value = depth.at<float>(before_y, before_x);
                } else if (have_after) {
                    const int after_x = horizontal ? x : after;
                    const int after_y = horizontal ? after : y;
                    value = after_from_canvas
                        ? canvas_depth.at<float>(after_y, after_x)
                        : depth.at<float>(after_y, after_x);
                }
                if (std::isfinite(value) && value > 1e-6f) {
                    depth.at<float>(py, px) = value;
                }
            }
        }
    }
}

cv::Mat fill_group_narrow_gaps(
    cv::Mat& depth,
    cv::Mat& confidence,
    cv::Mat& valid,
    const cv::Mat& support,
    const cv::Mat& source_mask) {
    if (depth.empty() || confidence.empty() || valid.empty() || support.empty()) {
        return cv::Mat::zeros(valid.size(), CV_8UC1);
    }
    cv::Mat inverse_valid;
    cv::bitwise_not(valid, inverse_valid);
    cv::Mat candidates;
    cv::bitwise_and(support, inverse_valid, candidates);
    const cv::Mat& fill_source = source_mask.empty() ? valid : source_mask;
    if (cv::countNonZero(candidates) == 0 || cv::countNonZero(fill_source) == 0) {
        return cv::Mat::zeros(valid.size(), CV_8UC1);
    }

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int component_count = cv::connectedComponentsWithStats(
        candidates, labels, stats, centroids, 8, CV_32S);
    cv::Mat fill = cv::Mat::zeros(valid.size(), CV_8UC1);
    for (int component = 1; component < component_count; ++component) {
        const int x = stats.at<int>(component, cv::CC_STAT_LEFT);
        const int y = stats.at<int>(component, cv::CC_STAT_TOP);
        const int width = stats.at<int>(component, cv::CC_STAT_WIDTH);
        const int height = stats.at<int>(component, cv::CC_STAT_HEIGHT);
        const int area = stats.at<int>(component, cv::CC_STAT_AREA);
        const int thickness = std::min(width, height);
        const int length = std::max(width, height);
        if (area < 32 || area > 30000 || thickness <= 0 || thickness > 24
            || length < thickness * 6) {
            continue;
        }

        // A model-confidence failure is a gap only when valid model samples
        // exist on both sides.  This rejects the genuine outer aperture while
        // accepting the long, thin strip inside the anchor support.
        const bool horizontal = width >= height;
        const int first = horizontal ? x : y;
        const int last = horizontal ? x + width : y + height;
        const int before = horizontal ? std::max(0, y - 2) : std::max(0, x - 2);
        const int after = horizontal
            ? std::min(valid.rows - 1, y + height + 1)
            : std::min(valid.cols - 1, x + width + 1);
        int before_valid = 0;
        int after_valid = 0;
        const int span = std::max(1, last - first);
        for (int offset = 0; offset < span; ++offset) {
            const int coordinate = first + offset;
            if (horizontal) {
                before_valid += fill_source.at<std::uint8_t>(before, coordinate) != 0U ? 1 : 0;
                after_valid += fill_source.at<std::uint8_t>(after, coordinate) != 0U ? 1 : 0;
            } else {
                before_valid += fill_source.at<std::uint8_t>(coordinate, before) != 0U ? 1 : 0;
                after_valid += fill_source.at<std::uint8_t>(coordinate, after) != 0U ? 1 : 0;
            }
        }
        if (before_valid * 10 < span * 5 || after_valid * 10 < span * 5) {
            continue;
        }
        fill.setTo(255U, labels == component);
    }
    if (cv::countNonZero(fill) == 0) {
        return fill;
    }

    const cv::Mat nearest_depth = nearest_fill_values(depth, fill_source);
    const cv::Mat nearest_confidence = nearest_fill_values(confidence, fill_source);
    for (int y = 0; y < valid.rows; ++y) {
        for (int x = 0; x < valid.cols; ++x) {
            if (fill.at<std::uint8_t>(y, x) == 0U) {
                continue;
            }
            depth.at<float>(y, x) = nearest_depth.at<float>(y, x);
            confidence.at<float>(y, x) = nearest_confidence.at<float>(y, x);
            valid.at<std::uint8_t>(y, x) = 255U;
        }
    }
    interpolate_group_gap_scalar(depth, fill, fill_source);
    interpolate_group_gap_scalar(confidence, fill, fill_source);
    return fill;
}

// The model-support margin can leave a one-pixel edge immediately adjacent to
// an accepted narrow gap.  That edge is still inside the previous canvas
// support, but is not part of valid_warp, so leaving it untouched produces a
// hairline black RGB/depth seam above the repaired strip.  Extend each
// accepted component by one pixel only across its short axis, and only where
// the dilated old support and finite model prediction both agree.  This does
// not close outer apertures or create side-only geometry.
cv::Mat expand_group_gap_edges(
    cv::Mat& depth,
    cv::Mat& confidence,
    cv::Mat& valid,
    const cv::Mat& fill,
    const cv::Mat& support,
    const cv::Mat& canvas_valid,
    const float min_confidence) {
    if (fill.empty() || support.empty() || canvas_valid.empty()
        || cv::countNonZero(fill) == 0) {
        return fill.clone();
    }
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int component_count = cv::connectedComponentsWithStats(
        fill, labels, stats, centroids, 8, CV_32S);
    cv::Mat expanded = fill.clone();
    cv::Mat support_halo;
    cv::dilate(support, support_halo, cv::Mat::ones(5, 5, CV_8U));
    cv::Mat not_canvas_valid;
    cv::bitwise_not(canvas_valid, not_canvas_valid);
    for (int component = 1; component < component_count; ++component) {
        const int width = stats.at<int>(component, cv::CC_STAT_WIDTH);
        const int height = stats.at<int>(component, cv::CC_STAT_HEIGHT);
        const bool horizontal = width >= height;
        const cv::Mat kernel = horizontal
            ? cv::Mat::ones(5, 1, CV_8U)
            : cv::Mat::ones(1, 5, CV_8U);
        cv::Mat component_mask;
        cv::compare(labels, component, component_mask, cv::CMP_EQ);
        cv::Mat halo;
        cv::dilate(component_mask, halo, kernel);
        cv::bitwise_and(halo, support_halo, halo);
        cv::bitwise_and(halo, not_canvas_valid, halo);
        for (int y = 0; y < halo.rows; ++y) {
            for (int x = 0; x < halo.cols; ++x) {
                if (halo.at<std::uint8_t>(y, x) == 0U) {
                    continue;
                }
                const float z = depth.at<float>(y, x);
                const float c = confidence.at<float>(y, x);
                if (!std::isfinite(z) || z <= 1e-6f
                    || !std::isfinite(c) || c < min_confidence) {
                    halo.at<std::uint8_t>(y, x) = 0U;
                    continue;
                }
                valid.at<std::uint8_t>(y, x) = 255U;
            }
        }
        cv::bitwise_or(expanded, halo, expanded);
    }
    return expanded;
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
    const cv::Mat& anchor_ring,
    const bool blend_old_overlap) {
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

    if (blend_old_overlap) {
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

// A grouped sliding window can expose a new canvas strip whose RGB is valid,
// while the immediately adjacent old-canvas pixels still carry the previous
// view's exposure.  Keep the new texture away from that boundary, but feather
// only a narrow edge to the nearest committed RGB samples.  The old canvas is
// never rewritten and the helper is not used by the single-image path.
void feather_group_new_rgb_to_canvas(
    cv::Mat& rgb,
    const cv::Mat& canvas_rgb,
    const cv::Mat& canvas_valid,
    const cv::Mat& new_mask,
    const int radius) {
    if (rgb.empty() || canvas_rgb.empty() || canvas_valid.empty()
        || new_mask.empty() || cv::countNonZero(new_mask) == 0
        || cv::countNonZero(canvas_valid) == 0 || radius <= 0) {
        return;
    }
    cv::Mat not_old_valid;
    cv::bitwise_not(canvas_valid, not_old_valid);
    cv::Mat distance_to_old;
    cv::distanceTransform(not_old_valid, distance_to_old, cv::DIST_L2, 3);

    std::vector<cv::Mat> old_channels;
    cv::split(canvas_rgb, old_channels);
    std::vector<cv::Mat> nearest_channels;
    nearest_channels.reserve(old_channels.size());
    for (const cv::Mat& channel : old_channels) {
        nearest_channels.push_back(nearest_fill_values(channel, canvas_valid));
    }

    for (int y = 0; y < rgb.rows; ++y) {
        for (int x = 0; x < rgb.cols; ++x) {
            if (new_mask.at<std::uint8_t>(y, x) == 0U) {
                continue;
            }
            const float distance = distance_to_old.at<float>(y, x);
            if (!std::isfinite(distance) || distance > static_cast<float>(radius)) {
                continue;
            }
            // Retain some of the new texture even on the first pixel, then
            // return to the unmodified new RGB after the narrow feather band.
            const float alpha = std::clamp(
                0.25f + 0.75f * distance / static_cast<float>(radius),
                0.25f,
                1.0f);
            cv::Vec3f value = rgb.at<cv::Vec3f>(y, x);
            for (int channel = 0; channel < 3; ++channel) {
                value[channel] = nearest_channels[channel].at<float>(y, x)
                    * (1.0f - alpha) + value[channel] * alpha;
            }
            rgb.at<cv::Vec3f>(y, x) = value;
        }
    }
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
            throw std::invalid_argument("group mode requires a B=1,S=3 TorchScript model");
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
    int copy_x = pad_left;
    int copy_y = pad_top;
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
    // Support is the real camera-image rectangle, not an Otsu/connected-
    // component foreground guess.  Dark table and device surfaces are valid
    // observations; removing them before the model runs is what made the
    // first Canvas collapse to a small bright vertical component.  Keep only
    // the padded-image bounds out of the support mask.
    frame.support = cv::Mat::zeros(frame.rgb_f.size(), CV_8UC1);
    const int support_x = copy_x;
    const int support_y = copy_y;
    const int support_width = std::min(
        frame.match_rgb_u8.cols, frame.support.cols - support_x);
    const int support_height = std::min(
        frame.match_rgb_u8.rows, frame.support.rows - support_y);
    if (support_width > 0 && support_height > 0) {
        frame.support(cv::Rect(support_x, support_y, support_width, support_height)).setTo(255U);
    }
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
        prediction.pose_translation = cv::Vec3f(
            pose_access[0][frame_index][0],
            pose_access[0][frame_index][1],
            pose_access[0][frame_index][2]);
        prediction.pose_quaternion = cv::Vec4f(
            pose_access[0][frame_index][3],
            pose_access[0][frame_index][4],
            pose_access[0][frame_index][5],
            pose_access[0][frame_index][6]);
        prediction.has_pose = true;
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
        throw std::runtime_error("B=1,S=3 group inference requires exactly three images");
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
    const torch::Tensor images = make_image_tensor_batch(rgb_u8).to(device_, dtype_);
    const auto float_options = torch::TensorOptions().device(device_).dtype(torch::kFloat32);
    const torch::Tensor extrinsics = torch::eye(4, float_options)
        .slice(0, 0, 3)
        .reshape({1, 1, 3, 4})
        .repeat({1, 3, 1, 1});
    const torch::Tensor intrinsics = torch::eye(3, float_options)
        .reshape({1, 1, 3, 3})
        .repeat({1, 3, 1, 1});
    const torch::Tensor depth_input = torch::zeros(
        {1, 3, height, width, 1}, torch::TensorOptions().dtype(torch::kFloat32)).to(device_, dtype_);
    const torch::Tensor mask = torch::zeros(
        {1, 3, height, width}, torch::TensorOptions().dtype(torch::kFloat32)).to(device_, dtype_);

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
    if (!has_world_points) {
        throw std::runtime_error(
            "B=1,S=3 group model must include world_points and world_points_confidence outputs");
    }
    torch::Tensor world_points;
    torch::Tensor world_points_confidence;
    if (has_world_points) {
        world_points = to_cpu_float(output_tuple->elements()[3].toTensor());
        world_points_confidence = to_cpu_float(output_tuple->elements()[4].toTensor());
    }
    if (depth.dim() != 5 || confidence.dim() != 4
        || depth.size(0) != 1 || depth.size(1) != 3
        || confidence.size(0) != 1 || confidence.size(1) != 3
        || depth.size(2) != height || depth.size(3) != width
        || confidence.size(2) != height || confidence.size(3) != width
        || (has_world_points && (world_points.dim() != 5 || world_points_confidence.dim() != 4
            || world_points.size(0) != 1 || world_points.size(1) != 3
            || world_points_confidence.size(0) != 1 || world_points_confidence.size(1) != 3
            || world_points.size(2) != height || world_points.size(3) != width
            || world_points_confidence.size(2) != height
            || world_points_confidence.size(3) != width))) {
        throw std::runtime_error("group TorchScript output must have shape [1,3,H,W,*]");
    }

    std::vector<Prediction> predictions;
    predictions.reserve(3U);
    const auto depth_access = depth.accessor<float, 5>();
    const auto confidence_access = confidence.accessor<float, 4>();
    const auto world_points_access = world_points.accessor<float, 5>();
    const auto world_confidence_access = world_points_confidence.accessor<float, 4>();
    for (int sequence = 0; sequence < 3; ++sequence) {
        Prediction prediction;
        prediction.depth = cv::Mat(height, width, CV_32FC1);
        prediction.confidence = cv::Mat(height, width, CV_32FC1);
        if (has_world_points) {
            prediction.world_points = cv::Mat(height, width, CV_32FC3);
            prediction.world_points_confidence = cv::Mat(height, width, CV_32FC1);
        }
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                prediction.depth.at<float>(y, x) = depth_access[0][sequence][y][x][0];
                prediction.confidence.at<float>(y, x) = confidence_access[0][sequence][y][x];
                if (has_world_points) {
                    prediction.world_points.at<cv::Vec3f>(y, x) = cv::Vec3f(
                        world_points_access[0][sequence][y][x][0],
                        world_points_access[0][sequence][y][x][1],
                        world_points_access[0][sequence][y][x][2]);
                    prediction.world_points_confidence.at<float>(y, x) =
                        world_confidence_access[0][sequence][y][x];
                }
            }
        }
        if (pose.dim() == 3 && pose.size(0) >= 1 && pose.size(1) >= 3 && pose.size(2) >= 9) {
            const auto pose_values = pose.accessor<float, 3>();
            prediction.pose_translation = cv::Vec3f(
                pose_values[0][sequence][0],
                pose_values[0][sequence][1],
                pose_values[0][sequence][2]);
            prediction.pose_quaternion = cv::Vec4f(
                pose_values[0][sequence][3],
                pose_values[0][sequence][4],
                pose_values[0][sequence][5],
                pose_values[0][sequence][6]);
            prediction.has_pose = true;
            prediction.fov_h = pose_values[0][sequence][7];
            prediction.fov_w = pose_values[0][sequence][8];
        }
        // Reprojection must use the FOV predicted for this B=1,S=3 view.
        // Do not silently replace an invalid value with a guessed camera
        // intrinsics fallback; GroupWorldFusion will reject the group.
        if (!std::isfinite(prediction.fov_h) || prediction.fov_h <= 0.01f
            || prediction.fov_h >= 3.1f) {
            prediction.fov_h = std::numeric_limits<float>::quiet_NaN();
        }
        if (!std::isfinite(prediction.fov_w) || prediction.fov_w <= 0.01f
            || prediction.fov_w >= 3.1f) {
            prediction.fov_w = std::numeric_limits<float>::quiet_NaN();
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
        // In the legacy independent-batch path, keep one forward for
        // throughput but do not let
        // side-view predictions write geometry.  Their edge pixels are in a
        // different local depth frame; filling anchor-invalid silhouettes with
        // those depths creates the vertical point-cloud sheets that the
        // single-image stream never produces.  The aligned anchor image is the
        // sole geometry owner; side views are only used for rejection metrics.
        ++metrics.group_rejected_sources;
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
        detector = cv::SIFT::create(3000);
        norm = cv::NORM_L2;
    } catch (const cv::Exception&) {
        detector = cv::ORB::create(3000);
    }
    std::vector<cv::KeyPoint> source_keypoints;
    std::vector<cv::KeyPoint> target_keypoints;
    cv::Mat source_descriptors;
    cv::Mat target_descriptors;
    detector->detectAndCompute(
        source_gray, cv::noArray(), source_keypoints, source_descriptors);
    detector->detectAndCompute(
        target_gray, cv::noArray(), target_keypoints, target_descriptors);
    if (source_descriptors.empty() || target_descriptors.empty()
        || source_keypoints.size() < 8U || target_keypoints.size() < 8U) {
        return cv::Mat::eye(3, 3, CV_32FC1);
    }
    cv::BFMatcher matcher(norm);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(source_descriptors, target_descriptors, knn, 2);
    std::vector<cv::DMatch> good;
    for (const auto& pair : knn) {
        if (pair.size() == 2U && pair[0].distance < 0.82f * pair[1].distance) {
            good.push_back(pair[0]);
        }
    }
    if (good.size() < 5U) {
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
        source_points, target_points, cv::RANSAC, 4.0, inlier_mask);
    if (homography.empty() || inlier_mask.empty()
        || cv::countNonZero(inlier_mask) < 5
        || !plausible_planar_homography(
            homography, source.match_rgb_u8.size(), target.match_rgb_u8.size())) {
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
    return process_prepared(FramePreprocessor(options_).prepare(raw), state);
}

CandidateCommit InferenceEngine::process_prepared(
    const PreparedInput& prepared,
    const CanvasState& state) {
    FrameImage frame;
    frame.path = prepared.path;
    frame.rgb_u8 = prepared.rgb_u8;
    frame.rgb_f = prepared.rgb_f;
    frame.match_rgb_u8 = prepared.match_rgb_u8;
    frame.match_rgb_f = prepared.match_rgb_f;
    frame.support = prepared.support;
    if (prepared.has_group) {
        PreparedGroup group;
        group.model_rgb_f = prepared.group_model_rgb_f;
        group.warped_rgb_f = prepared.group_warped_rgb_f;
        group.valid_warp = prepared.group_valid_warp;
        group.fused_rgb_f = prepared.group_fused_rgb_f;
        group.union_valid = prepared.group_union_valid;
        return process_world_group(prepared.raw, state, group, prepared.read_ms);
    }
    if (prepared.has_observation_group && prepared.observation_views.size() == 3U) {
        std::vector<FrameImage> views;
        views.reserve(prepared.observation_views.size());
        for (const PreparedView& prepared_view : prepared.observation_views) {
            FrameImage view;
            view.path = prepared_view.path;
            view.rgb_u8 = prepared_view.rgb_u8;
            view.rgb_f = prepared_view.rgb_f;
            view.match_rgb_u8 = prepared_view.match_rgb_u8;
            view.match_rgb_f = prepared_view.match_rgb_f;
            view.support = prepared_view.support;
            view.forced_homography = prepared_view.canvas_homography;
            view.forced_homography_valid = prepared_view.pair_alignment_valid;
            views.push_back(std::move(view));
        }
        return process_observation_group(
            prepared.raw, state, views, prepared.read_ms);
    }
    return process_impl(
        prepared.raw,
        state,
        frame,
        nullptr,
        prepared.has_observation_group && options_.group_stride >= 3,
        prepared.read_ms);
}

CandidateCommit InferenceEngine::process_observation_group(
    const RawFrame& raw,
    const CanvasState& state,
    const std::vector<FrameImage>& views,
    const double read_ms) {
    if (views.size() != 3U) {
        throw std::runtime_error("three-camera observation requires exactly three views");
    }
    Timer total_timer;
    const int anchor_index = std::clamp(raw.group_anchor_index, 0, 2);
    const FrameImage& anchor_view = views[static_cast<std::size_t>(anchor_index)];
    CandidateCommit anchor_result = process_impl(
        raw,
        state,
        anchor_view,
        nullptr,
        true,
        read_ms,
        nullptr);

    // The anchor is the only geometry owner for pixels already represented in
    // the Canvas. Side views are still real S1 inferences: they may add a
    // previously unsupported, homography-consistent strip, but can never
    // overwrite an anchor point. This prevents parallax from becoming a
    // second surface while allowing all three same-time images to contribute
    // visible coverage.
    std::vector<std::uint32_t> current_observed_slots;
    current_observed_slots.reserve(state.slot_count() / 2U);
    CanvasState merged_state = state;
    if (anchor_result.has_patch) {
        commit_patch(merged_state, anchor_result.patch);
        current_observed_slots.insert(
            current_observed_slots.end(),
            anchor_result.patch.observed_slots.begin(),
            anchor_result.patch.observed_slots.end());
    } else if (!state.initialized) {
        // A non-initialized Canvas cannot accept a side-only first frame: the
        // anchor path is the source of the required canvas dimensions and
        // anchor RGB. Preserve the original no-patch result in this impossible
        // recovery case instead of manufacturing an invalid transaction.
        anchor_result.metrics.total_ms = total_timer.ms();
        return anchor_result;
    }

    // The production GUI stores one point per raster slot and derives XY from
    // that slot.  A fully dense anchor floor leaves no local capacity for a
    // side camera to add the floor hidden by a raised robot/object at the same
    // XY.  Reserve a deterministic half-grid only from the robust anchor floor
    // (never from raised geometry).  The retained floor density still exceeds
    // the GUI display budget, while neighbouring free slots can carry the
    // complementary side-view floor without flattening the object.
    if (merged_state.shape_valid()) {
        std::vector<float> floor_x;
        std::vector<float> floor_y;
        std::vector<float> floor_z;
        for (int y = 0; y < merged_state.height; y += 3) {
            for (int x = 0; x < merged_state.width; x += 3) {
                const std::size_t slot = static_cast<std::size_t>(y)
                    * merged_state.width + x;
                if (merged_state.valid[slot] == 0U
                    || !std::isfinite(merged_state.depth[slot])
                    || merged_state.depth[slot] <= 1e-5f) {
                    continue;
                }
                floor_x.push_back(2.0f * static_cast<float>(x)
                    / static_cast<float>(std::max(1, merged_state.width - 1)) - 1.0f);
                floor_y.push_back(2.0f * static_cast<float>(y)
                    / static_cast<float>(std::max(1, merged_state.height - 1)) - 1.0f);
                floor_z.push_back(merged_state.depth[slot]);
            }
        }
        if (floor_z.size() >= 512U) {
            std::vector<std::uint8_t> keep(floor_z.size(), 1U);
            std::array<double, 6> coefficients{};
            float residual_center = 0.0f;
            float residual_limit = 0.0f;
            const float depth_scale = std::max(std::abs(median_value(floor_z)), 1e-5f);
            for (int iteration = 0; iteration < 5; ++iteration) {
                fit_quadratic_surface(floor_x, floor_y, floor_z, keep, coefficients);
                std::vector<float> residuals;
                residuals.reserve(floor_z.size());
                for (std::size_t index = 0; index < floor_z.size(); ++index) {
                    residuals.push_back(floor_z[index] - static_cast<float>(
                        quadratic_surface_value(
                            coefficients, floor_x[index], floor_y[index])));
                }
                std::vector<float> kept_residuals;
                kept_residuals.reserve(residuals.size());
                for (std::size_t index = 0; index < residuals.size(); ++index) {
                    if (keep[index] != 0U) {
                        kept_residuals.push_back(residuals[index]);
                    }
                }
                residual_center = median_value(kept_residuals);
                std::vector<float> deviations;
                deviations.reserve(kept_residuals.size());
                for (const float residual : kept_residuals) {
                    deviations.push_back(std::abs(residual - residual_center));
                }
                residual_limit = std::max(
                    0.008f * depth_scale,
                    3.0f * 1.4826f * std::max(median_value(deviations), 1e-6f));
                for (std::size_t index = 0; index < residuals.size(); ++index) {
                    keep[index] = std::abs(residuals[index] - residual_center)
                            <= residual_limit
                        ? 1U : 0U;
                }
            }
            for (int y = 0; y < merged_state.height; ++y) {
                for (int x = 0; x < merged_state.width; ++x) {
                    if (((x + y) & 1) == 0) {
                        continue;
                    }
                    const std::size_t slot = static_cast<std::size_t>(y)
                        * merged_state.width + x;
                    if (merged_state.valid[slot] == 0U
                        || !std::isfinite(merged_state.depth[slot])) {
                        continue;
                    }
                    const float normalized_x = 2.0f * static_cast<float>(x)
                        / static_cast<float>(std::max(1, merged_state.width - 1)) - 1.0f;
                    const float normalized_y = 2.0f * static_cast<float>(y)
                        / static_cast<float>(std::max(1, merged_state.height - 1)) - 1.0f;
                    const float expected = static_cast<float>(quadratic_surface_value(
                        coefficients, normalized_x, normalized_y));
                    if (std::abs(merged_state.depth[slot] - expected - residual_center)
                        > residual_limit) {
                        continue;
                    }
                    set_slot_value(
                        merged_state, static_cast<std::uint32_t>(slot), SlotValue{});
                    merged_state.support[slot] = 0U;
                }
            }
            current_observed_slots.erase(
                std::remove_if(
                    current_observed_slots.begin(),
                    current_observed_slots.end(),
                    [&merged_state](const std::uint32_t slot) {
                        return slot >= merged_state.slot_count()
                            || merged_state.valid[slot] == 0U;
                    }),
                current_observed_slots.end());
        }
    }
    const cv::Mat anchor_homography = last_homography_.clone();
    double total_model_ms = anchor_result.metrics.model_ms;
    int total_forward_calls = anchor_result.metrics.forward_calls;
    int planar_side_sources = 0;
    int homography_side_sources = 0;
    std::size_t accepted_side_sources = 0U;
    std::size_t rejected_side_sources = 0U;

    for (int view_index = 0; view_index < 3; ++view_index) {
        if (view_index == anchor_index) {
            continue;
        }

        // A 2-D homography cannot turn a fixed side-camera image into the
        // anchor camera's surface: parallax then becomes a duplicate/ghost
        // layer.  Run the existing S=2 pair model on the real anchor/side
        // images and project the side depth through its predicted relative
        // pose instead.  This keeps the GUI launcher on S1/S2; no three-image
        // model is involved.
        CandidateCommit side_result;
        try {
            side_result = process_pair_observation(
                raw,
                merged_state,
                anchor_view,
                views[static_cast<std::size_t>(view_index)],
                0.0);
        } catch (const std::exception&) {
            ++rejected_side_sources;
            continue;
        }
        total_model_ms += side_result.metrics.model_ms;
        total_forward_calls += side_result.metrics.forward_calls;
        if (side_result.metrics.homography_error_px > 0.5) {
            ++planar_side_sources;
        }
        if (side_result.metrics.group_max_depth_residual > 0.5) {
            ++homography_side_sources;
        }

        bool accepted = side_result.has_patch
            && !side_result.patch.scene_jump;
        CandidatePatch side_patch;
        if (accepted) {
            side_patch = side_result.patch;
            side_patch.base_version = merged_state.version;
            side_patch.initialize_canvas = false;
            side_patch.anchor_rgba.clear();
            side_patch.scene_jump = false;
            if (side_patch.updates.empty()) {
                accepted = false;
            } else {
                commit_patch(merged_state, side_patch);
                current_observed_slots.insert(
                    current_observed_slots.end(),
                    side_patch.observed_slots.begin(),
                    side_patch.observed_slots.end());
                ++accepted_side_sources;
            }
        }
        if (!accepted) {
            ++rejected_side_sources;
        }
    }

    // When a real model pass succeeded, discard only large old regions outside
    // the current three-camera observation footprint. The previous observation
    // path kept every old support cell forever, so a changing mask left stale
    // planes/objects behind and produced visible ghosts. Skipped-model frames
    // keep the old canvas unchanged to preserve temporal continuity.
    if (state.initialized && !anchor_result.metrics.skipped_model
        && current_observed_slots.size() >= 4096U) {
        cv::Mat current_footprint(
            state.height, state.width, CV_8UC1, cv::Scalar(0));
        for (const std::uint32_t slot : current_observed_slots) {
            if (slot >= state.slot_count()) {
                continue;
            }
            const int x = static_cast<int>(slot % static_cast<std::uint32_t>(state.width));
            const int y = static_cast<int>(slot / static_cast<std::uint32_t>(state.width));
            current_footprint.at<std::uint8_t>(y, x) = 255U;
        }
        if (cv::countNonZero(current_footprint) >= 4096) {
            cv::dilate(
                current_footprint,
                current_footprint,
                cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));
            cv::Mat outside_current;
            cv::bitwise_not(current_footprint, outside_current);
            cv::Mat stale;
            cv::bitwise_and(
                state_mask(state.valid, state.width, state.height),
                outside_current,
                stale);
            stale = filter_components(stale, 256);
            for (int y = 0; y < stale.rows; ++y) {
                for (int x = 0; x < stale.cols; ++x) {
                    if (stale.at<std::uint8_t>(y, x) == 0U) {
                        continue;
                    }
                    const std::uint32_t slot = static_cast<std::uint32_t>(
                        y * state.width + x);
                    set_slot_value(merged_state, slot, SlotValue{});
                    merged_state.support[slot] = 0U;
                }
            }
        }
    }

    // Rebuild one transaction from the original state to the merged result.
    // The commit worker therefore sees one logical frame/version, even though
    // the existing S1 graph was called once per real camera view.
    CandidatePatch merged_patch;
    merged_patch.frame_seq = raw.frame_seq;
    merged_patch.base_version = state.version;
    merged_patch.width = merged_state.width;
    merged_patch.height = merged_state.height;
    merged_patch.initialize_canvas = !state.initialized;
    merged_patch.anchor_camera = anchor_result.patch.anchor_camera;
    merged_patch.scene_jump = anchor_result.patch.scene_jump;
    if (merged_patch.initialize_canvas) {
        merged_patch.anchor_rgba = merged_state.anchor_rgba;
    }

    const std::size_t slot_count = merged_state.slot_count();
    merged_patch.updates.reserve(slot_count);
    for (std::size_t slot = 0; slot < slot_count; ++slot) {
        const SlotValue before = state.shape_valid()
            ? slot_value_at(state, static_cast<std::uint32_t>(slot))
            : SlotValue{};
        const SlotValue after = slot_value_at(
            merged_state, static_cast<std::uint32_t>(slot));
        if (!slot_value_equal(before, after)) {
            merged_patch.updates.push_back(SlotUpdate{
                static_cast<std::uint32_t>(slot), after});
        }
        const std::uint8_t before_support = state.shape_valid()
            ? state.support[slot]
            : 0U;
        const std::uint8_t after_support = merged_state.support[slot];
        if (before_support == 0U && after_support != 0U) {
            merged_patch.observed_slots.push_back(
                static_cast<std::uint32_t>(slot));
        } else if (before_support != 0U && after_support == 0U) {
            merged_patch.cleared_support_slots.push_back(
                static_cast<std::uint32_t>(slot));
        }
    }
    merged_patch.changed_ratio = slot_count == 0U ? 0.0f
        : static_cast<float>(static_cast<double>(merged_patch.updates.size())
            / static_cast<double>(slot_count));

    anchor_result.patch = std::move(merged_patch);
    anchor_result.has_patch = anchor_result.patch.initialize_canvas
        || !anchor_result.patch.updates.empty()
        || !anchor_result.patch.observed_slots.empty()
        || !anchor_result.patch.cleared_support_slots.empty();
    anchor_result.metrics.group_size = 3;
    anchor_result.metrics.group_stride = 3;
    anchor_result.metrics.group_anchor_index = anchor_index;
    anchor_result.metrics.group_fused_sources = static_cast<int>(
        1U + accepted_side_sources);
    anchor_result.metrics.group_rejected_sources = static_cast<int>(
        rejected_side_sources);
    anchor_result.metrics.forward_calls = total_forward_calls;
    anchor_result.metrics.forward_batch_size = 1;
    anchor_result.metrics.forward_sequence_size = 1;
    anchor_result.metrics.model_ms = total_model_ms;
    // In observation mode this field records how many of the two side views
    // actually passed the planar-homography branch (0..2). It makes fallback
    // to unstable pair-pose geometry visible in the existing metrics stream.
    anchor_result.metrics.homography_error_px =
        static_cast<double>(planar_side_sources);
    // Observation-mode diagnostic: how many side views used a validated
    // direct/chained image homography instead of pose-only floor projection.
    anchor_result.metrics.group_max_depth_residual =
        static_cast<double>(homography_side_sources);
    anchor_result.metrics.changed_ratio = anchor_result.patch.changed_ratio;
    anchor_result.metrics.changed_point_count = static_cast<std::uint32_t>(
        anchor_result.patch.updates.size());
    anchor_result.metrics.valid_point_count = static_cast<std::uint32_t>(
        std::count(merged_state.valid.begin(), merged_state.valid.end(),
            static_cast<std::uint8_t>(1U)));
    anchor_result.frame.status = anchor_result.has_patch
        ? FrameStatus::Committed
        : FrameStatus::NoChange;
    anchor_result.frame.changed_ratio = anchor_result.patch.changed_ratio;
    anchor_result.frame.changed_point_count = anchor_result.metrics.changed_point_count;
    anchor_result.frame.valid_point_count = anchor_result.metrics.valid_point_count;
    anchor_result.metrics.total_ms = total_timer.ms();

    // Do not carry a side-view RGB or transform into the next logical frame.
    // The committed merged Canvas is the only stable reference for the next
    // group; the anchor transform remains the temporal reference.
    live_rgb_float_ = state_rgb_float(merged_state);
    last_homography_ = anchor_homography;
    return anchor_result;
}

CandidateCommit InferenceEngine::process_pair_observation(
    const RawFrame& raw,
    const CanvasState& state,
    const FrameImage& anchor,
    const FrameImage& side,
    const double read_ms) {
    Timer total_timer;
    CandidateCommit result;
    result.frame.frame_seq = raw.frame_seq;
    result.frame.base_version = state.version;
    result.frame.commit_version = state.version;
    result.frame.image_name = side.path.filename().string();
    result.metrics.frame_seq = raw.frame_seq;
    result.metrics.image = result.frame.image_name;
    result.metrics.read_ms = read_ms;
    result.metrics.group_size = 3;
    result.metrics.group_stride = options_.group_stride;
    result.metrics.group_anchor_index = raw.group_anchor_index;
    result.metrics.forward_batch_size = 1;
    result.metrics.forward_sequence_size = 2;
    result.metrics.roi_width = options_.width;
    result.metrics.roi_height = options_.height;
    result.metrics.model_input_width = options_.width;
    result.metrics.model_input_height = options_.height;

    if (!state.initialized || !has_pair_module_
        || anchor.match_rgb_f.empty() || side.match_rgb_f.empty()) {
        result.frame.status = FrameStatus::NoChange;
        result.metrics.total_ms = total_timer.ms();
        return result;
    }

    const int model_width = options_.width;
    const int model_height = options_.height;
    const auto make_pair_input = [&](const FrameImage& frame) {
        const cv::Mat& source = frame.match_rgb_f;
        const double scale = static_cast<double>(model_width)
            / static_cast<double>(std::max(1, source.cols));
        const int resized_height = std::max(
            1, static_cast<int>(std::round(static_cast<double>(source.rows) * scale)));
        cv::Mat resized;
        cv::resize(
            source,
            resized,
            cv::Size(model_width, resized_height),
            0.0,
            0.0,
            cv::INTER_AREA);
        if (resized_height == model_height) {
            return resized;
        }
        if (resized_height < model_height) {
            const int top = (model_height - resized_height) / 2;
            cv::Mat padded;
            cv::copyMakeBorder(
                resized,
                padded,
                top,
                model_height - resized_height - top,
                0,
                0,
                cv::BORDER_REPLICATE);
            return padded;
        }
        const int crop_top = (resized_height - model_height) / 2;
        return resized(cv::Rect(0, crop_top, model_width, model_height)).clone();
    };

    const cv::Mat anchor_input = make_pair_input(anchor);
    const cv::Mat side_input = make_pair_input(side);
    const std::vector<cv::Mat> pair_inputs = {anchor_input, side_input};
    Timer model_timer;
    // The observer artifact exposes one selected frame per call.  Read both
    // poses from the same S=2 pair with two forwards; this is deliberately
    // still the existing pair model, never the rejected S=3 graph.
    const Prediction anchor_prediction = run_model(pair_inputs, 0);
    const Prediction side_prediction = run_model(pair_inputs, 1);
    result.metrics.model_ms = model_timer.ms();
    result.metrics.forward_calls = 2;
    if (!anchor_prediction.has_pose || !side_prediction.has_pose
        || anchor_prediction.depth.empty() || side_prediction.depth.empty()) {
        result.metrics.total_ms = total_timer.ms();
        return result;
    }

    const auto rotation_from_quaternion = [](const cv::Vec4f& quaternion) {
        const float x = quaternion[0];
        const float y = quaternion[1];
        const float z = quaternion[2];
        const float w = quaternion[3];
        const float denominator = x * x + y * y + z * z + w * w;
        if (!std::isfinite(denominator) || denominator < 1e-8f) {
            return cv::Matx33f::eye();
        }
        const float scale = 2.0f / denominator;
        return cv::Matx33f(
            1.0f - scale * (y * y + z * z),
            scale * (x * y - z * w),
            scale * (x * z + y * w),
            scale * (x * y + z * w),
            1.0f - scale * (x * x + z * z),
            scale * (y * z - x * w),
            scale * (x * z - y * w),
            scale * (y * z + x * w),
            1.0f - scale * (x * x + y * y));
    };
    const cv::Matx33f anchor_rotation = rotation_from_quaternion(
        anchor_prediction.pose_quaternion);
    const cv::Matx33f side_rotation = rotation_from_quaternion(
        side_prediction.pose_quaternion);
    const cv::Vec3f& anchor_translation = anchor_prediction.pose_translation;
    const cv::Vec3f& side_translation = side_prediction.pose_translation;
    const float anchor_fx = static_cast<float>(model_width * 0.5)
        / std::tan(std::max(0.05f, anchor_prediction.fov_w * 0.5f));
    const float anchor_fy = static_cast<float>(model_height * 0.5)
        / std::tan(std::max(0.05f, anchor_prediction.fov_h * 0.5f));
    const float side_fx = static_cast<float>(model_width * 0.5)
        / std::tan(std::max(0.05f, side_prediction.fov_w * 0.5f));
    const float side_fy = static_cast<float>(model_height * 0.5)
        / std::tan(std::max(0.05f, side_prediction.fov_h * 0.5f));
    const float anchor_cx = static_cast<float>(model_width) * 0.5f;
    const float anchor_cy = static_cast<float>(model_height) * 0.5f;
    const float side_cx = static_cast<float>(model_width) * 0.5f;
    const float side_cy = static_cast<float>(model_height) * 0.5f;
    const cv::Point origin = content_origin(
        anchor.support,
        std::max(32, static_cast<int>(std::round(options_.width * 0.05))),
        std::max(128, static_cast<int>(std::round(options_.width * 0.18))));

    const double anchor_scale = static_cast<double>(model_width)
        / static_cast<double>(std::max(1, anchor.match_rgb_f.cols));
    const int anchor_resized_height = std::max(
        1, static_cast<int>(std::round(
            static_cast<double>(anchor.match_rgb_f.rows) * anchor_scale)));
    const int anchor_pad_top = anchor_resized_height < model_height
        ? (model_height - anchor_resized_height) / 2
        : 0;

    auto project_anchor = [&](const cv::Vec3f& point, double& pixel_x, double& pixel_y) {
        if (!std::isfinite(point[0]) || !std::isfinite(point[1])
            || !std::isfinite(point[2]) || point[2] <= 1e-5f) {
            return false;
        }
        pixel_x = static_cast<double>(anchor_cx)
            + static_cast<double>(anchor_fx) * point[0] / point[2];
        pixel_y = static_cast<double>(anchor_cy)
            + static_cast<double>(anchor_fy) * point[1] / point[2];
        return std::isfinite(pixel_x) && std::isfinite(pixel_y);
    };

    // Pair-model depth has an arbitrary global scale. Calibrate that scale
    // against already committed anchor pixels before projecting the side
    // camera; using a raw pair depth here would make a second floating layer.
    std::vector<float> scale_values;
    scale_values.reserve(50000U);
    for (int y = anchor_pad_top + 8; y < model_height - 8; y += 4) {
        for (int x = 8; x < model_width - 8; x += 4) {
            const float depth = anchor_prediction.depth.at<float>(y, x);
            if (!std::isfinite(depth) || depth <= 1e-5f) {
                continue;
            }
            const cv::Vec3f camera_point(
                (static_cast<float>(x) - anchor_cx) * depth / anchor_fx,
                (static_cast<float>(y) - anchor_cy) * depth / anchor_fy,
                depth);
            const cv::Vec3f world_point = anchor_rotation.t()
                * (camera_point - anchor_translation);
            const cv::Vec3f anchor_point = anchor_rotation * world_point
                + anchor_translation;
            double projected_x = 0.0;
            double projected_y = 0.0;
            if (!project_anchor(anchor_point, projected_x, projected_y)) {
                continue;
            }
            const int canvas_x = origin.x + static_cast<int>(std::lround(projected_x));
            const int canvas_y = origin.y + static_cast<int>(std::lround(projected_y));
            if (canvas_x < 0 || canvas_x >= state.width
                || canvas_y < 0 || canvas_y >= state.height) {
                continue;
            }
            const std::size_t slot = static_cast<std::size_t>(canvas_y) * state.width
                + static_cast<std::size_t>(canvas_x);
            if (slot >= state.valid.size() || state.valid[slot] == 0U
                || !std::isfinite(state.depth[slot]) || state.depth[slot] <= 1e-5f) {
                continue;
            }
            scale_values.push_back(state.depth[slot] / std::max(anchor_point[2], 1e-5f));
        }
    }
    double pair_scale = 1.0;
    if (scale_values.size() >= 128U) {
        pair_scale = std::clamp(
            static_cast<double>(median_value(scale_values)),
            0.25,
            4.0);
    }

    const double side_scale = static_cast<double>(model_width)
        / static_cast<double>(std::max(1, side.match_rgb_f.cols));
    const int side_resized_height = std::max(
        1, static_cast<int>(std::round(
            static_cast<double>(side.match_rgb_f.rows) * side_scale)));
    const int side_pad_top = side_resized_height < model_height
        ? (model_height - side_resized_height) / 2
        : 0;
    const int side_crop_top = side_resized_height > model_height
        ? (side_resized_height - model_height) / 2
        : 0;
    const cv::Rect side_content_rect = [&]() {
        if (!side.support.empty() && cv::countNonZero(side.support) > 0) {
            return cv::boundingRect(side.support);
        }
        return cv::Rect(0, 0, side.match_rgb_u8.cols, side.match_rgb_u8.rows);
    }();
    const auto source_pixel_for_model = [&](const int model_x, const int model_y,
                                            int& source_x, int& source_y) {
        const int raw_x = static_cast<int>(std::lround(
            static_cast<double>(model_x) / side_scale));
        const int raw_y = static_cast<int>(std::lround(
            static_cast<double>(model_y - side_pad_top + side_crop_top)
                / side_scale));
        if (raw_x < 4 || raw_x >= side_content_rect.width - 4
            || raw_y < 4 || raw_y >= side_content_rect.height - 4
            || raw_x >= side.match_rgb_u8.cols
            || raw_y >= side.match_rgb_u8.rows) {
            return false;
        }
        const int support_x = side_content_rect.x + raw_x;
        const int support_y = side_content_rect.y + raw_y;
        if (!side.support.empty()
            && (support_x < 0 || support_x >= side.support.cols
                || support_y < 0 || support_y >= side.support.rows
                || side.support.at<std::uint8_t>(support_y, support_x) == 0U)) {
            return false;
        }
        source_x = raw_x;
        source_y = raw_y;
        return true;
    };

    std::array<std::vector<float>, 3> side_color_values;
    std::array<std::vector<float>, 3> anchor_color_values;
    for (int y = side_pad_top + 12; y < model_height - 12; y += 4) {
        for (int x = 12; x < model_width - 12; x += 4) {
            const float depth = side_prediction.depth.at<float>(y, x);
            const float confidence = side_prediction.confidence.at<float>(y, x);
            if (!std::isfinite(depth) || depth <= 1e-5f
                || !std::isfinite(confidence)
                || confidence < static_cast<float>(options_.min_conf)) {
                continue;
            }
            int source_x = 0;
            int source_y = 0;
            if (!source_pixel_for_model(x, y, source_x, source_y)) {
                continue;
            }
            const cv::Vec3f side_point(
                (static_cast<float>(x) - side_cx) * depth / side_fx,
                (static_cast<float>(y) - side_cy) * depth / side_fy,
                depth);
            const cv::Vec3f world_point = side_rotation.t()
                * (side_point - side_translation);
            cv::Vec3f anchor_point = anchor_rotation * world_point
                + anchor_translation;
            anchor_point *= static_cast<float>(pair_scale);
            double projected_x = 0.0;
            double projected_y = 0.0;
            if (!project_anchor(anchor_point, projected_x, projected_y)) {
                continue;
            }
            const int canvas_x = origin.x + static_cast<int>(std::lround(projected_x));
            const int canvas_y = origin.y + static_cast<int>(std::lround(projected_y));
            if (canvas_x < 0 || canvas_x >= state.width
                || canvas_y < 0 || canvas_y >= state.height) {
                continue;
            }
            const std::size_t slot = static_cast<std::size_t>(canvas_y) * state.width
                + static_cast<std::size_t>(canvas_x);
            if (slot >= state.slot_count() || state.valid[slot] == 0U
                || !std::isfinite(state.depth[slot]) || state.depth[slot] <= 1e-5f) {
                continue;
            }
            const cv::Vec3b source_color = side.match_rgb_u8.at<cv::Vec3b>(
                source_y, source_x);
            const auto destination_color = unpack_rgba(state.rgba[slot]);
            for (int channel = 0; channel < 3; ++channel) {
                const float source_value =
                    static_cast<float>(source_color[channel]) / 255.0f;
                const float destination_value =
                    static_cast<float>(destination_color[channel]) / 255.0f;
                if (source_value > 0.04f && source_value < 0.96f
                    && destination_value > 0.04f && destination_value < 0.96f) {
                    side_color_values[static_cast<std::size_t>(channel)]
                        .push_back(source_value);
                    anchor_color_values[static_cast<std::size_t>(channel)]
                        .push_back(destination_value);
                }
            }
        }
    }

    // Match the joint B1S3 colour rule used by the tuned local path.
    // A robust additive exposure offset preserves each camera's chroma;
    // fitting an independent gain per channel desaturated the dark green
    // floor in newly exposed side-camera regions.
    std::array<double, 3> side_color_bias{0.0, 0.0, 0.0};
    for (int channel = 0; channel < 3; ++channel) {
        const auto& source_values =
            side_color_values[static_cast<std::size_t>(channel)];
        const auto& destination_values =
            anchor_color_values[static_cast<std::size_t>(channel)];
        if (source_values.size() < 128U
            || destination_values.size() != source_values.size()) {
            continue;
        }
        std::vector<float> differences;
        differences.reserve(source_values.size());
        for (std::size_t index = 0; index < source_values.size(); ++index) {
            differences.push_back(
                destination_values[index] - source_values[index]);
        }
        side_color_bias[static_cast<std::size_t>(channel)] =
            std::clamp(
                static_cast<double>(median_value(differences)),
                -0.18,
                0.18);
    }

    // Prepare a local colour bridge from the already committed canvas.
    // Global per-channel exposure correction aligns the cameras on average;
    // this short feather removes the remaining visible boundary step without
    // allowing a side camera to rewrite existing anchor texture.
    const cv::Mat existing_valid =
        state_mask(state.valid, state.width, state.height);
    cv::Mat distance_to_existing;
    std::vector<cv::Mat> nearest_existing_channels;
    if (cv::countNonZero(existing_valid) > 0) {
        cv::Mat missing_existing;
        cv::bitwise_not(existing_valid, missing_existing);
        cv::distanceTransform(
            missing_existing,
            distance_to_existing,
            cv::DIST_L2,
            3);
        const cv::Mat existing_rgb = state_rgb_float(state);
        std::vector<cv::Mat> existing_channels;
        cv::split(existing_rgb, existing_channels);
        nearest_existing_channels.reserve(existing_channels.size());
        for (const cv::Mat& channel : existing_channels) {
            nearest_existing_channels.push_back(
                nearest_fill_values(channel, existing_valid));
        }
    }

    const int step = 2;
    const std::size_t slot_count = state.slot_count();
    std::vector<int> update_index(slot_count, -1);
    std::vector<std::uint8_t> observed(slot_count, 0U);
    std::vector<float> update_quality;
    CandidatePatch patch;
    patch.frame_seq = raw.frame_seq;
    patch.base_version = state.version;
    patch.width = state.width;
    patch.height = state.height;
    patch.initialize_canvas = false;
    patch.scene_jump = false;
    patch.anchor_camera = state.anchor_camera;
    patch.updates.reserve(slot_count / 8U);
    patch.observed_slots.reserve(slot_count / 8U);

    // For cameras surrounding the robot, the pair model's relative pose is
    // not stable enough to place the low-texture floor.  Use the real-image
    // side-to-anchor homography only for the dominant planar surface.  The
    // anchor keeps ownership of every existing cell, so this path can fill
    // floor hidden from one camera without creating a second robot or floor.
    cv::Mat side_to_canvas;
    if (side.forced_homography_valid && !side.forced_homography.empty()) {
        side_to_canvas = side.forced_homography.clone();
        if (!last_homography_.empty()) {
            side_to_canvas = last_homography_ * side_to_canvas;
        }
        side_to_canvas.convertTo(side_to_canvas, CV_32FC1);
        if (!cv::checkRange(side_to_canvas)) {
            side_to_canvas.release();
        }
    }

    std::vector<float> committed_depths;
    committed_depths.reserve(state.slot_count());
    for (std::size_t slot = 0; slot < state.slot_count(); ++slot) {
        if (state.valid[slot] != 0U && std::isfinite(state.depth[slot])
            && state.depth[slot] > 1e-5f) {
            committed_depths.push_back(state.depth[slot]);
        }
    }
    const float floor_depth = median_value(committed_depths);

    // Preserve the anchor floor's perspective instead of writing one constant
    // depth sheet.  A robust quadratic in normalized canvas coordinates is
    // sufficient for the gently curved model depth while rejecting raised
    // robot/object pixels as minority residuals.
    std::vector<float> floor_x;
    std::vector<float> floor_y;
    std::vector<float> floor_z;
    for (int canvas_y = 0; canvas_y < state.height; canvas_y += 3) {
        for (int canvas_x = 0; canvas_x < state.width; canvas_x += 3) {
            const std::size_t slot = static_cast<std::size_t>(canvas_y)
                * state.width + canvas_x;
            if (state.valid[slot] == 0U || !std::isfinite(state.depth[slot])
                || state.depth[slot] <= 1e-5f) {
                continue;
            }
            floor_x.push_back(2.0f * static_cast<float>(canvas_x)
                / static_cast<float>(std::max(1, state.width - 1)) - 1.0f);
            floor_y.push_back(2.0f * static_cast<float>(canvas_y)
                / static_cast<float>(std::max(1, state.height - 1)) - 1.0f);
            floor_z.push_back(state.depth[slot]);
        }
    }
    std::vector<std::uint8_t> floor_keep(floor_z.size(), 1U);
    std::array<double, 6> floor_coefficients{};
    float floor_surface_center = 0.0f;
    float floor_surface_limit = std::max(
        0.008f * std::max(floor_depth, 1e-3f), 1e-6f);
    for (int iteration = 0; iteration < 5 && floor_z.size() >= 512U; ++iteration) {
        fit_quadratic_surface(
            floor_x, floor_y, floor_z, floor_keep, floor_coefficients);
        std::vector<float> residuals;
        residuals.reserve(floor_z.size());
        for (std::size_t index = 0; index < floor_z.size(); ++index) {
            residuals.push_back(floor_z[index] - static_cast<float>(
                quadratic_surface_value(
                    floor_coefficients, floor_x[index], floor_y[index])));
        }
        const float center = median_value(residuals);
        std::vector<float> deviations;
        deviations.reserve(residuals.size());
        for (const float residual : residuals) {
            deviations.push_back(std::abs(residual - center));
        }
        const float limit = std::max(
            0.008f * std::max(floor_depth, 1e-3f),
            3.0f * 1.4826f * std::max(median_value(deviations), 1e-6f));
        floor_surface_center = center;
        floor_surface_limit = limit;
        for (std::size_t index = 0; index < residuals.size(); ++index) {
            floor_keep[index] = std::abs(residuals[index] - center) <= limit
                ? 1U : 0U;
        }
    }
    const bool floor_surface_valid =
        std::count(floor_keep.begin(), floor_keep.end(), 1U) >= 512;

    cv::Mat anchor_current_support = anchor.support.clone();
    if (!last_homography_.empty()) {
        anchor_current_support = warp_like(
            anchor.support,
            last_homography_,
            cv::Size(state.width, state.height),
            cv::INTER_NEAREST);
    }
    cv::threshold(
        anchor_current_support,
        anchor_current_support,
        127.0,
        255.0,
        cv::THRESH_BINARY);
    anchor_current_support.convertTo(anchor_current_support, CV_8UC1);
    cv::Mat outside_anchor_support;
    cv::bitwise_not(anchor_current_support, outside_anchor_support);
    cv::Mat distance_to_anchor_support;
    cv::distanceTransform(
        outside_anchor_support,
        distance_to_anchor_support,
        cv::DIST_L2,
        3);

    // A 3-D plane has affine inverse depth in normalized image coordinates.
    // Iteratively fit the dominant plane; raised robot/box pixels become
    // robust outliers and are never permitted to enter side-only coverage.
    std::vector<cv::Vec3d> plane_samples;
    plane_samples.reserve(
        static_cast<std::size_t>(model_width / 4) * (model_height / 4));
    for (int y = side_pad_top + 12; y < model_height - 12; y += 4) {
        for (int x = 12; x < model_width - 12; x += 4) {
            const float depth = side_prediction.depth.at<float>(y, x);
            const float confidence = side_prediction.confidence.at<float>(y, x);
            int source_x = 0;
            int source_y = 0;
            if (!std::isfinite(depth) || depth <= 1e-5f
                || !std::isfinite(confidence)
                || confidence < static_cast<float>(options_.min_conf)
                || !source_pixel_for_model(x, y, source_x, source_y)) {
                continue;
            }
            plane_samples.emplace_back(
                (static_cast<double>(x) - side_cx) / side_fx,
                (static_cast<double>(y) - side_cy) / side_fy,
                1.0 / static_cast<double>(depth));
        }
    }
    cv::Vec3d plane_coefficients(0.0, 0.0, 0.0);
    double plane_center = 0.0;
    double plane_limit = 0.0;
    std::vector<std::uint8_t> plane_keep(plane_samples.size(), 1U);
    for (int iteration = 0; iteration < 5 && plane_samples.size() >= 512U; ++iteration) {
        cv::Mat normal = cv::Mat::zeros(3, 3, CV_64F);
        cv::Mat rhs = cv::Mat::zeros(3, 1, CV_64F);
        std::size_t kept = 0U;
        for (std::size_t index = 0; index < plane_samples.size(); ++index) {
            if (plane_keep[index] == 0U) {
                continue;
            }
            const cv::Vec3d& sample = plane_samples[index];
            const double basis[3] = {sample[0], sample[1], 1.0};
            for (int row = 0; row < 3; ++row) {
                rhs.at<double>(row, 0) += basis[row] * sample[2];
                for (int column = 0; column < 3; ++column) {
                    normal.at<double>(row, column) += basis[row] * basis[column];
                }
            }
            ++kept;
        }
        cv::Mat solution;
        if (kept < 512U || !cv::solve(normal, rhs, solution, cv::DECOMP_SVD)) {
            plane_keep.assign(plane_keep.size(), 0U);
            break;
        }
        plane_coefficients = cv::Vec3d(
            solution.at<double>(0, 0),
            solution.at<double>(1, 0),
            solution.at<double>(2, 0));
        std::vector<float> residuals;
        residuals.reserve(plane_samples.size());
        for (const cv::Vec3d& sample : plane_samples) {
            residuals.push_back(static_cast<float>(sample[2]
                - (plane_coefficients[0] * sample[0]
                    + plane_coefficients[1] * sample[1]
                    + plane_coefficients[2])));
        }
        plane_center = median_value(residuals);
        std::vector<float> deviations;
        deviations.reserve(residuals.size());
        for (const float residual : residuals) {
            deviations.push_back(std::abs(residual - plane_center));
        }
        plane_limit = std::max(
            0.012 * std::abs(plane_coefficients[2]),
            3.0 * 1.4826 * static_cast<double>(
                std::max(median_value(deviations), 1e-7f)));
        for (std::size_t index = 0; index < residuals.size(); ++index) {
            plane_keep[index] =
                std::abs(static_cast<double>(residuals[index]) - plane_center)
                    <= plane_limit
                ? 1U : 0U;
        }
    }
    const bool dominant_side_plane_valid =
        std::isfinite(floor_depth) && floor_depth > 1e-5f
        && floor_surface_valid
        && std::count(plane_keep.begin(), plane_keep.end(), 1U) >= 512;
    const bool use_planar_homography = dominant_side_plane_valid
        && !side_to_canvas.empty();
    const bool use_planar_completion = dominant_side_plane_valid;
    result.metrics.homography_error_px = use_planar_completion ? 1.0 : 0.0;
    result.metrics.group_max_depth_residual = use_planar_homography ? 1.0 : 0.0;
    const cv::Point side_origin = content_origin(
        side.support,
        std::max(32, static_cast<int>(std::round(options_.width * 0.05))),
        std::max(128, static_cast<int>(std::round(options_.width * 0.18))));
    const cv::Mat side_bright_nonplanar =
        bright_nonplanar_mask(side.match_rgb_u8);

    const int projection_step = use_planar_completion ? 1 : step;
    for (int y = side_pad_top + 8; y < model_height - 8; y += projection_step) {
        for (int x = 8; x < model_width - 8; x += projection_step) {
            const float depth = side_prediction.depth.at<float>(y, x);
            const float confidence = side_prediction.confidence.at<float>(y, x);
            if (!std::isfinite(depth) || depth <= 1e-5f
                || !std::isfinite(confidence)
                || confidence < static_cast<float>(options_.min_conf)) {
                continue;
            }
            int source_x = 0;
            int source_y = 0;
            if (!source_pixel_for_model(x, y, source_x, source_y)) {
                continue;
            }
            if (use_planar_completion
                && !side_bright_nonplanar.empty()
                && side_bright_nonplanar.at<std::uint8_t>(source_y, source_x) != 0U) {
                // Pair depth can flatten the bright robot into the dominant
                // inverse-depth plane. It is still an occluder in RGB and
                // must never be copied as floor completion.
                continue;
            }
            if (!use_planar_completion) {
                // A side view may contribute only its robust dominant floor.
                // Falling back to raw S2 object depth duplicates the robot and
                // creates the vertical curtains seen in the real GUI.
                continue;
            }
            int canvas_x = -1;
            int canvas_y = -1;
            float candidate_depth = 0.0f;
            if (use_planar_completion) {
                const double normalized_x =
                    (static_cast<double>(x) - side_cx) / side_fx;
                const double normalized_y =
                    (static_cast<double>(y) - side_cy) / side_fy;
                const double inverse_depth = 1.0 / static_cast<double>(depth);
                const double residual = inverse_depth
                    - (plane_coefficients[0] * normalized_x
                        + plane_coefficients[1] * normalized_y
                        + plane_coefficients[2]);
                if (std::abs(residual - plane_center) > plane_limit) {
                    continue;
                }
                if (use_planar_homography) {
                    const double source_canvas_x = side_origin.x + source_x;
                    const double source_canvas_y = side_origin.y + source_y;
                    const double denominator =
                        side_to_canvas.at<float>(2, 0) * source_canvas_x
                        + side_to_canvas.at<float>(2, 1) * source_canvas_y
                        + side_to_canvas.at<float>(2, 2);
                    if (!std::isfinite(denominator)
                        || std::abs(denominator) < 1e-8) {
                        continue;
                    }
                    canvas_x = static_cast<int>(std::lround((
                        side_to_canvas.at<float>(0, 0) * source_canvas_x
                        + side_to_canvas.at<float>(0, 1) * source_canvas_y
                        + side_to_canvas.at<float>(0, 2)) / denominator));
                    canvas_y = static_cast<int>(std::lround((
                        side_to_canvas.at<float>(1, 0) * source_canvas_x
                        + side_to_canvas.at<float>(1, 1) * source_canvas_y
                        + side_to_canvas.at<float>(1, 2)) / denominator));
                } else {
                    // The third camera has little direct texture overlap with
                    // the anchor. Intersect its ray with the fitted dominant
                    // plane, then use the existing S2 relative camera pose for
                    // XY only. Object/robot depth is deliberately discarded.
                    const double fitted_inverse_depth =
                        plane_coefficients[0] * normalized_x
                        + plane_coefficients[1] * normalized_y
                        + plane_coefficients[2] + plane_center;
                    if (!std::isfinite(fitted_inverse_depth)
                        || fitted_inverse_depth <= 1e-8) {
                        continue;
                    }
                    const float planar_side_depth = static_cast<float>(
                        1.0 / fitted_inverse_depth);
                    const cv::Vec3f side_floor_point(
                        (static_cast<float>(x) - side_cx)
                            * planar_side_depth / side_fx,
                        (static_cast<float>(y) - side_cy)
                            * planar_side_depth / side_fy,
                        planar_side_depth);
                    const cv::Vec3f world_floor_point = side_rotation.t()
                        * (side_floor_point - side_translation);
                    cv::Vec3f anchor_floor_point = anchor_rotation
                        * world_floor_point + anchor_translation;
                    anchor_floor_point *= static_cast<float>(pair_scale);
                    double projected_x = 0.0;
                    double projected_y = 0.0;
                    if (!project_anchor(
                            anchor_floor_point, projected_x, projected_y)) {
                        continue;
                    }
                    const double unscaled_canvas_x = origin.x + projected_x;
                    const double unscaled_canvas_y = origin.y + projected_y;
                    if (!last_homography_.empty()) {
                        const double denominator =
                            last_homography_.at<float>(2, 0) * unscaled_canvas_x
                            + last_homography_.at<float>(2, 1) * unscaled_canvas_y
                            + last_homography_.at<float>(2, 2);
                        if (!std::isfinite(denominator)
                            || std::abs(denominator) < 1e-8) {
                            continue;
                        }
                        canvas_x = static_cast<int>(std::lround((
                            last_homography_.at<float>(0, 0) * unscaled_canvas_x
                            + last_homography_.at<float>(0, 1) * unscaled_canvas_y
                            + last_homography_.at<float>(0, 2)) / denominator));
                        canvas_y = static_cast<int>(std::lround((
                            last_homography_.at<float>(1, 0) * unscaled_canvas_x
                            + last_homography_.at<float>(1, 1) * unscaled_canvas_y
                            + last_homography_.at<float>(1, 2)) / denominator));
                    } else {
                        canvas_x = static_cast<int>(std::lround(unscaled_canvas_x));
                        canvas_y = static_cast<int>(std::lround(unscaled_canvas_y));
                    }
                }
                if (canvas_x < 0 || canvas_x >= state.width
                    || canvas_y < 0 || canvas_y >= state.height
                    || distance_to_anchor_support.at<float>(canvas_y, canvas_x) > 160.0f) {
                    continue;
                }
                const float normalized_canvas_x =
                    2.0f * static_cast<float>(canvas_x)
                    / static_cast<float>(std::max(1, state.width - 1)) - 1.0f;
                const float normalized_canvas_y =
                    2.0f * static_cast<float>(canvas_y)
                    / static_cast<float>(std::max(1, state.height - 1)) - 1.0f;
                candidate_depth = static_cast<float>(quadratic_surface_value(
                    floor_coefficients,
                    normalized_canvas_x,
                    normalized_canvas_y));
            }
            if (canvas_x < 0 || canvas_x >= state.width
                || canvas_y < 0 || canvas_y >= state.height) {
                continue;
            }
            std::size_t slot = static_cast<std::size_t>(canvas_y) * state.width
                + static_cast<std::size_t>(canvas_x);
            if (slot >= slot_count) {
                continue;
            }
            SlotValue before = slot_value_at(
                state, static_cast<std::uint32_t>(slot));
            bool already_valid = before.valid != 0U
                && std::isfinite(before.depth) && before.depth > 1e-5f;
            if (!std::isfinite(candidate_depth) || candidate_depth <= 1e-5f) {
                continue;
            }
            if (use_planar_completion && already_valid) {
                const float normalized_canvas_x =
                    2.0f * static_cast<float>(canvas_x)
                    / static_cast<float>(std::max(1, state.width - 1)) - 1.0f;
                const float normalized_canvas_y =
                    2.0f * static_cast<float>(canvas_y)
                    / static_cast<float>(std::max(1, state.height - 1)) - 1.0f;
                const float expected_floor_depth = static_cast<float>(
                    quadratic_surface_value(
                        floor_coefficients,
                        normalized_canvas_x,
                        normalized_canvas_y));
                const bool existing_is_floor = std::isfinite(expected_floor_depth)
                    && std::abs(before.depth - expected_floor_depth
                        - floor_surface_center) <= floor_surface_limit;
                if (existing_is_floor) {
                    // The anchor already represents this floor location.
                    continue;
                }

                // One GUI raster slot can carry only one Z value.  Keep the
                // raised anchor point (robot/object) and place the side-view
                // floor in the nearest locally reserved slot.  This is a
                // sub-pixel XY dither, not a second registration transform.
                int best_x = -1;
                int best_y = -1;
                int best_distance_squared = std::numeric_limits<int>::max();
                constexpr int kConflictSearchRadius = 6;
                for (int dy = -kConflictSearchRadius;
                     dy <= kConflictSearchRadius;
                     ++dy) {
                    for (int dx = -kConflictSearchRadius;
                         dx <= kConflictSearchRadius;
                         ++dx) {
                        const int distance_squared = dx * dx + dy * dy;
                        if (distance_squared == 0
                            || distance_squared > kConflictSearchRadius
                                * kConflictSearchRadius
                            || distance_squared >= best_distance_squared) {
                            continue;
                        }
                        const int candidate_x = canvas_x + dx;
                        const int candidate_y = canvas_y + dy;
                        if (candidate_x < 0 || candidate_x >= state.width
                            || candidate_y < 0 || candidate_y >= state.height) {
                            continue;
                        }
                        const std::size_t candidate_slot =
                            static_cast<std::size_t>(candidate_y) * state.width
                            + static_cast<std::size_t>(candidate_x);
                        if (state.valid[candidate_slot] != 0U
                            || update_index[candidate_slot] >= 0) {
                            continue;
                        }
                        best_x = candidate_x;
                        best_y = candidate_y;
                        best_distance_squared = distance_squared;
                    }
                }
                if (best_x < 0 || best_y < 0) {
                    continue;
                }
                canvas_x = best_x;
                canvas_y = best_y;
                slot = static_cast<std::size_t>(canvas_y) * state.width
                    + static_cast<std::size_t>(canvas_x);
                before = slot_value_at(
                    state, static_cast<std::uint32_t>(slot));
                already_valid = false;
                const float displaced_normalized_x =
                    2.0f * static_cast<float>(canvas_x)
                    / static_cast<float>(std::max(1, state.width - 1)) - 1.0f;
                const float displaced_normalized_y =
                    2.0f * static_cast<float>(canvas_y)
                    / static_cast<float>(std::max(1, state.height - 1)) - 1.0f;
                candidate_depth = static_cast<float>(quadratic_surface_value(
                    floor_coefficients,
                    displaced_normalized_x,
                    displaced_normalized_y));
                if (!std::isfinite(candidate_depth)
                    || candidate_depth <= 1e-5f) {
                    continue;
                }
            }
            float depth_residual = 0.0f;
            if (already_valid) {
                depth_residual = std::abs(candidate_depth - before.depth)
                    / std::max(std::abs(before.depth), 1e-3f);
                // Existing anchor geometry remains authoritative unless the
                // pair result agrees locally. This is the anti-ghost gate.
                if (depth_residual > 0.18f) {
                    continue;
                }
            }

            const cv::Vec3b source_color = side.match_rgb_u8.at<cv::Vec3b>(
                source_y, source_x);
            const auto old_color = unpack_rgba(before.rgba);
            const float color_alpha = already_valid ? 0.18f : 1.0f;
            std::array<std::uint8_t, 3> corrected_color{};
            for (int channel = 0; channel < 3; ++channel) {
                const double source_value =
                    static_cast<double>(source_color[channel]) / 255.0;
                const double corrected_value = std::clamp(
                    source_value
                    + side_color_bias[static_cast<std::size_t>(channel)],
                    0.0,
                    1.0);
                corrected_color[static_cast<std::size_t>(channel)] =
                    static_cast<std::uint8_t>(std::clamp(
                        static_cast<int>(std::lround(corrected_value * 255.0)),
                        0,
                        255));
            }
            if (!already_valid
                && !distance_to_existing.empty()
                && nearest_existing_channels.size() == 3U) {
                const float distance =
                    distance_to_existing.at<float>(canvas_y, canvas_x);
                if (std::isfinite(distance) && distance < 16.0f) {
                    const float old_weight =
                        0.55f * (1.0f - distance / 16.0f);
                    for (int channel = 0; channel < 3; ++channel) {
                        const float nearest = nearest_existing_channels[
                            static_cast<std::size_t>(channel)]
                                .at<float>(canvas_y, canvas_x);
                        const float current = static_cast<float>(
                            corrected_color[static_cast<std::size_t>(channel)])
                            / 255.0f;
                        corrected_color[static_cast<std::size_t>(channel)] =
                            static_cast<std::uint8_t>(std::clamp(
                                static_cast<int>(std::lround(
                                    (nearest * old_weight
                                        + current * (1.0f - old_weight))
                                    * 255.0f)),
                                0,
                                255));
                    }
                }
            }
            const std::uint8_t red = static_cast<std::uint8_t>(std::clamp(
                static_cast<int>(std::lround(
                    static_cast<double>(old_color[0]) * (1.0 - color_alpha)
                    + static_cast<double>(corrected_color[0]) * color_alpha)),
                0,
                255));
            const std::uint8_t green = static_cast<std::uint8_t>(std::clamp(
                static_cast<int>(std::lround(
                    static_cast<double>(old_color[1]) * (1.0 - color_alpha)
                    + static_cast<double>(corrected_color[1]) * color_alpha)),
                0,
                255));
            const std::uint8_t blue = static_cast<std::uint8_t>(std::clamp(
                static_cast<int>(std::lround(
                    static_cast<double>(old_color[2]) * (1.0 - color_alpha)
                    + static_cast<double>(corrected_color[2]) * color_alpha)),
                0,
                255));
            SlotValue after;
            after.depth = already_valid
                ? before.depth * 0.90f + candidate_depth * 0.10f
                : candidate_depth;
            after.confidence = already_valid
                ? std::max(before.confidence, confidence)
                : confidence;
            // Match the joint three-view path's ownership rule: once a canvas
            // cell has anchor texture, side cameras may validate its geometry
            // but must not repeatedly rewrite its RGB. Re-blending calibrated
            // side colour every live frame washed the green floor toward grey.
            // A side camera still supplies colour for genuinely new coverage.
            after.rgba = already_valid
                ? before.rgba
                : pack_rgba(red, green, blue);
            after.last_update_frame = static_cast<std::uint32_t>(raw.frame_seq);
            after.valid = 1U;
            const SlotUpdate update{
                static_cast<std::uint32_t>(slot),
                after};
            const int previous_index = update_index[slot];
            if (previous_index < 0) {
                update_index[slot] = static_cast<int>(patch.updates.size());
                patch.updates.push_back(update);
                update_quality.push_back(confidence - depth_residual);
            } else if (confidence - depth_residual
                > update_quality[static_cast<std::size_t>(previous_index)]) {
                patch.updates[static_cast<std::size_t>(previous_index)] = update;
                update_quality[static_cast<std::size_t>(previous_index)] =
                    confidence - depth_residual;
            }
            if (observed[slot] == 0U) {
                observed[slot] = 1U;
                patch.observed_slots.push_back(static_cast<std::uint32_t>(slot));
            }
        }
    }

    patch.changed_ratio = slot_count == 0U ? 0.0f
        : static_cast<float>(static_cast<double>(patch.updates.size())
            / static_cast<double>(slot_count));
    result.metrics.changed_ratio = patch.changed_ratio;
    result.metrics.changed_point_count = static_cast<std::uint32_t>(
        patch.updates.size());
    result.metrics.group_fused_sources = patch.updates.size() >= 128U ? 1 : 0;
    result.metrics.group_rejected_sources = patch.updates.size() >= 128U ? 0 : 1;
    std::size_t valid_count = static_cast<std::size_t>(std::count(
        state.valid.begin(), state.valid.end(), static_cast<std::uint8_t>(1U)));
    for (const SlotUpdate& update : patch.updates) {
        if (update.slot_id < state.valid.size() && state.valid[update.slot_id] == 0U
            && update.after.valid != 0U) {
            ++valid_count;
        }
    }
    result.metrics.valid_point_count = static_cast<std::uint32_t>(valid_count);
    result.frame.status = patch.updates.size() >= 128U
        ? FrameStatus::Committed
        : FrameStatus::NoChange;
    result.frame.changed_point_count = result.metrics.changed_point_count;
    result.frame.valid_point_count = result.metrics.valid_point_count;
    result.patch = std::move(patch);
    result.has_patch = result.frame.status == FrameStatus::Committed;
    result.metrics.total_ms = total_timer.ms();
    return result;
}

CandidateCommit InferenceEngine::process_world_group(
    const RawFrame& raw,
    const CanvasState& state,
    const PreparedGroup& group,
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
    result.metrics.group_size = 3;
    result.metrics.group_stride = options_.group_stride;
    result.metrics.group_anchor_index = raw.group_anchor_index;
    result.metrics.forward_calls = 1;
    result.metrics.forward_batch_size = 1;
    result.metrics.forward_sequence_size = 3;
    result.metrics.group_fused_sources = 3;
    result.metrics.group_rejected_sources = 0;
    result.metrics.model_input_width = options_.group_width;
    result.metrics.model_input_height = options_.group_height;
    result.metrics.roi_width = options_.group_width;
    result.metrics.roi_height = options_.group_height;

    Timer model_timer;
    const std::vector<Prediction> predictions = run_group_model(group.model_rgb_f);
    result.metrics.model_ms = model_timer.ms();
    if (predictions.size() != 3U) {
        throw std::runtime_error("B=1,S=3 group model did not return three predictions");
    }

    std::array<GroupWorldView, 3> views;
    for (std::size_t view = 0; view < 3U; ++view) {
        const Prediction& prediction = predictions[view];
        views[view].world_points = prediction.world_points;
        views[view].world_confidence = prediction.world_points_confidence;
        views[view].rgb = group.model_rgb_f[view];
        views[view].translation = prediction.pose_translation;
        views[view].quaternion = prediction.pose_quaternion;
        views[view].fov_h = prediction.fov_h;
        views[view].fov_w = prediction.fov_w;
        views[view].has_pose = prediction.has_pose;
    }

    Timer patch_timer;
    const GroupWorldFusionResult fusion = group_world_fusion_.fuse(views, state);
    if (!fusion.accepted) {
        result.metrics.group_rejected_sources = 3;
        result.metrics.fallback = "world_fusion_rejected: " + fusion.rejection_reason;
        result.metrics.valid_point_count = static_cast<std::uint32_t>(std::count(
            state.valid.begin(), state.valid.end(), static_cast<std::uint8_t>(1U)));
        result.frame.status = FrameStatus::NoChange;
        result.frame.valid_point_count = result.metrics.valid_point_count;
        result.metrics.patch_ms = patch_timer.ms();
        result.metrics.total_ms = total_timer.ms();
        return result;
    }

    const std::size_t slot_count = state.slot_count();
    if (slot_count == 0U || !state.shape_valid()) {
        throw std::runtime_error("world group requires an initialized CanvasState");
    }
    CandidatePatch patch;
    patch.frame_seq = raw.frame_seq;
    patch.base_version = state.version;
    patch.width = state.width;
    patch.height = state.height;
    patch.initialize_canvas = !state.initialized;
    patch.changed_ratio = 0.0f;
    patch.scene_jump = false;
    patch.anchor_camera.depth_scale = -1.0f;
    // The preview is intentionally separate from geometry. Real GUI point
    // positions come solely from physical slot row/column plus state.depth.
    if (patch.initialize_canvas) {
        patch.anchor_rgba.assign(slot_count, 0U);
    }

    std::vector<std::uint8_t> desired_valid(slot_count, 0U);
    std::vector<FusedSlot> desired(slot_count);
    for (const FusedSlot& fused_slot : fusion.slots) {
        if (fused_slot.slot_id >= slot_count) {
            throw std::runtime_error("world fusion produced a slot outside CanvasState");
        }
        desired_valid[fused_slot.slot_id] = 1U;
        desired[fused_slot.slot_id] = fused_slot;
        patch.observed_slots.push_back(fused_slot.slot_id);
    }

    const auto color_difference = [](const std::uint32_t lhs, const std::uint32_t rhs) {
        const auto left = unpack_rgba(lhs);
        const auto right = unpack_rgba(rhs);
        return std::max({
            std::abs(static_cast<int>(left[0]) - static_cast<int>(right[0])),
            std::abs(static_cast<int>(left[1]) - static_cast<int>(right[1])),
            std::abs(static_cast<int>(left[2]) - static_cast<int>(right[2]))});
    };
    patch.updates.reserve(fusion.slots.size() + 128U);
    for (std::size_t slot = 0; slot < slot_count; ++slot) {
        if (desired_valid[slot] == 0U) {
            continue;
        }
        const FusedSlot& next = desired[slot];
        const SlotValue before = slot_value_at(state, static_cast<std::uint32_t>(slot));
        const bool old_valid = before.valid != 0U && std::isfinite(before.depth);
        const float z_delta = std::abs(before.depth - next.depth);
        const int rgb_delta = color_difference(before.rgba, next.rgba);
        const bool unchanged = old_valid
            && (next.floor
                ? (z_delta < 0.0001f && rgb_delta < 3)
                : (z_delta < 0.002f && rgb_delta < 6));
        if (unchanged) {
            continue;
        }
        SlotValue after = before;
        after.depth = next.depth;
        after.confidence = next.confidence;
        after.rgba = next.rgba;
        after.last_update_frame = static_cast<std::uint32_t>(raw.frame_seq);
        after.valid = 1U;
        patch.updates.push_back(SlotUpdate{
            static_cast<std::uint32_t>(slot), after});
    }

    // Every object layer is tied to its logical cell. Missing current object
    // clusters are explicit invalidations; persistent floor cells are not
    // cleared when a later view is temporarily occluded.
    const int logical_width = state.width / 2;
    const int logical_height = state.height / 2;
    for (int logical_y = 0; logical_y < logical_height; ++logical_y) {
        for (int logical_x = 0; logical_x < logical_width; ++logical_x) {
            for (int layer = 1; layer <= 3; ++layer) {
                const int dx = (layer == 1 || layer == 3) ? 1 : 0;
                const int dy = layer >= 2 ? 1 : 0;
                const int pixel_x = logical_x * 2 + dx;
                const int pixel_y = logical_y * 2 + dy;
                const std::uint32_t slot = static_cast<std::uint32_t>(
                    pixel_y * state.width + pixel_x);
                if (state.valid[slot] == 0U || desired_valid[slot] != 0U) {
                    continue;
                }
                SlotValue after = slot_value_at(state, slot);
                after.depth = 0.0f;
                after.confidence = 0.0f;
                after.rgba = 0U;
                after.last_update_frame = static_cast<std::uint32_t>(raw.frame_seq);
                after.valid = 0U;
                patch.updates.push_back(SlotUpdate{slot, after});
                patch.cleared_support_slots.push_back(slot);
            }
        }
    }

    bool support_change = false;
    for (const std::uint32_t slot : patch.observed_slots) {
        if (state.support[slot] == 0U) {
            support_change = true;
            break;
        }
    }
    patch.changed_ratio = slot_count > 0U
        ? static_cast<float>(patch.updates.size()) / static_cast<float>(slot_count)
        : 0.0f;
    std::size_t valid_count = static_cast<std::size_t>(std::count(
        state.valid.begin(), state.valid.end(), static_cast<std::uint8_t>(1U)));
    for (const SlotUpdate& update : patch.updates) {
        const bool old_valid = state.valid[update.slot_id] != 0U;
        const bool new_valid = update.after.valid != 0U;
        if (old_valid != new_valid) {
            if (new_valid) {
                ++valid_count;
            } else if (valid_count > 0U) {
                --valid_count;
            }
        }
    }
    result.metrics.changed_point_count = static_cast<std::uint32_t>(patch.updates.size());
    result.metrics.valid_point_count = static_cast<std::uint32_t>(valid_count);
    result.metrics.changed_ratio = patch.changed_ratio;
    result.metrics.support_changed_ratio = support_change ? 1.0 : 0.0;
    result.frame.changed_point_count = result.metrics.changed_point_count;
    result.frame.valid_point_count = result.metrics.valid_point_count;
    result.frame.changed_ratio = patch.changed_ratio;
    result.frame.status = patch.initialize_canvas || !patch.updates.empty() || support_change
        ? FrameStatus::Committed
        : FrameStatus::NoChange;
    result.patch = std::move(patch);
    result.has_patch = result.frame.status == FrameStatus::Committed;
    result.metrics.patch_ms = patch_timer.ms();
    result.metrics.total_ms = total_timer.ms();
    return result;
}

CandidateCommit InferenceEngine::process_impl(
    const RawFrame& raw,
    const CanvasState& state,
    const FrameImage& frame,
    const PreparedGroup* prepared_group,
    const bool observation_group,
    const double read_ms,
    const cv::Mat* forced_homography) {
    Timer total_timer;
    CandidateCommit result;
    result.frame.frame_seq = raw.frame_seq;
    result.frame.base_version = state.version;
    result.frame.commit_version = state.version;
    result.frame.image_name = raw.path.filename().string();
    result.metrics.frame_seq = raw.frame_seq;
    result.metrics.image = result.frame.image_name;

    result.metrics.read_ms = read_ms;
    const bool logical_group = prepared_group != nullptr || observation_group;
    result.metrics.group_size = logical_group ? 3 : 1;
    result.metrics.group_stride = logical_group ? options_.group_stride : 1;
    if (prepared_group != nullptr
        && (!state.initialized
            || group_gap_protected_.size() != cv::Size(state.width, state.height))) {
        group_gap_protected_ = cv::Mat::zeros(state.height, state.width, CV_8UC1);
    }
    result.metrics.group_anchor_index = logical_group ? raw.group_anchor_index : 0;
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
    const bool has_forced_homography = forced_homography != nullptr
        && !forced_homography->empty();
    cv::Mat homography = has_forced_homography
        ? forced_homography->clone()
        : estimate_homography(frame, state, metrics);
    if (observation_group) {
        // The three cameras and the anchor selection are fixed. Keep the
        // anchor image centred at a stable reduced scale so the surrounding
        // cameras have real atlas space instead of being clipped at the
        // 770x630 canvas border.
        constexpr float kObservationAtlasScale = 0.68f;
        homography = cv::Mat::eye(3, 3, CV_32FC1);
        homography.at<float>(0, 0) = kObservationAtlasScale;
        homography.at<float>(1, 1) = kObservationAtlasScale;
        homography.at<float>(0, 2) =
            0.5f * (1.0f - kObservationAtlasScale)
            * static_cast<float>(state.width);
        homography.at<float>(1, 2) =
            0.5f * (1.0f - kObservationAtlasScale)
            * static_cast<float>(state.height);
        metrics.fallback.clear();
        metrics.homography_inliers = 12;
        metrics.homography_error_px = 0.0;
    }
    if (has_forced_homography) {
        // The transform was computed before the Canvas was updated, against
        // the two real source images in this group. Do not run temporal
        // fallback or phase refinement on a fixed-camera side view.
        metrics.fallback.clear();
        metrics.homography_inliers = 12;
        metrics.homography_error_px = 0.0;
    }
    if (!observation_group && !has_forced_homography && state.initialized
        && !metrics.fallback.empty() && !last_homography_.empty()) {
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
    if (!observation_group && !has_forced_homography && state.initialized
        && metrics.fallback.empty()) {
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
        const int anchor_index = std::clamp(raw.group_anchor_index, 0, 2);
        // A three-image batch is a single inference call, but its three
        // views can span a large, non-convex part of the rotating aperture.
        // Using that union as one model ROI creates a wide bounding box with
        // unsupported strips between the views; those strips become visible
        // holes in the accumulated canvas.  Keep the anchor view as the
        // authoritative geometry footprint.  The other two views remain in
        // the batch as same-forward context and their calibrated depth is
        // considered only inside that footprint.
        const cv::Mat anchor_valid = grouped_valid_warp[static_cast<std::size_t>(anchor_index)].clone();
        warped_rgb_f = grouped_warped_rgb_f[static_cast<std::size_t>(anchor_index)].clone();
        valid_warp = anchor_valid.clone();
        // Do not paint side-view RGB into the canvas.  Their rotated support
        // rectangles are valid model context but are not valid geometry
        // ownership; compositing them here would expose black/colour wedges
        // in debug images and could reintroduce a second colour layer.
        const cv::Mat canvas_support = state.initialized
            ? state_mask(state.support, state.width, state.height)
            : cv::Mat::zeros(canvas_size, CV_8UC1);
        cv::Mat group_support_change;
        cv::bitwise_not(canvas_support, group_support_change);
        cv::bitwise_and(anchor_valid, group_support_change, group_support_change);
        group_support_change = dilate_mask(group_support_change, options_.dilate_ksize);
        cv::bitwise_and(group_support_change, anchor_valid, group_support_change);
        if (state.initialized) {
            cv::bitwise_or(support_change, group_support_change, support_change);
        } else {
            support_change = anchor_valid.clone();
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
            const cv::Point origin = content_origin(
                frame.support,
                std::max(32, static_cast<int>(std::round(options_.width * 0.05))),
                std::max(128, static_cast<int>(std::round(options_.width * 0.18))));
            roi = cv::Rect(
                std::clamp(origin.x, 0, std::max(0, state.width - 1)),
                std::clamp(origin.y, 0, std::max(0, state.height - 1)),
                std::min(source_width, state.width - std::clamp(origin.x, 0, std::max(0, state.width - 1))),
                std::min(source_height, state.height - std::clamp(origin.y, 0, std::max(0, state.height - 1))));
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
        const cv::Point origin = content_origin(
            frame.support,
            std::max(32, static_cast<int>(std::round(options_.width * 0.05))),
            std::max(128, static_cast<int>(std::round(options_.width * 0.18))));
        model_to_canvas.at<float>(0, 0) = static_cast<float>(frame.match_rgb_u8.cols)
            / static_cast<float>(first_width);
        model_to_canvas.at<float>(1, 1) = static_cast<float>(frame.match_rgb_u8.rows)
            / static_cast<float>(first_height);
        model_to_canvas.at<float>(0, 2) = static_cast<float>(origin.x);
        model_to_canvas.at<float>(1, 2) = static_cast<float>(origin.y);
        if (observation_group) {
            model_to_canvas = homography * model_to_canvas;
        }
        model_support = cv::Mat::ones(first_height, first_width, CV_32FC1);
    } else if (observation_group) {
        // A non-overlapping three-image input group is one static logical
        // frame, not the next item in the model's temporal S=2 sequence.
        // Reusing the anchor/current pair graph here makes the pair model
        // explain the camera-view change as a second depth layer.  Keep the
        // fast single-image graph, but project its current anchor result
        // through the accepted current-to-canvas homography.
        const int first_width = options_.first_model_width;
        const int first_height = options_.first_model_height;
        cv::Mat current_model;
        cv::resize(
            frame.match_rgb_f,
            current_model,
            cv::Size(first_width, first_height),
            0.0,
            0.0,
            cv::INTER_AREA);
        model_inputs.push_back(std::move(current_model));
        result.metrics.roi_width = first_width;
        result.metrics.roi_height = first_height;
        result.metrics.model_input_width = first_width;
        result.metrics.model_input_height = first_height;
        const cv::Point origin = content_origin(
            frame.support,
            std::max(32, static_cast<int>(std::round(options_.width * 0.05))),
            std::max(128, static_cast<int>(std::round(options_.width * 0.18))));
        cv::Mat source_to_frame = cv::Mat::eye(3, 3, CV_32FC1);
        source_to_frame.at<float>(0, 0) = static_cast<float>(frame.match_rgb_f.cols)
            / static_cast<float>(first_width);
        source_to_frame.at<float>(1, 1) = static_cast<float>(frame.match_rgb_f.rows)
            / static_cast<float>(first_height);
        source_to_frame.at<float>(0, 2) = static_cast<float>(origin.x);
        source_to_frame.at<float>(1, 2) = static_cast<float>(origin.y);
        model_to_canvas = homography * source_to_frame;
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
    const int output_frame_index = state.initialized && !observation_group ? 1 : 0;
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
    if (!state.initialized && !observation_group) {
        release_single_model_after_first_frame();
    }

    const cv::Size canvas_size(state.width, state.height);
    cv::Mat model_depth_canvas = warp_like(
        prediction.depth, model_to_canvas, canvas_size, cv::INTER_LINEAR);
    cv::Mat model_confidence_canvas = warp_like(
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
    if (observation_group && cv::countNonZero(candidate_valid) >= 512) {
        cv::Mat depth_consistent = candidate_valid.clone();
        for (int y = 2; y < model_depth_canvas.rows - 2; ++y) {
            for (int x = 2; x < model_depth_canvas.cols - 2; ++x) {
                if (candidate_valid.at<std::uint8_t>(y, x) == 0U) {
                    continue;
                }
                const float depth = model_depth_canvas.at<float>(y, x);
                if (!std::isfinite(depth) || depth <= 1e-5f) {
                    depth_consistent.at<std::uint8_t>(y, x) = 0U;
                    continue;
                }
                int agreeing_neighbors = 0;
                for (int dy = -2; dy <= 2; dy += 2) {
                    for (int dx = -2; dx <= 2; dx += 2) {
                        if (dx == 0 && dy == 0) {
                            continue;
                        }
                        const int nx = x + dx;
                        const int ny = y + dy;
                        if (candidate_valid.at<std::uint8_t>(ny, nx) == 0U) {
                            continue;
                        }
                        const float neighbor_depth =
                            model_depth_canvas.at<float>(ny, nx);
                        if (!std::isfinite(neighbor_depth)
                            || neighbor_depth <= 1e-5f) {
                            continue;
                        }
                        const float relative_difference =
                            std::abs(neighbor_depth - depth)
                            / std::max(
                                std::max(std::abs(neighbor_depth), std::abs(depth)),
                                1e-3f);
                        if (relative_difference <= 0.18f) {
                            ++agreeing_neighbors;
                        }
                    }
                }
                if (agreeing_neighbors < 2) {
                    depth_consistent.at<std::uint8_t>(y, x) = 0U;
                }
            }
        }
        candidate_valid = std::move(depth_consistent);
    }
    cv::Mat group_gap_fill = cv::Mat::zeros(candidate_valid.size(), CV_8UC1);
    cv::Mat group_rgb_source_valid = candidate_valid.clone();
    if (prepared_group != nullptr) {
        // B3S1 can leave a thin strip in the authoritative canvas support
        // without writing geometry there.  Detect only old-support holes so
        // this cannot turn a genuinely new outer aperture into a surface.
        cv::Mat group_support = state.initialized
            ? state_mask(state.support, state.width, state.height)
            : cv::Mat::zeros(valid_warp.size(), CV_8UC1);
        cv::bitwise_and(group_support, valid_warp, group_support);
        cv::Mat group_detection_valid = canvas_valid.clone();
        cv::Mat group_hole;
        cv::Mat not_group_detection_valid;
        cv::bitwise_not(group_detection_valid, not_group_detection_valid);
        cv::bitwise_and(group_support, not_group_detection_valid, group_hole);
        cv::Mat not_group_hole;
        cv::bitwise_not(group_hole, not_group_hole);
        cv::bitwise_and(group_rgb_source_valid, not_group_hole, group_rgb_source_valid);
        group_gap_fill = fill_group_narrow_gaps(
            model_depth_canvas,
            model_confidence_canvas,
            group_detection_valid,
            group_support,
            group_rgb_source_valid);
        // Keep only the part of the support hole that lies on the anchor
        // silhouette.  The old-support component can extend into the dark
        // aperture; committing that full rectangle would replace a natural
        // curved outline with a straight horizontal RGB/depth edge.
        for (int y = 0; y < group_gap_fill.rows; ++y) {
            for (int x = 0; x < group_gap_fill.cols; ++x) {
                if (group_gap_fill.at<std::uint8_t>(y, x) != 0U
                    && std::max({
                        warped_rgb_f.at<cv::Vec3f>(y, x)[0],
                        warped_rgb_f.at<cv::Vec3f>(y, x)[1],
                        warped_rgb_f.at<cv::Vec3f>(y, x)[2]}) < 0.08f) {
                    group_gap_fill.at<std::uint8_t>(y, x) = 0U;
                }
            }
        }
        group_gap_fill = expand_group_gap_edges(
            model_depth_canvas,
            model_confidence_canvas,
            group_detection_valid,
            group_gap_fill,
            group_support,
            canvas_valid,
            static_cast<float>(options_.min_conf));
        cv::bitwise_or(candidate_valid, group_gap_fill, candidate_valid);
        if (cv::countNonZero(group_gap_fill) > 0) {
            cv::bitwise_or(group_gap_protected_, group_gap_fill, group_gap_protected_);
        }
    }

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

    if (prepared_group != nullptr && state.initialized && cv::countNonZero(quality_valid) >= 512) {
        cv::Mat median_input = aligned_depth.clone();
        const cv::Mat inverse_quality = [&]() {
            cv::Mat inverse;
            cv::bitwise_not(quality_valid, inverse);
            return inverse;
        }();
        median_input.setTo(0.0f, inverse_quality);
        cv::Mat local_median;
        cv::medianBlur(median_input, local_median, 5);
        cv::Mat residual = aligned_depth - local_median;
        std::vector<float> residual_values;
        residual_values.reserve(static_cast<std::size_t>(cv::countNonZero(quality_valid)));
        for (int y = 0; y < quality_valid.rows; ++y) {
            for (int x = 0; x < quality_valid.cols; ++x) {
                if (quality_valid.at<std::uint8_t>(y, x) != 0U
                    && std::isfinite(residual.at<float>(y, x))) {
                    residual_values.push_back(residual.at<float>(y, x));
                }
            }
        }
        if (residual_values.size() >= 512U) {
            const float center = median_value(residual_values);
            std::vector<float> deviations;
            deviations.reserve(residual_values.size());
            for (const float value : residual_values) {
                deviations.push_back(std::abs(value - center));
            }
            const float limit = std::max(
                0.006f,
                4.0f * 1.4826f * std::max(median_value(deviations), 1e-6f));
            cv::Mat neighborhood;
            cv::Mat quality_f;
            quality_valid.convertTo(quality_f, CV_32FC1, 1.0 / 255.0);
            cv::blur(quality_f, neighborhood, cv::Size(5, 5), cv::Point(-1, -1), cv::BORDER_REPLICATE);
            for (int y = 0; y < quality_valid.rows; ++y) {
                for (int x = 0; x < quality_valid.cols; ++x) {
                    if (quality_valid.at<std::uint8_t>(y, x) != 0U
                        && neighborhood.at<float>(y, x) > 0.6f
                        && std::abs(residual.at<float>(y, x) - center) > limit) {
                        quality_valid.at<std::uint8_t>(y, x) = 0U;
                        model_valid.at<std::uint8_t>(y, x) = 0U;
                    }
                }
            }
        }
    }

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
    if (prepared_group != nullptr && !group_gap_protected_.empty()) {
        cv::Mat not_group_gap_protected;
        cv::bitwise_not(group_gap_protected_, not_group_gap_protected);
        cv::bitwise_and(update_mask, not_group_gap_protected, update_mask);
    }
    // A grouped model can expose a previously unsupported strip only after
    // the current sliding window enlarges the anchor footprint.  The strip
    // is not a photometric/change-mask pixel yet, but its nearest-filled
    // depth/color is an explicit geometry repair and must be committed.
    if (!group_gap_fill.empty()) {
        cv::bitwise_or(update_mask, group_gap_fill, update_mask);
    }

    // The B=1,S=3 group model owns a complete non-overlapping footprint and
    // may replace it between groups.  The S1/S2 observation path does not:
    // its foreground mask is allowed to flicker with camera exposure, so
    // clearing old slots here would turn a harmless mask change into a large
    // point-cloud jump.  Observation fusion only adds validated side-view
    // coverage and keeps the stable Canvas history.
    cv::Mat group_stale_clear = cv::Mat::zeros(update_mask.size(), CV_8UC1);
    const bool replace_group_footprint =
        prepared_group != nullptr && options_.group_stride >= 3;
    if (replace_group_footprint && state.initialized && result.metrics.fallback.empty()) {
        cv::Mat current_footprint;
        cv::bitwise_or(valid_warp, group_gap_fill, current_footprint);
        const cv::Mat halo_kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, cv::Size(7, 7));
        cv::dilate(current_footprint, current_footprint, halo_kernel);
        cv::Mat outside_current;
        cv::bitwise_not(current_footprint, outside_current);
        cv::bitwise_and(canvas_valid, outside_current, group_stale_clear);
        group_stale_clear = filter_components(group_stale_clear, 256);
    }

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
        const int bridge_extent = prepared_group != nullptr ? 7 : 65;
        const cv::Mat bridge_kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE, cv::Size(bridge_extent, bridge_extent));
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
                    const float mix_radius = prepared_group != nullptr ? 3.0f : 8.0f;
                    color_bridge_mix.at<float>(y, x) = std::clamp(
                        distance_to_ring.at<float>(y, x) / mix_radius, 0.0f, 1.0f);
                }
            }
        }
    } else if (state.initialized) {
        canvas_rgb = !live_rgb_float_.empty()
            && live_rgb_float_.size() == cv::Size(state.width, state.height)
            ? live_rgb_float_
            : state_rgb_float(state);
    }

    // A repaired grouped strip is geometry-authoritative, but its RGB must
    // follow later grouped views as the surrounding surface is refreshed.
    // Commit that RGB as a color-only update; never let a later model depth
    // prediction overwrite the protected geometry.
    if constexpr (false) {
        // Refresh the full existing anchor overlap as color-only updates.  A
        // repaired strip must not retain the exposure/texture layer from the
        // group that first exposed its geometry; geometry remains protected
        // by group_gap_protected_ above.
        cv::Mat group_color_refresh;
        cv::bitwise_and(valid_warp, canvas_valid, group_color_refresh);
        if (!group_gap_protected_.empty()) {
            cv::bitwise_or(group_color_refresh, group_gap_protected_, group_color_refresh);
            cv::bitwise_and(group_color_refresh, canvas_valid, group_color_refresh);
        }
        cv::bitwise_or(color_bridge_mask, group_color_refresh, color_bridge_mask);
        cv::Mat anchor_distance;
        cv::distanceTransform(valid_warp, anchor_distance, cv::DIST_L2, 3);
        for (int y = 0; y < group_color_refresh.rows; ++y) {
            for (int x = 0; x < group_color_refresh.cols; ++x) {
                if (group_color_refresh.at<std::uint8_t>(y, x) != 0U) {
                    const bool protected_gap = !group_gap_protected_.empty()
                        && group_gap_protected_.at<std::uint8_t>(y, x) != 0U;
                    color_bridge_mix.at<float>(y, x) = protected_gap
                        ? 1.0f
                        : std::clamp(anchor_distance.at<float>(y, x) / 16.0f, 0.0f, 1.0f);
                }
            }
        }
    }

    cv::Mat color_apply_mask;
    cv::bitwise_or(update_mask, color_bridge_mask, color_apply_mask);
    // Group mode keeps the anchor as the only RGB source, but still needs the
    // global overlap correction because each sliding window has a new anchor
    // exposure/view.  It must not use the single-view hard old-overlap blend:
    // that blend turns a narrow rotated support edge into a long rectangle.
    cv::Mat fused_rgb;
    fused_rgb = state.initialized
        ? anchor_texture_transfer(
            warped_rgb_f,
            canvas_rgb,
            canvas_valid,
            valid_warp,
            color_apply_mask,
            support_change,
            anchor_ring,
            prepared_group == nullptr)
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

    // The grouped model may expose a thin strip whose current RGB exposure
    // differs from the committed rows immediately above and below it.  Keep
    // the current texture, but apply a per-row low-frequency colour offset so
    // the repaired strip joins those committed boundary colours without
    // creating a copied/striped texture layer.
    if constexpr (false) {
        cv::Mat gap_labels;
        cv::Mat gap_stats;
        cv::Mat gap_centroids;
        const int gap_components = cv::connectedComponentsWithStats(
            group_gap_protected_, gap_labels, gap_stats, gap_centroids, 8, CV_32S);
        cv::Mat not_gap_protected;
        cv::bitwise_not(group_gap_protected_, not_gap_protected);
        cv::Mat canvas_boundary_valid;
        cv::bitwise_and(canvas_valid, not_gap_protected, canvas_boundary_valid);
        cv::Mat valid_f;
        valid_warp.convertTo(valid_f, CV_32FC1, 1.0 / 255.0);
        std::vector<cv::Mat> local_channels;
        cv::split(fused_rgb, local_channels);
        cv::Mat local_denominator;
        cv::GaussianBlur(valid_f, local_denominator, cv::Size(9, 9), 0.0, 0.0);
        for (cv::Mat& channel : local_channels) {
            cv::Mat weighted;
            cv::multiply(channel, valid_f, weighted);
            cv::GaussianBlur(weighted, weighted, cv::Size(9, 9), 0.0, 0.0);
            cv::Mat safe_denominator;
            cv::add(local_denominator, cv::Scalar(1e-5), safe_denominator);
            cv::divide(weighted, safe_denominator, channel);
        }
        cv::Mat local_rgb;
        cv::merge(local_channels, local_rgb);
        for (int component = 1; component < gap_components; ++component) {
            const int left = gap_stats.at<int>(component, cv::CC_STAT_LEFT);
            const int top = gap_stats.at<int>(component, cv::CC_STAT_TOP);
            const int width = gap_stats.at<int>(component, cv::CC_STAT_WIDTH);
            const int height = gap_stats.at<int>(component, cv::CC_STAT_HEIGHT);
            const bool horizontal = width >= height;
            const int first = horizontal ? left : top;
            const int last = horizontal ? left + width : top + height;
            const int span = std::max(1, last - first);
            int before = horizontal ? top - 1 : left - 1;
            int after = horizontal ? top + height : left + width;
            const int before_limit = horizontal ? 0 : 0;
            const int after_limit = horizontal
                ? canvas_valid.rows - 1
                : canvas_valid.cols - 1;
            auto boundary_coverage = [&](const int coordinate) {
                int count = 0;
                for (int offset = 0; offset < span; ++offset) {
                    const int along = first + offset;
                    const int x = horizontal ? along : coordinate;
                    const int y = horizontal ? coordinate : along;
                    count += canvas_boundary_valid.at<std::uint8_t>(y, x) != 0U ? 1 : 0;
                }
                return count;
            };
            const int before_canvas_start = before;
            const int after_canvas_start = after;
            while (before >= before_limit && boundary_coverage(before) < span / 2) {
                --before;
            }
            while (after <= after_limit && boundary_coverage(after) < span / 2) {
                ++after;
            }
            const bool before_from_canvas = before >= before_limit;
            const bool after_from_canvas = after <= after_limit;
            auto current_coverage = [&](const int coordinate) {
                int count = 0;
                for (int offset = 0; offset < span; ++offset) {
                    const int along = first + offset;
                    const int x = horizontal ? along : coordinate;
                    const int y = horizontal ? coordinate : along;
                    count += valid_warp.at<std::uint8_t>(y, x) != 0U ? 1 : 0;
                }
                return count;
            };
            if (!before_from_canvas) {
                before = before_canvas_start;
                while (before >= before_limit && current_coverage(before) < span / 2) {
                    --before;
                }
            }
            if (!after_from_canvas) {
                after = after_canvas_start;
                while (after <= after_limit && current_coverage(after) < span / 2) {
                    ++after;
                }
            }
            if (before < before_limit || after > after_limit) {
                continue;
            }
            auto boundary_mean = [&](const cv::Mat& mask, const cv::Mat& image,
                                     const int coordinate) {
                cv::Vec3f sum(0.0f, 0.0f, 0.0f);
                int count = 0;
                for (int offset = 0; offset < span; ++offset) {
                    const int along = first + offset;
                    const int x = horizontal ? along : coordinate;
                    const int y = horizontal ? coordinate : along;
                    if (mask.at<std::uint8_t>(y, x) == 0U) {
                        continue;
                    }
                    sum += image.at<cv::Vec3f>(y, x);
                    ++count;
                }
                return count > 0 ? sum * (1.0f / static_cast<float>(count))
                                 : cv::Vec3f(0.0f, 0.0f, 0.0f);
            };
            const cv::Vec3f before_mean = before_from_canvas
                ? boundary_mean(canvas_boundary_valid, canvas_rgb, before)
                : boundary_mean(valid_warp, fused_rgb, before);
            const cv::Vec3f after_mean = after_from_canvas
                ? boundary_mean(canvas_boundary_valid, canvas_rgb, after)
                : boundary_mean(valid_warp, fused_rgb, after);
            for (int y = top; y < top + height; ++y) {
                for (int x = left; x < left + width; ++x) {
                    if (gap_labels.at<int>(y, x) != component) {
                        continue;
                    }
                    cv::Vec3f current_sum(0.0f, 0.0f, 0.0f);
                    int current_count = 0;
                    for (int along = first; along < last; ++along) {
                        const int px = horizontal ? along : x;
                        const int py = horizontal ? y : along;
                        if (gap_labels.at<int>(py, px) != component
                            || valid_warp.at<std::uint8_t>(py, px) == 0U) {
                            continue;
                        }
                        current_sum += fused_rgb.at<cv::Vec3f>(py, px);
                        ++current_count;
                    }
                    if (current_count == 0) {
                        continue;
                    }
                    const int coordinate = horizontal ? y : x;
                    const float alpha = std::clamp(
                        static_cast<float>(coordinate - before)
                            / static_cast<float>(after - before),
                        0.0f,
                        1.0f);
                    const cv::Vec3f target = before_mean * (1.0f - alpha)
                        + after_mean * alpha;
                    const cv::Vec3f current_mean = current_sum
                        * (1.0f / static_cast<float>(current_count));
                    cv::Vec3f current_value = fused_rgb.at<cv::Vec3f>(y, x);
                    bool has_current = valid_warp.at<std::uint8_t>(y, x) != 0U
                        && std::max({current_value[0], current_value[1], current_value[2]}) >= 0.08f;
                    if (!has_current) {
                        if (local_denominator.at<float>(y, x) > 0.05f) {
                            current_value = local_rgb.at<cv::Vec3f>(y, x);
                            has_current = true;
                        }
                    }
                    if (!has_current) {
                        for (int distance = 1; distance < std::max(width, height); ++distance) {
                            const int before_x = horizontal ? x - distance : x;
                            const int before_y = horizontal ? y : y - distance;
                            const int after_x = horizontal ? x + distance : x;
                            const int after_y = horizontal ? y : y + distance;
                            if (before_x >= left && before_x < left + width
                                && before_y >= top && before_y < top + height
                                && valid_warp.at<std::uint8_t>(before_y, before_x) != 0U) {
                                current_value = fused_rgb.at<cv::Vec3f>(before_y, before_x);
                                has_current = true;
                                break;
                            }
                            if (after_x >= left && after_x < left + width
                                && after_y >= top && after_y < top + height
                                && valid_warp.at<std::uint8_t>(after_y, after_x) != 0U) {
                                current_value = fused_rgb.at<cv::Vec3f>(after_y, after_x);
                                has_current = true;
                                break;
                            }
                        }
                    }
                    cv::Vec3f value = has_current
                        ? current_value + target - current_mean
                        : target;
                    for (int channel = 0; channel < 3; ++channel) {
                        value[channel] = std::clamp(value[channel], 0.0f, 1.0f);
                    }
                    fused_rgb.at<cv::Vec3f>(y, x) = value;
                }
            }
        }
    }

    // In grouped mode an existing canvas colour is authoritative.  The next
    // three-image window may have a different exposure or ROI, but it must
    // not rewrite the already committed RGB layer and create a horizontal
    // colour band.  Newly exposed cells are not in canvas_valid and keep the
    // current anchor RGB.
    // Grouped mode refreshes the current foreground anchor above.  The old
    // implementation copied every canvas-valid pixel back from canvas_rgb
    // here, which resurrected the stale rectangular texture layer immediately
    // before the patch was serialized.  Keep that legacy block disabled; the
    // single-image path does not enter it.
    if constexpr (false) {
        cv::Mat group_not_old_valid;
        cv::bitwise_not(canvas_valid, group_not_old_valid);
        cv::Mat group_new_mask_for_feather;
        cv::bitwise_and(valid_warp, group_not_old_valid, group_new_mask_for_feather);
        std::vector<cv::Mat> group_current_channels;
        cv::split(fused_rgb, group_current_channels);
        std::vector<cv::Mat> group_nearest_new_channels;
        group_nearest_new_channels.reserve(group_current_channels.size());
        for (const cv::Mat& channel : group_current_channels) {
            group_nearest_new_channels.push_back(
                nearest_fill_values(channel, group_new_mask_for_feather));
        }
        for (int y = 0; y < fused_rgb.rows; ++y) {
            for (int x = 0; x < fused_rgb.cols; ++x) {
                if (canvas_valid.at<std::uint8_t>(y, x) != 0U) {
                    fused_rgb.at<cv::Vec3f>(y, x) = canvas_rgb.at<cv::Vec3f>(y, x);
                }
            }
        }

        // Newly exposed geometry still needs to meet the old canvas without
        // an exposure step.  Estimate the old low-frequency colour field and
        // feather it into only the new cells; the committed canvas itself is
        // left untouched by this correction.
        cv::Mat old_valid_f;
        canvas_valid.convertTo(old_valid_f, CV_32FC1, 1.0 / 255.0);
        cv::Mat old_denominator;
        cv::GaussianBlur(old_valid_f, old_denominator, cv::Size(), 16.0, 16.0);
        std::vector<cv::Mat> old_fields;
        cv::split(canvas_rgb, old_fields);
        for (cv::Mat& channel : old_fields) {
            cv::Mat weighted;
            cv::multiply(channel, old_valid_f, weighted);
            cv::GaussianBlur(weighted, weighted, cv::Size(), 16.0, 16.0);
            cv::Mat safe_denominator;
            cv::add(old_denominator, cv::Scalar(1e-5), safe_denominator);
            cv::divide(weighted, safe_denominator, channel);
        }
        cv::Mat old_field;
        cv::merge(old_fields, old_field);
        cv::Mat not_old_valid;
        cv::bitwise_not(canvas_valid, not_old_valid);
        cv::Mat distance_to_old;
        cv::distanceTransform(not_old_valid, distance_to_old, cv::DIST_L2, 3);
        cv::Mat new_color_mask;
        // Use every newly exposed model-supported pixel for the feather.  A
        // change-mask-only selection can leave the first row outside
        // update_mask, which is exactly where a thin RGB seam survives.
        cv::bitwise_and(valid_warp, not_old_valid, new_color_mask);
        for (int y = 0; y < fused_rgb.rows; ++y) {
            for (int x = 0; x < fused_rgb.cols; ++x) {
                if (new_color_mask.at<std::uint8_t>(y, x) == 0U
                    || old_denominator.at<float>(y, x) <= 0.05f) {
                    continue;
                }
                const float alpha = std::clamp(
                    0.45f + distance_to_old.at<float>(y, x) / 40.0f,
                    0.45f,
                    1.0f);
                fused_rgb.at<cv::Vec3f>(y, x) =
                    old_field.at<cv::Vec3f>(y, x) * (1.0f - alpha)
                    + fused_rgb.at<cv::Vec3f>(y, x) * alpha;
            }
        }
        feather_group_new_rgb_to_canvas(
            fused_rgb, canvas_rgb, canvas_valid, new_color_mask, 8);
        // Symmetrically feather the old-canvas side.  Locking only the old
        // pixels leaves a one-pixel exposure step at the new/old boundary.
        cv::Mat not_group_new;
        cv::bitwise_not(group_new_mask_for_feather, not_group_new);
        cv::Mat distance_to_new;
        cv::distanceTransform(not_group_new, distance_to_new, cv::DIST_L2, 3);
        for (int y = 0; y < fused_rgb.rows; ++y) {
            for (int x = 0; x < fused_rgb.cols; ++x) {
                if (canvas_valid.at<std::uint8_t>(y, x) == 0U) {
                    continue;
                }
                const float distance = distance_to_new.at<float>(y, x);
                if (!std::isfinite(distance) || distance > 8.0f) {
                    continue;
                }
                const float old_weight = std::clamp(
                    0.5f + (distance - 1.0f) / 14.0f, 0.5f, 1.0f);
                cv::Vec3f value = canvas_rgb.at<cv::Vec3f>(y, x);
                for (int channel = 0; channel < 3; ++channel) {
                    value[channel] = value[channel] * old_weight
                        + group_nearest_new_channels[channel].at<float>(y, x)
                            * (1.0f - old_weight);
                }
                fused_rgb.at<cv::Vec3f>(y, x) = value;
            }
        }
    }

    // A newly repaired narrow support strip has no old RGB samples.  Fill it
    // once from the surrounding anchor texture; the protected mask prevents
    // subsequent grouped windows from changing it again.
    if (prepared_group != nullptr && cv::countNonZero(group_gap_fill) > 0) {
        // First propagate colours along the strip's short (vertical) axis.
        // The warped RGB support can start several rows below the geometric
        // support; ordinary inpainting then sees black background above the
        // object and reproduces the very band we are removing.  Only pixels
        // outside the repair mask and brighter than the background are used
        // as sources.
        cv::Mat source = cv::Mat::zeros(group_gap_fill.size(), CV_8UC1);
        for (int y = 0; y < fused_rgb.rows; ++y) {
            for (int x = 0; x < fused_rgb.cols; ++x) {
                if (group_gap_fill.at<std::uint8_t>(y, x) != 0U) {
                    continue;
                }
                const cv::Vec3f value = fused_rgb.at<cv::Vec3f>(y, x);
                if (std::max({value[0], value[1], value[2]}) >= 0.08f) {
                    source.at<std::uint8_t>(y, x) = 255U;
                }
            }
        }
        cv::Mat repaired = fused_rgb.clone();
        cv::Mat gap_labels;
        cv::Mat gap_stats;
        cv::Mat gap_centroids;
        const int gap_components = cv::connectedComponentsWithStats(
            group_gap_fill, gap_labels, gap_stats, gap_centroids, 8, CV_32S);
        for (int component = 1; component < gap_components; ++component) {
            const int left = gap_stats.at<int>(component, cv::CC_STAT_LEFT);
            const int top = gap_stats.at<int>(component, cv::CC_STAT_TOP);
            const int width = gap_stats.at<int>(component, cv::CC_STAT_WIDTH);
            const int height = gap_stats.at<int>(component, cv::CC_STAT_HEIGHT);
            const bool horizontal = width >= height;
            const int span = std::max(1, horizontal ? height : width);
            for (int y = top; y < top + height; ++y) {
                for (int x = left; x < left + width; ++x) {
                    if (gap_labels.at<int>(y, x) != component) {
                        continue;
                    }
                    int before = horizontal ? y - span : x - span;
                    while (before >= 0) {
                        const int sx = horizontal ? x : before;
                        const int sy = horizontal ? before : y;
                        if (source.at<std::uint8_t>(sy, sx) != 0U) {
                            break;
                        }
                        --before;
                    }
                    int after = horizontal
                        ? y + span
                        : x + span;
                    const int after_limit = horizontal
                        ? fused_rgb.rows
                        : fused_rgb.cols;
                    while (after < after_limit) {
                        const int sx = horizontal ? x : after;
                        const int sy = horizontal ? after : y;
                        if (source.at<std::uint8_t>(sy, sx) != 0U) {
                            break;
                        }
                        ++after;
                    }
                    cv::Vec3f value(0.0f, 0.0f, 0.0f);
                    if (before >= 0 && after < after_limit) {
                        const int before_x = horizontal ? x : before;
                        const int before_y = horizontal ? before : y;
                        const int after_x = horizontal ? x : after;
                        const int after_y = horizontal ? after : y;
                        const float t = horizontal
                            ? static_cast<float>(y - top)
                                / static_cast<float>(std::max(1, height - 1))
                            : static_cast<float>(x - left)
                                / static_cast<float>(std::max(1, width - 1));
                        value = fused_rgb.at<cv::Vec3f>(before_y, before_x)
                            * (1.0f - t)
                            + fused_rgb.at<cv::Vec3f>(after_y, after_x) * t;
                    } else if (before >= 0) {
                        const int before_x = horizontal ? x : before;
                        const int before_y = horizontal ? before : y;
                        value = fused_rgb.at<cv::Vec3f>(before_y, before_x);
                    } else if (after < after_limit) {
                        const int after_x = horizontal ? x : after;
                        const int after_y = horizontal ? after : y;
                        value = fused_rgb.at<cv::Vec3f>(after_y, after_x);
                    }
                    // The canvas-boundary feather may already have supplied a
                    // valid, continuous colour for this pixel.  Do not
                    // overwrite it with a directional strip sample; only
                    // fill genuinely dark/unresolved gap pixels.
                    const cv::Vec3f existing = fused_rgb.at<cv::Vec3f>(y, x);
                    const bool existing_valid = std::max({
                        existing[0], existing[1], existing[2]}) >= 0.08f;
                    repaired.at<cv::Vec3f>(y, x) = existing_valid ? existing : value;
                }
            }
        }
        // Any component that touches the RGB silhouette may have no vertical
        // source at all.  Finish only those unresolved pixels with a small
        // radius inpaint; the directional pass above remains authoritative.
        cv::Mat unresolved = cv::Mat::zeros(group_gap_fill.size(), CV_8UC1);
        for (int y = 0; y < fused_rgb.rows; ++y) {
            for (int x = 0; x < fused_rgb.cols; ++x) {
                if (group_gap_fill.at<std::uint8_t>(y, x) != 0U
                    && cv::norm(repaired.at<cv::Vec3f>(y, x)) < 1e-5) {
                    unresolved.at<std::uint8_t>(y, x) = 255U;
                }
            }
        }
        if (cv::countNonZero(unresolved) > 0) {
            cv::Mat repaired_u8;
            repaired.convertTo(repaired_u8, CV_8UC3, 255.0);
            cv::Mat inpainted_u8;
            cv::inpaint(repaired_u8, unresolved, inpainted_u8, 3.0, cv::INPAINT_NS);
            inpainted_u8.convertTo(repaired, CV_32FC3, 1.0 / 255.0);
        }
        fused_rgb = repaired;

        // The directional samples above preserve texture, but a dark model
        // border can still contribute a one-pixel exposure step at a support
        // margin.  Re-estimate a normalized low-frequency RGB field from
        // non-gap, non-background pixels.  Use it only to replace implausibly
        // dark outliers; keeping the normal current texture avoids turning
        // the whole strip into a flat, bright horizontal colour band.
        cv::Mat source_f;
        source.convertTo(source_f, CV_32FC1, 1.0 / 255.0);
        cv::Mat source_denominator;
        cv::GaussianBlur(source_f, source_denominator, cv::Size(), 6.0, 6.0);
        std::vector<cv::Mat> source_fields;
        cv::split(fused_rgb, source_fields);
        for (cv::Mat& channel : source_fields) {
            cv::Mat weighted;
            cv::multiply(channel, source_f, weighted);
            cv::GaussianBlur(weighted, weighted, cv::Size(), 6.0, 6.0);
            cv::Mat safe_denominator;
            cv::add(source_denominator, cv::Scalar(1e-5), safe_denominator);
            cv::divide(weighted, safe_denominator, channel);
        }
        cv::Mat source_field;
        cv::merge(source_fields, source_field);
        for (int y = 0; y < fused_rgb.rows; ++y) {
            for (int x = 0; x < fused_rgb.cols; ++x) {
                if (group_gap_fill.at<std::uint8_t>(y, x) == 0U
                    || source_denominator.at<float>(y, x) <= 0.03f) {
                    continue;
                }
                const cv::Vec3f field = source_field.at<cv::Vec3f>(y, x);
                const cv::Vec3f current = fused_rgb.at<cv::Vec3f>(y, x);
                const float field_mean = (field[0] + field[1] + field[2]) / 3.0f;
                const float current_mean = (current[0] + current[1] + current[2]) / 3.0f;
                if (current_mean < field_mean * 0.65f) {
                    fused_rgb.at<cv::Vec3f>(y, x) = field * 0.8f + current * 0.2f;
                }
            }
        }
    }

    const cv::Mat continuity_mask = [&]() {
        cv::Mat mask;
        cv::bitwise_and(update_mask, support_change, mask);
        if (prepared_group != nullptr && !group_gap_fill.empty()) {
            cv::bitwise_or(mask, group_gap_fill, mask);
        }
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

    if (prepared_group != nullptr && state.initialized
        && cv::countNonZero(group_gap_fill) > 0) {
        cv::Mat canvas_depth(state.height, state.width, CV_32FC1);
        for (int y = 0; y < state.height; ++y) {
            for (int x = 0; x < state.width; ++x) {
                canvas_depth.at<float>(y, x) = state.depth[
                    static_cast<std::size_t>(y) * state.width + x];
            }
        }
        interpolate_group_gap_depth_to_canvas(
            aligned_depth, group_gap_fill, canvas_depth, canvas_valid);
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
            if (group_stale_clear.at<std::uint8_t>(y, x) != 0U) {
                SlotValue after;
                after.depth = 0.0f;
                after.confidence = 0.0f;
                after.rgba = 0U;
                after.last_update_frame = static_cast<std::uint32_t>(raw.frame_seq);
                after.valid = 0U;
                const std::uint32_t slot_id = static_cast<std::uint32_t>(y * state.width + x);
                patch.updates.push_back(SlotUpdate{slot_id, after});
                patch.cleared_support_slots.push_back(slot_id);
                continue;
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
        if (update.slot_id >= state.valid.size()) {
            continue;
        }
        const bool was_valid = state.valid[update.slot_id] != 0U;
        const bool is_valid = update.after.valid != 0U;
        if (!was_valid && is_valid) {
            ++valid_point_count;
        } else if (was_valid && !is_valid && valid_point_count > 0U) {
            --valid_point_count;
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
    return process(raw, state);
}

FramePreprocessor::FramePreprocessor(InferenceOptions options)
    : options_(std::move(options)) {}

FramePreprocessor::FrameImage FramePreprocessor::load_frame(
    const std::filesystem::path& path) const {
    return load_frame(path, read_rgb(path));
}

FramePreprocessor::FrameImage FramePreprocessor::load_frame(
    const std::filesystem::path& path,
    const cv::Mat& original) const {
    FrameImage frame;
    frame.path = path;
    const double scale_x = static_cast<double>(options_.width) / static_cast<double>(original.cols);
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
    frame.match_rgb_u8 = quantize_rgb_u8(frame.match_rgb_f);

    frame.rgb_f = cv::Mat::zeros(options_.canvas_height, options_.canvas_width, CV_32FC3);
    frame.rgb_u8 = cv::Mat::zeros(options_.canvas_height, options_.canvas_width, CV_8UC3);
    const int pad_left = std::max(32, static_cast<int>(std::round(options_.width * 0.05)));
    const int pad_top = std::max(128, static_cast<int>(std::round(options_.width * 0.18)));
    int copy_x = pad_left;
    int copy_y = pad_top;
    if (copy_x >= 0 && copy_y >= 0
        && copy_x + frame.match_rgb_u8.cols <= frame.rgb_u8.cols
        && copy_y + frame.match_rgb_u8.rows <= frame.rgb_u8.rows) {
        frame.match_rgb_u8.copyTo(frame.rgb_u8(cv::Rect(
            copy_x, copy_y, frame.match_rgb_u8.cols, frame.match_rgb_u8.rows)));
        frame.match_rgb_f.copyTo(frame.rgb_f(cv::Rect(
            copy_x, copy_y, frame.match_rgb_f.cols, frame.match_rgb_f.rows)));
    } else {
        const int copy_width = std::min(frame.match_rgb_u8.cols, frame.rgb_u8.cols);
        const int copy_height = std::min(frame.match_rgb_u8.rows, frame.rgb_u8.rows);
        const int source_x = std::max(0, (frame.match_rgb_u8.cols - copy_width) / 2);
        const int source_y = std::max(0, (frame.match_rgb_u8.rows - copy_height) / 2);
        const int target_x = std::max(0, (frame.rgb_u8.cols - copy_width) / 2);
        const int target_y = std::max(0, (frame.rgb_u8.rows - copy_height) / 2);
        copy_x = target_x;
        copy_y = target_y;
        frame.match_rgb_u8(
            cv::Rect(source_x, source_y, copy_width, copy_height)).copyTo(
                frame.rgb_u8(cv::Rect(target_x, target_y, copy_width, copy_height)));
        frame.match_rgb_f(
            cv::Rect(source_x, source_y, copy_width, copy_height)).copyTo(
                frame.rgb_f(cv::Rect(target_x, target_y, copy_width, copy_height)));
    }
    // See the engine loader above: support all pixels from the actual camera
    // image, including dark but geometrically meaningful surfaces, while
    // excluding only the synthetic canvas padding.
    frame.support = cv::Mat::zeros(frame.rgb_f.size(), CV_8UC1);
    const int support_x = copy_x;
    const int support_y = copy_y;
    const int support_width = std::min(
        frame.match_rgb_u8.cols, frame.support.cols - support_x);
    const int support_height = std::min(
        frame.match_rgb_u8.rows, frame.support.rows - support_y);
    if (support_width > 0 && support_height > 0) {
        frame.support(cv::Rect(support_x, support_y, support_width, support_height)).setTo(255U);
    }
    return frame;
}

cv::Mat FramePreprocessor::warp_like(
    const cv::Mat& source,
    const cv::Mat& homography,
    const cv::Size size,
    const int interpolation) {
    cv::Mat warped;
    cv::warpPerspective(
        source,
        warped,
        homography,
        size,
        interpolation,
        cv::BORDER_CONSTANT,
        cv::Scalar(0, 0, 0));
    return warped;
}

cv::Mat FramePreprocessor::gray_u8(const cv::Mat& rgb_u8) {
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

cv::Mat FramePreprocessor::estimate_pair_homography(
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
        detector = cv::SIFT::create(3000);
        norm = cv::NORM_L2;
    } catch (const cv::Exception&) {
        detector = cv::ORB::create(3000);
    }
    std::vector<cv::KeyPoint> source_keypoints;
    std::vector<cv::KeyPoint> target_keypoints;
    cv::Mat source_descriptors;
    cv::Mat target_descriptors;
    detector->detectAndCompute(
        source_gray, cv::noArray(), source_keypoints, source_descriptors);
    detector->detectAndCompute(
        target_gray, cv::noArray(), target_keypoints, target_descriptors);
    if (source_descriptors.empty() || target_descriptors.empty()
        || source_keypoints.size() < 8U || target_keypoints.size() < 8U) {
        return cv::Mat::eye(3, 3, CV_32FC1);
    }
    cv::BFMatcher matcher(norm);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(source_descriptors, target_descriptors, knn, 2);
    std::vector<cv::DMatch> good;
    for (const auto& pair : knn) {
        if (pair.size() == 2U && pair[0].distance < 0.82f * pair[1].distance) {
            good.push_back(pair[0]);
        }
    }
    if (good.size() < 5U) {
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
        source_points, target_points, cv::RANSAC, 4.0, inlier_mask);
    if (homography.empty() || inlier_mask.empty()
        || cv::countNonZero(inlier_mask) < 5
        || !plausible_planar_homography(
            homography, source.match_rgb_u8.size(), target.match_rgb_u8.size())) {
        return cv::Mat::eye(3, 3, CV_32FC1);
    }
    homography.convertTo(homography, CV_32FC1);
    return homography;
}

PreparedInput FramePreprocessor::prepare(const RawFrame& raw) const {
    PreparedInput prepared_input;
    prepared_input.raw = raw;
    prepared_input.frame_seq = raw.frame_seq;
    Timer read_timer;
    if (options_.group_mode) {
        if (raw.group_paths.size() != 3U) {
            throw std::runtime_error("B=1,S=3 preprocessing requires exactly three images");
        }
        // group_paths/group_images are in GUI arrival order. LiveFrameSource
        // supplies the fixed arrival-position permutation {1,0,2}; source_seq
        // itself is never used to infer a camera identity.
        std::array<std::size_t, 3> arrival_for_model{};
        std::array<bool, 3> seen_model{};
        for (std::size_t arrival = 0; arrival < 3U; ++arrival) {
            const std::size_t model_slot = raw.group_model_slot_for_arrival[arrival];
            if (model_slot >= 3U || seen_model[model_slot]) {
                throw std::runtime_error("invalid GUI burst model-slot permutation");
            }
            seen_model[model_slot] = true;
            arrival_for_model[model_slot] = arrival;
        }

        std::array<FrameImage, 3> frames_by_model;
        for (std::size_t model_slot = 0; model_slot < 3U; ++model_slot) {
            const std::size_t arrival = arrival_for_model[model_slot];
            if (arrival < raw.group_images.size() && raw.group_images[arrival]) {
                frames_by_model[model_slot] = load_frame(
                    raw.group_paths[arrival], *raw.group_images[arrival]);
            } else {
                frames_by_model[model_slot] = load_frame(raw.group_paths[arrival]);
            }
        }
        const FrameImage& anchor = frames_by_model[static_cast<std::size_t>(
            std::clamp(raw.group_anchor_index, 0, 2))];
        prepared_input.path = anchor.path;
        prepared_input.rgb_u8 = anchor.rgb_u8;
        prepared_input.rgb_f = anchor.rgb_f;
        prepared_input.match_rgb_u8 = anchor.match_rgb_u8;
        prepared_input.match_rgb_f = anchor.match_rgb_f;
        prepared_input.support = anchor.support;
        prepared_input.group_model_rgb_f.reserve(3U);
        for (const FrameImage& frame : frames_by_model) {
            // Preserve aspect ratio, then round height to the model patch
            // multiple before center crop/pad to exactly 406x252.
            const cv::Mat& source = frame.match_rgb_f;
            const double model_scale = static_cast<double>(options_.group_width)
                / static_cast<double>(source.cols);
            const int resized_height = round_to_multiple_14(
                static_cast<double>(source.rows) * model_scale);
            cv::Mat resized_model;
            cv::resize(
                source,
                resized_model,
                cv::Size(options_.group_width, resized_height),
                0.0,
                0.0,
                cv::INTER_AREA);
            cv::Mat model_rgb;
            if (resized_height > options_.group_height) {
                const int crop_y = (resized_height - options_.group_height) / 2;
                model_rgb = resized_model(cv::Rect(
                    0, crop_y, options_.group_width, options_.group_height)).clone();
            } else if (resized_height < options_.group_height) {
                model_rgb = cv::Mat::zeros(
                    options_.group_height, options_.group_width, CV_32FC3);
                const int pad_top = (options_.group_height - resized_height) / 2;
                resized_model.copyTo(model_rgb(cv::Rect(
                    0, pad_top, options_.group_width, resized_height)));
            } else {
                model_rgb = std::move(resized_model);
            }
            prepared_input.group_model_rgb_f.push_back(std::move(model_rgb));
        }
        prepared_input.has_group = true;
        prepared_input.read_ms = read_timer.ms();
        prepared_input.image_names.reserve(raw.group_paths.size());
        for (const auto& path : raw.group_paths) {
            prepared_input.image_names.push_back(path.filename().string());
        }
        return prepared_input;
    }

    if (raw.group_paths.size() == 3U) {
        std::vector<FrameImage> frames;
        frames.reserve(raw.group_paths.size());
        for (std::size_t index = 0; index < raw.group_paths.size(); ++index) {
            // LiveFrameSource already provides decoded images.  Directory
            // replay provides only the three paths, so load those paths here
            // instead of silently falling back to the anchor image alone.
            if (index < raw.group_images.size() && raw.group_images[index]) {
                frames.push_back(load_frame(raw.group_paths[index], *raw.group_images[index]));
            } else {
                frames.push_back(load_frame(raw.group_paths[index]));
            }
        }
        const int anchor_index = std::clamp(raw.group_anchor_index, 0, 2);
        const FrameImage& anchor = frames[static_cast<std::size_t>(anchor_index)];
        prepared_input.observation_views.reserve(frames.size());
        for (const FrameImage& frame : frames) {
            PreparedView view;
            view.path = frame.path;
            view.rgb_u8 = frame.rgb_u8;
            view.rgb_f = frame.rgb_f;
            view.match_rgb_u8 = frame.match_rgb_u8;
            view.match_rgb_f = frame.match_rgb_f;
            view.support = frame.support;
            prepared_input.observation_views.push_back(std::move(view));
        }
        prepared_input.path = anchor.path;
        prepared_input.rgb_u8 = anchor.rgb_u8;
        prepared_input.rgb_f = anchor.rgb_f;
        prepared_input.match_rgb_u8 = anchor.match_rgb_u8;
        prepared_input.match_rgb_f = anchor.match_rgb_f;
        prepared_input.support = anchor.support;
        prepared_input.group_model_rgb_f.reserve(3U);
        prepared_input.group_warped_rgb_f.reserve(3U);
        prepared_input.group_valid_warp.reserve(3U);
        const auto canvas_translation = [&](const FrameImage& frame) {
            const cv::Point origin = content_origin(
                frame.support,
                std::max(32, static_cast<int>(std::round(options_.width * 0.05))),
                std::max(128, static_cast<int>(std::round(options_.width * 0.18))));
            cv::Mat result = cv::Mat::eye(3, 3, CV_32FC1);
            result.at<float>(0, 2) = static_cast<float>(origin.x);
            result.at<float>(1, 2) = static_cast<float>(origin.y);
            return result;
        };
        const cv::Mat anchor_translation = canvas_translation(anchor);
        std::array<cv::Mat, 3> frame_to_anchor;
        std::array<bool, 3> frame_to_anchor_valid{};
        for (int index = 0; index < 3; ++index) {
            if (index == anchor_index) {
                frame_to_anchor[static_cast<std::size_t>(index)] =
                    cv::Mat::eye(3, 3, CV_32FC1);
                frame_to_anchor_valid[static_cast<std::size_t>(index)] = true;
                continue;
            }
            cv::Mat direct = estimate_pair_homography(
                frames[static_cast<std::size_t>(index)], anchor);
            const bool valid = !direct.empty()
                && cv::norm(
                    direct - cv::Mat::eye(3, 3, CV_32FC1),
                    cv::NORM_INF) > 1e-3
                && plausible_planar_homography(
                    direct,
                    frames[static_cast<std::size_t>(index)].match_rgb_u8.size(),
                    anchor.match_rgb_u8.size());
            frame_to_anchor[static_cast<std::size_t>(index)] = std::move(direct);
            frame_to_anchor_valid[static_cast<std::size_t>(index)] = valid;
        }
        // A camera opposite the anchor can have too little direct floor
        // overlap for a non-degenerate homography. Route it through the third
        // fixed camera when that camera has a validated direct anchor map.
        for (int index = 0; index < 3; ++index) {
            if (index == anchor_index
                || frame_to_anchor_valid[static_cast<std::size_t>(index)]) {
                continue;
            }
            const int bridge_index = 3 - anchor_index - index;
            if (bridge_index < 0 || bridge_index >= 3
                || !frame_to_anchor_valid[static_cast<std::size_t>(bridge_index)]) {
                continue;
            }
            const cv::Mat source_to_bridge = estimate_pair_homography(
                frames[static_cast<std::size_t>(index)],
                frames[static_cast<std::size_t>(bridge_index)]);
            if (source_to_bridge.empty()
                || cv::norm(
                    source_to_bridge - cv::Mat::eye(3, 3, CV_32FC1),
                    cv::NORM_INF) <= 1e-3
                || !plausible_planar_homography(
                    source_to_bridge,
                    frames[static_cast<std::size_t>(index)].match_rgb_u8.size(),
                    frames[static_cast<std::size_t>(bridge_index)].match_rgb_u8.size())) {
                continue;
            }
            cv::Mat chained =
                frame_to_anchor[static_cast<std::size_t>(bridge_index)]
                * source_to_bridge;
            chained.convertTo(chained, CV_32FC1);
            if (!plausible_planar_homography(
                    chained,
                    frames[static_cast<std::size_t>(index)].match_rgb_u8.size(),
                    anchor.match_rgb_u8.size())) {
                continue;
            }
            frame_to_anchor[static_cast<std::size_t>(index)] = std::move(chained);
            frame_to_anchor_valid[static_cast<std::size_t>(index)] = true;
        }
        for (int index = 0; index < 3; ++index) {
            // Match omnivggt_edge preprocessing exactly. The source images
            // are 5:4, while the group graph is 406x252; directly resizing
            // 700x560 to that shape squashes Y by about 22 percent and causes
            // incorrect camera poses, duplicated floor edges and large gaps.
            const cv::Mat& source = frames[static_cast<std::size_t>(index)].match_rgb_f;
            const double model_scale = static_cast<double>(options_.group_width)
                / static_cast<double>(source.cols);
            const int resized_height = round_to_multiple_14(
                static_cast<double>(source.rows) * model_scale);
            cv::Mat resized_model;
            cv::resize(
                source,
                resized_model,
                cv::Size(options_.group_width, resized_height),
                0.0,
                0.0,
                cv::INTER_AREA);
            cv::Mat model_rgb;
            if (resized_height > options_.group_height) {
                const int crop_y = (resized_height - options_.group_height) / 2;
                model_rgb = resized_model(cv::Rect(
                    0, crop_y, options_.group_width, options_.group_height)).clone();
            } else if (resized_height < options_.group_height) {
                model_rgb = cv::Mat::zeros(
                    options_.group_height, options_.group_width, CV_32FC3);
                const int pad_top = (options_.group_height - resized_height) / 2;
                resized_model.copyTo(model_rgb(cv::Rect(
                    0, pad_top, options_.group_width, resized_height)));
            } else {
                model_rgb = std::move(resized_model);
            }
            prepared_input.group_model_rgb_f.push_back(std::move(model_rgb));
            cv::Mat aligned_rgb = frames[static_cast<std::size_t>(index)].rgb_f.clone();
            cv::Mat aligned_support = frames[static_cast<std::size_t>(index)].support.clone();
            cv::Mat canvas_homography = cv::Mat::eye(3, 3, CV_32FC1);
            bool pair_alignment_valid = index == anchor_index;
            if (index != anchor_index) {
                const cv::Mat& match_homography =
                    frame_to_anchor[static_cast<std::size_t>(index)];
                pair_alignment_valid =
                    frame_to_anchor_valid[static_cast<std::size_t>(index)];
                const cv::Mat source_translation = canvas_translation(
                    frames[static_cast<std::size_t>(index)]);
                const cv::Mat inverse_source_translation = source_translation.inv();
                canvas_homography = anchor_translation
                    * match_homography * inverse_source_translation;
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
            prepared_input.observation_views[static_cast<std::size_t>(index)].canvas_homography =
                canvas_homography;
            prepared_input.observation_views[static_cast<std::size_t>(index)].pair_alignment_valid =
                pair_alignment_valid;
            prepared_input.group_warped_rgb_f.push_back(std::move(aligned_rgb));
            prepared_input.group_valid_warp.push_back(std::move(aligned_support));
        }
        prepared_input.group_union_valid = cv::Mat::zeros(
            options_.canvas_height, options_.canvas_width, CV_8UC1);
        for (const cv::Mat& support : prepared_input.group_valid_warp) {
            cv::bitwise_or(prepared_input.group_union_valid, support, prepared_input.group_union_valid);
        }
        if (options_.group_mode) {
            prepared_input.has_group = true;
            prepared_input.group_fused_rgb_f =
                prepared_input.group_warped_rgb_f[static_cast<std::size_t>(anchor_index)].clone();
        } else {
            prepared_input.has_observation_group = true;
            // Observation-mode three-image windows must not blend side-view
            // RGB in image space.  A single 2D homography cannot align
            // silhouettes and non-planar edge folds from neighbouring views;
            // feeding that blended RGB to the standard S=2 stream model makes
            // the model reconstruct the parallax smear as a real bent surface.
            // Keep the current observation identical to the selected anchor
            // frame so the downstream change mask, depth alignment and seam
            // repair remain bit-for-bit comparable to the single-image stream.
            // Side frames are retained in group_warped_rgb_f/group_valid_warp
            // for diagnostics and for a future canvas/model-space fusion path.
            prepared_input.support = anchor.support.clone();
            prepared_input.rgb_f = anchor.rgb_f.clone();
            prepared_input.rgb_f.convertTo(prepared_input.rgb_u8, CV_8UC3, 255.0);
        }
    } else {
        const FrameImage frame = load_frame(raw.path);
        prepared_input.path = frame.path;
        prepared_input.rgb_u8 = frame.rgb_u8;
        prepared_input.rgb_f = frame.rgb_f;
        prepared_input.match_rgb_u8 = frame.match_rgb_u8;
        prepared_input.match_rgb_f = frame.match_rgb_f;
        prepared_input.support = frame.support;
    }
    prepared_input.read_ms = read_timer.ms();
    prepared_input.image_names.push_back(prepared_input.path.filename().string());
    return prepared_input;
}

}  // namespace omnivggt::observer
