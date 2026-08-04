#include "pointcloud_export.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <vector>

namespace omnivggt::observer {

namespace {

float median_value(std::vector<float> values) {
    values.erase(std::remove_if(values.begin(), values.end(), [](const float value) {
        return !std::isfinite(value);
    }), values.end());
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

std::vector<std::int32_t> nearest_indices(const cv::Mat& source_mask) {
    const int width = source_mask.cols;
    const int height = source_mask.rows;
    std::vector<std::int32_t> nearest(
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height), -1);
    std::deque<int> queue;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (source_mask.at<std::uint8_t>(y, x) != 0U) {
                const int index = y * width + x;
                nearest[static_cast<std::size_t>(index)] = index;
                queue.push_back(index);
            }
        }
    }
    constexpr std::array<int, 8> dx = {-1, 0, 1, -1, 1, -1, 0, 1};
    constexpr std::array<int, 8> dy = {-1, -1, -1, 0, 0, 1, 1, 1};
    while (!queue.empty()) {
        const int index = queue.front();
        queue.pop_front();
        const int x = index % width;
        const int y = index / width;
        const int source = nearest[static_cast<std::size_t>(index)];
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
            nearest[static_cast<std::size_t>(next)] = source;
            queue.push_back(next);
        }
    }
    return nearest;
}

void fit_quadratic(
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

double quadratic_value(const std::array<double, 6>& coefficients, const float x, const float y) {
    return coefficients[0] * x + coefficients[1] * y
        + coefficients[2] * static_cast<double>(x) * x
        + coefficients[3] * static_cast<double>(y) * y
        + coefficients[4] * static_cast<double>(x) * y
        + coefficients[5];
}

}  // namespace

std::vector<ExportPoint> export_clean_canvas_points(
    const CanvasState& state,
    const FrameSeq changed_frame) {
    if (!state.shape_valid() || state.depth.size() != state.slot_count()
        || state.rgba.size() != state.slot_count() || state.valid.size() != state.slot_count()) {
        return {};
    }

    const int height = state.height;
    const int width = state.width;
    cv::Mat depth(height, width, CV_32FC1, cv::Scalar(0));
    cv::Mat rgb(height, width, CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat valid(height, width, CV_8UC1, cv::Scalar(0));
    cv::Mat changed(height, width, CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const std::size_t index = static_cast<std::size_t>(y) * width + x;
            depth.at<float>(y, x) = state.depth[index];
            if (state.valid[index] == 0U || !std::isfinite(state.depth[index])) {
                continue;
            }
            valid.at<std::uint8_t>(y, x) = 255U;
            const auto color = unpack_rgba(state.rgba[index]);
            rgb.at<cv::Vec3f>(y, x) = cv::Vec3f(
                static_cast<float>(color[0]) / 255.0f,
                static_cast<float>(color[1]) / 255.0f,
                static_cast<float>(color[2]) / 255.0f);
            if (changed_frame != 0U && state.last_update_frame[index]
                == static_cast<std::uint32_t>(changed_frame)) {
                changed.at<std::uint8_t>(y, x) = 255U;
            }
        }
    }
    if (cv::countNonZero(valid) == 0) {
        return {};
    }

    // Python _fill_narrow_gaps: close only narrow internal holes and copy the
    // nearest valid sample.  This is deliberately not written back to state.
    const cv::Mat close_kernel = cv::Mat::ones(11, 11, CV_8UC1);
    cv::Mat closed;
    cv::morphologyEx(valid, closed, cv::MORPH_CLOSE, close_kernel);
    cv::Mat fill;
    cv::Mat inverse_valid;
    cv::bitwise_not(valid, inverse_valid);
    cv::bitwise_and(closed, inverse_valid, fill);
    if (cv::countNonZero(fill) > 0) {
        const std::vector<std::int32_t> nearest = nearest_indices(valid);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (fill.at<std::uint8_t>(y, x) == 0U) {
                    continue;
                }
                const int source = nearest[static_cast<std::size_t>(y * width + x)];
                if (source < 0) {
                    continue;
                }
                const int sy = source / width;
                const int sx = source % width;
                depth.at<float>(y, x) = depth.at<float>(sy, sx);
                rgb.at<cv::Vec3f>(y, x) = rgb.at<cv::Vec3f>(sy, sx);
                changed.at<std::uint8_t>(y, x) = changed.at<std::uint8_t>(sy, sx);
                valid.at<std::uint8_t>(y, x) = 255U;
            }
        }
    }

    cv::Mat depth_work = depth.clone();
    cv::Mat valid_f;
    valid.convertTo(valid_f, CV_32FC1, 1.0 / 255.0);
    if (cv::countNonZero(valid) >= 64) {
        // Python's local median uses nearest-valid padding for invalid cells.
        cv::Mat median_input = depth_work.clone();
        const std::vector<std::int32_t> nearest = nearest_indices(valid);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (valid.at<std::uint8_t>(y, x) != 0U) {
                    continue;
                }
                const int source = nearest[static_cast<std::size_t>(y * width + x)];
                if (source >= 0) {
                    median_input.at<float>(y, x) = depth_work.at<float>(
                        source / width, source % width);
                }
            }
        }
        cv::Mat local_median;
        cv::medianBlur(median_input, local_median, 5);
        std::vector<float> residual_values;
        residual_values.reserve(static_cast<std::size_t>(cv::countNonZero(valid)));
        cv::Mat residual = depth_work - local_median;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (valid.at<std::uint8_t>(y, x) != 0U
                    && std::isfinite(residual.at<float>(y, x))) {
                    residual_values.push_back(residual.at<float>(y, x));
                }
            }
        }
        if (residual_values.size() >= 64U) {
            const float center = median_value(residual_values);
            std::vector<float> deviations;
            deviations.reserve(residual_values.size());
            for (const float value : residual_values) {
                deviations.push_back(std::abs(value - center));
            }
            const float limit = std::max(
                0.025f, 6.0f * 1.4826f * std::max(median_value(deviations), 1e-6f));
            cv::Mat neighborhood;
            cv::blur(valid_f, neighborhood, cv::Size(5, 5), cv::Point(-1, -1), cv::BORDER_REPLICATE);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    if (valid.at<std::uint8_t>(y, x) != 0U
                        && neighborhood.at<float>(y, x) > 0.72f
                        && std::abs(residual.at<float>(y, x) - center) > limit) {
                        depth_work.at<float>(y, x) = local_median.at<float>(y, x);
                    }
                }
            }
        }
    }

    cv::Mat weighted;
    cv::Mat weight;
    cv::GaussianBlur(depth_work.mul(valid_f), weighted, cv::Size(), 4.0, 4.0);
    cv::GaussianBlur(valid_f, weight, cv::Size(), 4.0, 4.0);
    cv::Mat smooth;
    cv::divide(weighted, weight + 1e-6f, smooth);
    cv::Mat regularized = smooth * 0.85f + depth_work * 0.15f;
    regularized.copyTo(depth_work, valid);

    cv::Mat finite_depth(height, width, CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (std::isfinite(depth_work.at<float>(y, x))) {
                finite_depth.at<std::uint8_t>(y, x) = 255U;
            }
        }
    }
    cv::bitwise_and(valid, finite_depth, valid);
    const int trim_px = std::max(
        3, std::min(4, static_cast<int>(std::lround(std::min(height, width) * 0.006))));
    cv::Mat trimmed = valid.clone();
    if (cv::countNonZero(valid) > 0) {
        const cv::Mat trim_kernel = cv::Mat::ones(trim_px * 2 + 1, trim_px * 2 + 1, CV_8UC1);
        cv::erode(valid, trimmed, trim_kernel);
        if (cv::countNonZero(trimmed) == 0) {
            trimmed = valid;
        }
    }

    std::vector<float> x_values;
    std::vector<float> y_values;
    std::vector<float> z_values;
    std::vector<cv::Point> pixels;
    x_values.reserve(static_cast<std::size_t>(cv::countNonZero(trimmed)));
    y_values.reserve(x_values.capacity());
    z_values.reserve(x_values.capacity());
    pixels.reserve(x_values.capacity());
    const float scale = static_cast<float>(std::max(height, width));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (trimmed.at<std::uint8_t>(y, x) == 0U) {
                continue;
            }
            x_values.push_back((static_cast<float>(x) - width * 0.5f) / scale);
            y_values.push_back(-(static_cast<float>(y) - height * 0.5f) / scale);
            z_values.push_back(depth_work.at<float>(y, x));
            pixels.emplace_back(x, y);
        }
    }
    if (z_values.size() < 3U) {
        return {};
    }

    const float base_depth = std::max(std::abs(median_value(z_values)), 1e-6f);
    std::vector<std::uint8_t> keep(z_values.size(), 1U);
    std::array<double, 6> coefficients{};
    fit_quadratic(x_values, y_values, z_values, keep, coefficients);
    for (int iteration = 0; iteration < 3; ++iteration) {
        // Match Python _plane_residual: each iteration fits using the current
        // inlier set, then updates that set. The final mask update is not
        // followed by an extra fit; that extra fit changes the low-order
        // surface at the displayed boundary.
        fit_quadratic(x_values, y_values, z_values, keep, coefficients);
        std::vector<float> residuals;
        residuals.reserve(z_values.size());
        for (std::size_t index = 0; index < z_values.size(); ++index) {
            residuals.push_back(static_cast<float>(z_values[index]
                - quadratic_value(coefficients, x_values[index], y_values[index])));
        }
        std::vector<float> kept_residuals;
        for (std::size_t index = 0; index < residuals.size(); ++index) {
            if (keep[index] != 0U) {
                kept_residuals.push_back(residuals[index]);
            }
        }
        const float center = median_value(kept_residuals);
        std::vector<float> deviations;
        deviations.reserve(kept_residuals.size());
        for (const float value : kept_residuals) {
            deviations.push_back(std::abs(value - center));
        }
        const float limit = 4.0f * 1.4826f * std::max(median_value(deviations), 1e-6f);
        std::size_t kept = 0;
        for (std::size_t index = 0; index < residuals.size(); ++index) {
            keep[index] = std::abs(residuals[index] - center) <= limit ? 1U : 0U;
            kept += keep[index] != 0U ? 1U : 0U;
        }
        if (kept < 128U) {
            std::fill(keep.begin(), keep.end(), static_cast<std::uint8_t>(1U));
            break;
        }
    }
    std::vector<float> residuals;
    residuals.reserve(z_values.size());
    for (std::size_t index = 0; index < z_values.size(); ++index) {
        residuals.push_back(static_cast<float>(z_values[index]
            - quadratic_value(coefficients, x_values[index], y_values[index])));
    }
    const float center = median_value(residuals);
    std::vector<float> deviations;
    deviations.reserve(residuals.size());
    for (const float value : residuals) {
        deviations.push_back(std::abs(value - center));
    }
    const float limit = std::min(
        3.0f * 1.4826f * std::max(median_value(deviations), 1e-6f),
        0.03f * base_depth);
    std::vector<float> clipped;
    clipped.reserve(residuals.size());
    for (const float value : residuals) {
        clipped.push_back(std::clamp(value, center - limit, center + limit));
    }
    const float clipped_center = median_value(clipped);

    std::vector<ExportPoint> points;
    points.reserve(clipped.size());
    for (std::size_t index = 0; index < clipped.size(); ++index) {
        const cv::Point pixel = pixels[index];
        const cv::Vec3f color = rgb.at<cv::Vec3f>(pixel.y, pixel.x);
        points.push_back(ExportPoint{
            x_values[index],
            y_values[index],
            (clipped[index] - clipped_center) / base_depth,
            static_cast<std::uint8_t>(std::clamp(
                static_cast<int>(std::floor(std::clamp(color[0] * 255.0f, 0.0f, 255.0f))), 0, 255)),
            static_cast<std::uint8_t>(std::clamp(
                static_cast<int>(std::floor(std::clamp(color[1] * 255.0f, 0.0f, 255.0f))), 0, 255)),
            static_cast<std::uint8_t>(std::clamp(
                static_cast<int>(std::floor(std::clamp(color[2] * 255.0f, 0.0f, 255.0f))), 0, 255)),
            changed.at<std::uint8_t>(pixel.y, pixel.x) != 0U});
    }
    return points;
}

}  // namespace omnivggt::observer
