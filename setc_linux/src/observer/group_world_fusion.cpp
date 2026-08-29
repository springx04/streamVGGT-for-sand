#include "group_world_fusion.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace omnivggt::observer {

namespace {

constexpr int kMaxPlaneSamplesPerView = 4096;
constexpr int kPlaneRansacTrials = 4096;
constexpr int kMinPlaneInliersPerView = 8;
constexpr float kBaselineEpsilon = 1e-5f;
constexpr float kNumericEpsilon = 1e-7f;
constexpr float kGainAlpha = 0.2f;
constexpr std::size_t kMinimumGainOverlap = 32U;

struct CandidatePoint {
    cv::Vec3f point;
    float confidence = 0.0f;
    int view = 0;
    int x = 0;
    int y = 0;
};

struct Plane {
    cv::Vec3f normal = cv::Vec3f(0.0f, 0.0f, 1.0f);
    float distance = 0.0f;
};

struct Sim3 {
    cv::Matx33f rotation = cv::Matx33f::eye();
    float scale = 1.0f;
    cv::Vec3f translation = cv::Vec3f(0.0f, 0.0f, 0.0f);
};

struct FloorSample {
    cv::Vec3f color = cv::Vec3f(0.0f, 0.0f, 0.0f);
    float normalized_confidence = 0.0f;
    float weight = 0.0f;
    bool valid = false;
};

struct ObjectSample {
    float height = 0.0f;
    float normalized_confidence = 0.0f;
    float weight = 0.0f;
    cv::Vec3f color = cv::Vec3f(0.0f, 0.0f, 0.0f);
    int view = 0;
};

struct HeightCluster {
    std::vector<ObjectSample> samples;
    float total_weight = 0.0f;
    float height = 0.0f;
};

struct FloorUVSample {
    float u = 0.0f;
    float v = 0.0f;
};

bool finite_vec(const cv::Vec3f& value) {
    return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]);
}

bool finite_quaternion(const cv::Vec4f& value) {
    return std::isfinite(value[0]) && std::isfinite(value[1])
        && std::isfinite(value[2]) && std::isfinite(value[3]);
}

float vector_norm(const cv::Vec3f& value) {
    return std::sqrt(value.dot(value));
}

bool normalize_vec(cv::Vec3f& value) {
    const float norm = vector_norm(value);
    if (!std::isfinite(norm) || norm <= kNumericEpsilon) {
        return false;
    }
    value *= 1.0f / norm;
    return finite_vec(value);
}

float median_value(std::vector<float> values) {
    values.erase(std::remove_if(values.begin(), values.end(), [](const float value) {
        return !std::isfinite(value);
    }), values.end());
    if (values.empty()) {
        return 0.0f;
    }
    const std::size_t middle = values.size() / 2U;
    auto middle_it = values.begin() + static_cast<std::ptrdiff_t>(middle);
    std::nth_element(values.begin(), middle_it, values.end());
    const float upper = *middle_it;
    if ((values.size() & 1U) != 0U) {
        return upper;
    }
    auto lower_it = values.begin() + static_cast<std::ptrdiff_t>(middle - 1U);
    std::nth_element(values.begin(), lower_it, middle_it);
    return 0.5f * (*lower_it + upper);
}

float percentile_value(std::vector<float> values, const float percentile) {
    values.erase(std::remove_if(values.begin(), values.end(), [](const float value) {
        return !std::isfinite(value);
    }), values.end());
    if (values.empty()) {
        return 0.0f;
    }
    const float clamped = std::clamp(percentile, 0.0f, 1.0f);
    const std::size_t index = static_cast<std::size_t>(std::lround(
        clamped * static_cast<float>(values.size() - 1U)));
    auto it = values.begin() + static_cast<std::ptrdiff_t>(index);
    std::nth_element(values.begin(), it, values.end());
    return *it;
}

cv::Vec3f componentwise_median(const std::vector<cv::Vec3f>& values) {
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;
    x.reserve(values.size());
    y.reserve(values.size());
    z.reserve(values.size());
    for (const cv::Vec3f& value : values) {
        x.push_back(value[0]);
        y.push_back(value[1]);
        z.push_back(value[2]);
    }
    return cv::Vec3f(
        median_value(std::move(x)),
        median_value(std::move(y)),
        median_value(std::move(z)));
}

bool rotation_from_quaternion(const cv::Vec4f& quaternion, cv::Matx33f& rotation) {
    if (!finite_quaternion(quaternion)) {
        return false;
    }
    const float x = quaternion[0];
    const float y = quaternion[1];
    const float z = quaternion[2];
    const float w = quaternion[3];
    const float denominator = x * x + y * y + z * z + w * w;
    if (!std::isfinite(denominator) || denominator <= kNumericEpsilon) {
        return false;
    }
    const float scale = 2.0f / denominator;
    rotation = cv::Matx33f(
        1.0f - scale * (y * y + z * z),
        scale * (x * y - z * w),
        scale * (x * z + y * w),
        scale * (x * y + z * w),
        1.0f - scale * (x * x + z * z),
        scale * (y * z - x * w),
        scale * (x * z - y * w),
        scale * (y * z + x * w),
        1.0f - scale * (x * x + y * y));
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            if (!std::isfinite(rotation(row, column))) {
                return false;
            }
        }
    }
    return cv::determinant(cv::Mat(rotation)) > 0.0;
}

bool camera_center(const GroupWorldView& view, cv::Vec3f& center) {
    cv::Matx33f rotation;
    if (!view.has_pose || !rotation_from_quaternion(view.quaternion, rotation)
        || !finite_vec(view.translation)) {
        return false;
    }
    const cv::Vec3f rotated = rotation.t() * view.translation;
    center = cv::Vec3f(-rotated[0], -rotated[1], -rotated[2]);
    return finite_vec(center);
}

cv::Vec3f apply_sim3(const Sim3& transform, const cv::Vec3f& point) {
    return transform.rotation * point * transform.scale + transform.translation;
}

bool calculate_sim3(
    const std::array<cv::Vec3f, 3>& current,
    const std::array<cv::Vec3f, 3>& reference,
    Sim3& result,
    float& aligned_rms) {
    cv::Vec3d current_mean(0.0, 0.0, 0.0);
    cv::Vec3d reference_mean(0.0, 0.0, 0.0);
    for (int index = 0; index < 3; ++index) {
        current_mean += cv::Vec3d(
            current[static_cast<std::size_t>(index)][0],
            current[static_cast<std::size_t>(index)][1],
            current[static_cast<std::size_t>(index)][2]);
        reference_mean += cv::Vec3d(
            reference[static_cast<std::size_t>(index)][0],
            reference[static_cast<std::size_t>(index)][1],
            reference[static_cast<std::size_t>(index)][2]);
    }
    current_mean *= 1.0 / 3.0;
    reference_mean *= 1.0 / 3.0;

    cv::Mat covariance = cv::Mat::zeros(3, 3, CV_64F);
    double denominator = 0.0;
    for (int index = 0; index < 3; ++index) {
        const cv::Vec3d current_delta(
            static_cast<double>(current[static_cast<std::size_t>(index)][0]) - current_mean[0],
            static_cast<double>(current[static_cast<std::size_t>(index)][1]) - current_mean[1],
            static_cast<double>(current[static_cast<std::size_t>(index)][2]) - current_mean[2]);
        const cv::Vec3d reference_delta(
            static_cast<double>(reference[static_cast<std::size_t>(index)][0]) - reference_mean[0],
            static_cast<double>(reference[static_cast<std::size_t>(index)][1]) - reference_mean[1],
            static_cast<double>(reference[static_cast<std::size_t>(index)][2]) - reference_mean[2]);
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                covariance.at<double>(row, column) +=
                    reference_delta[row] * current_delta[column] / 3.0;
            }
        }
        denominator += current_delta.dot(current_delta);
    }
    // The covariance above is the mean covariance (the required /3 factor),
    // so use the matching mean centered energy in the Umeyama scale ratio.
    denominator /= 3.0;
    if (!std::isfinite(denominator) || denominator <= kNumericEpsilon) {
        return false;
    }

    cv::SVD svd(covariance, cv::SVD::FULL_UV);
    if (svd.w.rows < 3 || svd.u.rows != 3 || svd.vt.rows != 3) {
        return false;
    }
    cv::Mat correction = cv::Mat::eye(3, 3, CV_64F);
    const cv::Mat uncorrected_rotation = svd.u * svd.vt;
    if (cv::determinant(uncorrected_rotation) < 0.0) {
        correction.at<double>(2, 2) = -1.0;
    }
    const cv::Mat rotation = svd.u * correction * svd.vt;
    const double rotation_determinant = cv::determinant(rotation);
    if (!std::isfinite(rotation_determinant) || rotation_determinant <= 0.0) {
        return false;
    }
    const double numerator =
        svd.w.at<double>(0, 0) * correction.at<double>(0, 0)
        + svd.w.at<double>(1, 0) * correction.at<double>(1, 1)
        + svd.w.at<double>(2, 0) * correction.at<double>(2, 2);
    const double scale = numerator / denominator;
    if (!std::isfinite(scale) || scale < 0.5 || scale > 2.0) {
        return false;
    }

    cv::Matx33f rotation_f;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            rotation_f(row, column) = static_cast<float>(rotation.at<double>(row, column));
        }
    }
    const cv::Vec3f translation(
        static_cast<float>(reference_mean[0]
            - scale * (rotation.at<double>(0, 0) * current_mean[0]
                + rotation.at<double>(0, 1) * current_mean[1]
                + rotation.at<double>(0, 2) * current_mean[2])),
        static_cast<float>(reference_mean[1]
            - scale * (rotation.at<double>(1, 0) * current_mean[0]
                + rotation.at<double>(1, 1) * current_mean[1]
                + rotation.at<double>(1, 2) * current_mean[2])),
        static_cast<float>(reference_mean[2]
            - scale * (rotation.at<double>(2, 0) * current_mean[0]
                + rotation.at<double>(2, 1) * current_mean[1]
                + rotation.at<double>(2, 2) * current_mean[2])));
    if (!finite_vec(translation)) {
        return false;
    }

    result.rotation = rotation_f;
    result.scale = static_cast<float>(scale);
    result.translation = translation;
    double squared_error = 0.0;
    for (int index = 0; index < 3; ++index) {
        const cv::Vec3f aligned = apply_sim3(
            result, current[static_cast<std::size_t>(index)]);
        const cv::Vec3f residual = aligned - reference[static_cast<std::size_t>(index)];
        squared_error += residual.dot(residual);
    }
    aligned_rms = static_cast<float>(std::sqrt(squared_error / 3.0));
    return std::isfinite(aligned_rms);
}

bool plane_from_three_points(
    const cv::Vec3f& first,
    const cv::Vec3f& second,
    const cv::Vec3f& third,
    Plane& plane) {
    cv::Vec3f normal = (second - first).cross(third - first);
    if (!normalize_vec(normal)) {
        return false;
    }
    plane.normal = normal;
    plane.distance = normal.dot(first);
    return std::isfinite(plane.distance);
}

bool refine_plane(
    const std::vector<CandidatePoint>& candidates,
    const std::vector<int>& indices,
    Plane& plane) {
    if (indices.size() < 3U) {
        return false;
    }
    cv::Vec3d mean(0.0, 0.0, 0.0);
    for (const int index : indices) {
        const cv::Vec3f& point = candidates[static_cast<std::size_t>(index)].point;
        mean += cv::Vec3d(point[0], point[1], point[2]);
    }
    mean *= 1.0 / static_cast<double>(indices.size());
    cv::Mat covariance = cv::Mat::zeros(3, 3, CV_64F);
    for (const int index : indices) {
        const cv::Vec3f& point = candidates[static_cast<std::size_t>(index)].point;
        const cv::Vec3d delta(
            static_cast<double>(point[0]) - mean[0],
            static_cast<double>(point[1]) - mean[1],
            static_cast<double>(point[2]) - mean[2]);
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                covariance.at<double>(row, column) +=
                    delta[row] * delta[column];
            }
        }
    }
    cv::SVD svd(covariance, cv::SVD::FULL_UV);
    if (svd.vt.rows != 3 || svd.vt.cols != 3) {
        return false;
    }
    cv::Vec3f normal(
        static_cast<float>(svd.vt.at<double>(2, 0)),
        static_cast<float>(svd.vt.at<double>(2, 1)),
        static_cast<float>(svd.vt.at<double>(2, 2)));
    if (!normalize_vec(normal)) {
        return false;
    }
    plane.normal = normal;
    plane.distance = static_cast<float>(
        normal[0] * mean[0] + normal[1] * mean[1] + normal[2] * mean[2]);
    return std::isfinite(plane.distance);
}

bool find_dominant_plane(
    const std::vector<CandidatePoint>& candidates,
    const float scene_scale,
    Plane& refined_plane,
    std::vector<int>& refined_inliers) {
    if (candidates.size() < 3U || !std::isfinite(scene_scale) || scene_scale <= kNumericEpsilon) {
        return false;
    }
    const float threshold = 0.015f * scene_scale;
    std::uint32_t random_state = 0x6d2b79f5U;
    int best_count = -1;
    int best_min_view_count = -1;
    std::vector<int> best_inliers;
    for (int trial = 0; trial < kPlaneRansacTrials; ++trial) {
        random_state = random_state * 1664525U + 1013904223U;
        const int first = static_cast<int>(random_state % candidates.size());
        random_state = random_state * 1664525U + 1013904223U;
        const int second = static_cast<int>(random_state % candidates.size());
        random_state = random_state * 1664525U + 1013904223U;
        const int third = static_cast<int>(random_state % candidates.size());
        if (first == second || first == third || second == third) {
            continue;
        }
        Plane plane;
        if (!plane_from_three_points(
                candidates[static_cast<std::size_t>(first)].point,
                candidates[static_cast<std::size_t>(second)].point,
                candidates[static_cast<std::size_t>(third)].point,
                plane)) {
            continue;
        }
        std::array<int, 3> view_counts{};
        std::vector<int> inliers;
        inliers.reserve(candidates.size());
        for (std::size_t index = 0; index < candidates.size(); ++index) {
            const CandidatePoint& candidate = candidates[index];
            const float residual = std::abs(plane.normal.dot(candidate.point) - plane.distance);
            if (residual <= threshold) {
                inliers.push_back(static_cast<int>(index));
                if (candidate.view >= 0 && candidate.view < 3) {
                    ++view_counts[static_cast<std::size_t>(candidate.view)];
                }
            }
        }
        const int minimum_view_count = std::min({view_counts[0], view_counts[1], view_counts[2]});
        if (minimum_view_count < kMinPlaneInliersPerView) {
            continue;
        }
        const int count = static_cast<int>(inliers.size());
        if (count > best_count
            || (count == best_count && minimum_view_count > best_min_view_count)) {
            best_count = count;
            best_min_view_count = minimum_view_count;
            best_inliers = std::move(inliers);
        }
    }
    if (best_count < 0 || best_inliers.size() < 3U
        || !refine_plane(candidates, best_inliers, refined_plane)) {
        return false;
    }

    std::array<int, 3> view_counts{};
    refined_inliers.clear();
    refined_inliers.reserve(candidates.size());
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        const CandidatePoint& candidate = candidates[index];
        const float residual = std::abs(
            refined_plane.normal.dot(candidate.point) - refined_plane.distance);
        if (residual <= threshold) {
            refined_inliers.push_back(static_cast<int>(index));
            if (candidate.view >= 0 && candidate.view < 3) {
                ++view_counts[static_cast<std::size_t>(candidate.view)];
            }
        }
    }
    return refined_inliers.size() >= 3U
        && std::min({view_counts[0], view_counts[1], view_counts[2]})
            >= kMinPlaneInliersPerView;
}

std::vector<CandidatePoint> collect_balanced_candidates(
    const std::array<GroupWorldView, 3>& views,
    const std::array<Sim3, 3>& transforms) {
    std::array<std::vector<CandidatePoint>, 3> per_view;
    for (int view_index = 0; view_index < 3; ++view_index) {
        const GroupWorldView& view = views[static_cast<std::size_t>(view_index)];
        const int rows = view.world_points.rows;
        const int columns = view.world_points.cols;
        const std::size_t total = static_cast<std::size_t>(rows)
            * static_cast<std::size_t>(columns);
        const int step = std::max(1, static_cast<int>(std::ceil(std::sqrt(
            static_cast<double>(total) / static_cast<double>(kMaxPlaneSamplesPerView)))));
        per_view[static_cast<std::size_t>(view_index)].reserve(kMaxPlaneSamplesPerView);
        for (int y = step / 2; y < rows; y += step) {
            for (int x = step / 2; x < columns; x += step) {
                const cv::Vec3f point = apply_sim3(
                    transforms[static_cast<std::size_t>(view_index)],
                    view.world_points.at<cv::Vec3f>(y, x));
                const float confidence = view.world_confidence.at<float>(y, x);
                if (finite_vec(point) && std::isfinite(confidence)) {
                    per_view[static_cast<std::size_t>(view_index)].push_back(
                        CandidatePoint{point, confidence, view_index, x, y});
                }
            }
        }
    }
    const std::size_t balanced_count = std::min({
        per_view[0].size(), per_view[1].size(), per_view[2].size()});
    std::vector<CandidatePoint> result;
    result.reserve(balanced_count * 3U);
    for (int view_index = 0; view_index < 3; ++view_index) {
        const auto& source = per_view[static_cast<std::size_t>(view_index)];
        result.insert(result.end(), source.begin(), source.begin()
            + static_cast<std::ptrdiff_t>(balanced_count));
    }
    return result;
}

float normalized_confidence(const float value, const float lower, const float upper) {
    if (!std::isfinite(value)) {
        return 0.0f;
    }
    if (!std::isfinite(lower) || !std::isfinite(upper) || upper - lower <= kNumericEpsilon) {
        return 1.0f;
    }
    return std::clamp((value - lower) / (upper - lower), 0.0f, 1.0f);
}

float border_weight(const int x, const int y, const int width, const int height) {
    const int distance = std::min({x, y, width - 1 - x, height - 1 - y});
    // The model crop boundary is less trustworthy, but it remains a valid
    // geometric sample. This weight is never used as a brightness mask.
    return std::clamp(static_cast<float>(distance) / 16.0f, 0.25f, 1.0f);
}

std::uint32_t color_to_rgba(const cv::Vec3f& color) {
    const auto to_byte = [](const float value) {
        return static_cast<std::uint8_t>(std::lround(
            std::clamp(std::isfinite(value) ? value : 0.0f, 0.0f, 1.0f) * 255.0f));
    };
    return pack_rgba(to_byte(color[0]), to_byte(color[1]), to_byte(color[2]));
}

float rgba_luminance(const std::uint32_t rgba) {
    const auto channels = unpack_rgba(rgba);
    return 0.2126f * static_cast<float>(channels[0]) / 255.0f
        + 0.7152f * static_cast<float>(channels[1]) / 255.0f
        + 0.0722f * static_cast<float>(channels[2]) / 255.0f;
}

std::uint32_t apply_global_color_gain(
    const std::uint32_t rgba,
    const float gain) {
    const auto channels = unpack_rgba(rgba);
    const auto scale_channel = [gain](const std::uint8_t channel) {
        return static_cast<std::uint8_t>(std::lround(std::clamp(
            static_cast<float>(channel) * gain, 0.0f, 255.0f)));
    };
    return pack_rgba(
        scale_channel(channels[0]),
        scale_channel(channels[1]),
        scale_channel(channels[2]),
        channels[3]);
}

bool valid_model_mats(const GroupWorldView& view) {
    return !view.world_points.empty()
        && !view.world_confidence.empty()
        && !view.rgb.empty()
        && view.world_points.type() == CV_32FC3
        && view.world_confidence.type() == CV_32FC1
        && view.rgb.type() == CV_32FC3
        && view.world_points.size() == view.world_confidence.size()
        && view.world_points.size() == view.rgb.size();
}

bool valid_aligned_point(
    const cv::Mat& aligned_points,
    const cv::Mat& confidence,
    const int x,
    const int y,
    cv::Vec3f& point) {
    if (x < 0 || y < 0 || x >= aligned_points.cols || y >= aligned_points.rows) {
        return false;
    }
    point = aligned_points.at<cv::Vec3f>(y, x);
    return finite_vec(point) && std::isfinite(confidence.at<float>(y, x));
}

bool local_object_continuity(
    const cv::Mat& aligned_points,
    const cv::Mat& confidence,
    const int x,
    const int y,
    const cv::Vec3f& point,
    const float scene_scale) {
    std::vector<cv::Vec3f> neighbors;
    std::vector<float> spacing;
    neighbors.reserve(8U);
    spacing.reserve(16U);
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            const int nx = x + dx;
            const int ny = y + dy;
            cv::Vec3f neighbor;
            if (!valid_aligned_point(aligned_points, confidence, nx, ny, neighbor)) {
                continue;
            }
            if (dx != 0 || dy != 0) {
                neighbors.push_back(neighbor);
            }
            // Estimate spacing strictly inside this pixel's 3x3 window. The
            // previous implementation also compared the window's right/bottom
            // edge with pixels at x+2/y+2; a single curtain/ray outside the
            // neighborhood could therefore inflate the robust spacing and
            // let the same outlier pass its continuity test.
            if (dx < 1 && nx + 1 < aligned_points.cols) {
                cv::Vec3f right;
                if (valid_aligned_point(aligned_points, confidence, nx + 1, ny, right)) {
                    spacing.push_back(vector_norm(right - neighbor));
                }
            }
            if (dy < 1 && ny + 1 < aligned_points.rows) {
                cv::Vec3f below;
                if (valid_aligned_point(aligned_points, confidence, nx, ny + 1, below)) {
                    spacing.push_back(vector_norm(below - neighbor));
                }
            }
        }
    }
    if (neighbors.size() < 3U || spacing.size() < 2U) {
        return false;
    }
    const cv::Vec3f local_median = componentwise_median(neighbors);
    const float local_distance = vector_norm(point - local_median);
    const float local_spacing = median_value(std::move(spacing));
    if (!std::isfinite(local_distance) || !std::isfinite(local_spacing)
        || local_spacing <= kNumericEpsilon || !std::isfinite(scene_scale)) {
        return false;
    }
    const float threshold = std::max(4.0f * local_spacing, 0.015f * scene_scale);
    return std::isfinite(threshold) && local_distance <= threshold;
}

bool floor_neighborhood_consistent(
    const cv::Mat& aligned_points,
    const cv::Mat& confidence,
    const int x,
    const int y,
    const cv::Vec3f& plane_origin,
    const cv::Vec3f& plane_normal,
    const float floor_band) {
    std::vector<float> absolute_heights;
    absolute_heights.reserve(9U);
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            cv::Vec3f point;
            if (!valid_aligned_point(aligned_points, confidence, x + dx, y + dy, point)) {
                continue;
            }
            absolute_heights.push_back(std::abs(
                plane_normal.dot(point - plane_origin)));
        }
    }
    return absolute_heights.size() >= 3U
        && median_value(std::move(absolute_heights)) <= 1.5f * floor_band;
}

int distinct_view_count(const HeightCluster& cluster) {
    std::array<bool, 3> seen{false, false, false};
    for (const ObjectSample& sample : cluster.samples) {
        if (sample.view >= 0 && sample.view < 3) {
            seen[static_cast<std::size_t>(sample.view)] = true;
        }
    }
    return static_cast<int>(std::count(seen.begin(), seen.end(), true));
}

float weighted_median_height(const std::vector<ObjectSample>& samples) {
    std::vector<std::pair<float, float>> values;
    values.reserve(samples.size());
    for (const ObjectSample& sample : samples) {
        if (std::isfinite(sample.height) && std::isfinite(sample.weight)
            && sample.weight > 0.0f) {
            values.emplace_back(sample.height, sample.weight);
        }
    }
    if (values.empty()) {
        return 0.0f;
    }
    std::sort(values.begin(), values.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.first < rhs.first;
    });
    float total = 0.0f;
    for (const auto& value : values) {
        total += value.second;
    }
    float accumulated = 0.0f;
    for (const auto& value : values) {
        accumulated += value.second;
        if (accumulated >= total * 0.5f) {
            return value.first;
        }
    }
    return values.back().first;
}

cv::Vec3f weighted_color(
    const std::vector<ObjectSample>& samples,
    const std::array<std::array<float, 3>, 3>& gains,
    float& confidence) {
    cv::Vec3f color(0.0f, 0.0f, 0.0f);
    float color_weight = 0.0f;
    float confidence_weight = 0.0f;
    for (const ObjectSample& sample : samples) {
        const float weight = std::max(sample.weight, 0.0f);
        if (weight <= 0.0f) {
            continue;
        }
        const std::size_t view_index = sample.view >= 0 && sample.view < 3
            ? static_cast<std::size_t>(sample.view)
            : 0U;
        const auto& gain = gains[view_index];
        const cv::Vec3f corrected(
            std::clamp(sample.color[0] * gain[0], 0.0f, 1.0f),
            std::clamp(sample.color[1] * gain[1], 0.0f, 1.0f),
            std::clamp(sample.color[2] * gain[2], 0.0f, 1.0f));
        if (finite_vec(corrected)) {
            color += corrected * weight;
            color_weight += weight;
        }
        confidence += sample.normalized_confidence * weight;
        confidence_weight += weight;
    }
    if (color_weight > kNumericEpsilon) {
        color *= 1.0f / color_weight;
    }
    confidence = confidence_weight > kNumericEpsilon
        ? std::clamp(confidence / confidence_weight, 0.0f, 1.0f)
        : 0.0f;
    return color;
}

}  // namespace

void GroupWorldFusion::reset() {
    reference_initialized_ = false;
    reference_centers_ = {};
    plane_origin_ = cv::Vec3f(0.0f, 0.0f, 0.0f);
    plane_normal_ = cv::Vec3f(0.0f, 0.0f, 1.0f);
    axis_x_ = cv::Vec3f(1.0f, 0.0f, 0.0f);
    axis_y_ = cv::Vec3f(0.0f, 1.0f, 0.0f);
    u_min_ = 0.0f;
    u_max_ = 0.0f;
    v_min_ = 0.0f;
    v_max_ = 0.0f;
    center_u_ = 0.0f;
    center_v_ = 0.0f;
    display_scale_ = 1.0f;
    scene_scale_ = 1.0f;
    floor_band_ = 0.01f;
    max_object_height_ = 1.0f;
    global_color_gain_ = 1.0f;
    color_gain_ = {{
        {{1.0f, 1.0f, 1.0f}},
        {{1.0f, 1.0f, 1.0f}},
        {{1.0f, 1.0f, 1.0f}}}};
    canvas_width_ = 0;
    canvas_height_ = 0;
    logical_width_ = 0;
    logical_height_ = 0;
    floor_cells_.clear();
    floor_cell_valid_.clear();
}

GroupWorldFusionResult GroupWorldFusion::fuse(
    const std::array<GroupWorldView, 3>& views,
    const CanvasState& state) {
    GroupWorldFusionResult result;
    result.scene_scale = scene_scale_;
    result.floor_band = floor_band_;
    const auto reject = [&](const char* reason) {
        result.accepted = false;
        result.rejection_reason = reason;
        return result;
    };
    if (state.width <= 1 || state.height <= 1
        || (state.width & 1) != 0 || (state.height & 1) != 0) {
        return reject("world fusion requires an even, positive canvas");
    }
    if (!state.shape_valid() || state.slot_count() == 0U) {
        return reject("world fusion requires a shaped CanvasState");
    }
    if (canvas_width_ == 0) {
        canvas_width_ = state.width;
        canvas_height_ = state.height;
        logical_width_ = canvas_width_ / 2;
        logical_height_ = canvas_height_ / 2;
    } else if (canvas_width_ != state.width || canvas_height_ != state.height) {
        return reject("world fusion canvas dimensions changed after initialization");
    }
    for (const GroupWorldView& view : views) {
        if (!valid_model_mats(view)) {
            return reject("group point head returned invalid world-point maps");
        }
    }

    std::array<cv::Vec3f, 3> current_centers{};
    for (int index = 0; index < 3; ++index) {
        if (!camera_center(views[static_cast<std::size_t>(index)],
                current_centers[static_cast<std::size_t>(index)])) {
            return reject("group pose is missing or invalid");
        }
    }

    const auto median_baseline = [](const std::array<cv::Vec3f, 3>& centers) {
        std::vector<float> distances;
        distances.reserve(3U);
        for (int first = 0; first < 3; ++first) {
            for (int second = first + 1; second < 3; ++second) {
                distances.push_back(vector_norm(
                    centers[static_cast<std::size_t>(first)]
                    - centers[static_cast<std::size_t>(second)]));
            }
        }
        return median_value(std::move(distances));
    };

    std::array<Sim3, 3> transforms{};
    if (!reference_initialized_) {
        const float reference_baseline = median_baseline(current_centers);
        if (!std::isfinite(reference_baseline) || reference_baseline <= kBaselineEpsilon) {
            return reject("reference camera rig baseline is degenerate");
        }
        transforms = {Sim3{}, Sim3{}, Sim3{}};
        const std::vector<CandidatePoint> candidates = collect_balanced_candidates(views, transforms);
        if (candidates.size() < static_cast<std::size_t>(3 * kMinPlaneInliersPerView)) {
            return reject("group point head has too few finite balanced samples");
        }
        cv::Vec3f robust_center(0.0f, 0.0f, 0.0f);
        for (int axis = 0; axis < 3; ++axis) {
            std::vector<float> values;
            values.reserve(candidates.size());
            for (const CandidatePoint& candidate : candidates) {
                values.push_back(candidate.point[axis]);
            }
            robust_center[axis] = median_value(std::move(values));
        }
        std::vector<float> radii;
        radii.reserve(candidates.size());
        for (const CandidatePoint& candidate : candidates) {
            radii.push_back(vector_norm(candidate.point - robust_center));
        }
        const float scene_scale = percentile_value(std::move(radii), 0.90f);
        if (!std::isfinite(scene_scale) || scene_scale <= kNumericEpsilon) {
            return reject("reference world-point scene scale is degenerate");
        }

        Plane plane;
        std::vector<int> inliers;
        if (!find_dominant_plane(candidates, scene_scale, plane, inliers)) {
            return reject("no deterministic floor plane has inliers in all three views");
        }
        std::vector<float> residuals;
        residuals.reserve(inliers.size());
        cv::Vec3f inlier_center(0.0f, 0.0f, 0.0f);
        for (const int index : inliers) {
            const cv::Vec3f& point = candidates[static_cast<std::size_t>(index)].point;
            residuals.push_back(std::abs(plane.normal.dot(point) - plane.distance));
            inlier_center += point;
        }
        inlier_center *= 1.0f / static_cast<float>(inliers.size());
        const float residual_median = median_value(residuals);
        std::vector<float> deviations;
        deviations.reserve(residuals.size());
        for (const float residual : residuals) {
            deviations.push_back(std::abs(residual - residual_median));
        }
        const float mad = median_value(std::move(deviations));
        const float floor_band = std::clamp(
            3.0f * 1.4826f * mad,
            0.008f * scene_scale,
            0.03f * scene_scale);
        if (!std::isfinite(floor_band) || floor_band <= 0.0f) {
            return reject("floor residual band is invalid");
        }

        // The camera side of the plane is positive Z. The camera centers are
        // the only orientation cue; no RGB/brightness rule participates.
        const float camera_signed_distance = median_value({
            plane.normal.dot(current_centers[0]) - plane.distance,
            plane.normal.dot(current_centers[1]) - plane.distance,
            plane.normal.dot(current_centers[2]) - plane.distance});
        if (!std::isfinite(camera_signed_distance)) {
            return reject("reference camera height above floor is invalid");
        }
        if (camera_signed_distance < 0.0f) {
            plane.normal *= -1.0f;
            plane.distance *= -1.0f;
        }
        const float reference_camera_height = median_value({
            plane.normal.dot(current_centers[0]) - plane.distance,
            plane.normal.dot(current_centers[1]) - plane.distance,
            plane.normal.dot(current_centers[2]) - plane.distance});
        if (!std::isfinite(reference_camera_height) || reference_camera_height <= kNumericEpsilon) {
            return reject("reference camera height above floor is degenerate");
        }

        cv::Vec3f axis_x = current_centers[2] - current_centers[0];
        axis_x -= plane.normal * plane.normal.dot(axis_x);
        if (!normalize_vec(axis_x)) {
            axis_x = cv::Vec3f(1.0f, 0.0f, 0.0f);
            axis_x -= plane.normal * plane.normal.dot(axis_x);
            if (!normalize_vec(axis_x)) {
                axis_x = cv::Vec3f(0.0f, 1.0f, 0.0f);
                axis_x -= plane.normal * plane.normal.dot(axis_x);
                if (!normalize_vec(axis_x)) {
                    return reject("reference floor axis is degenerate");
                }
            }
        }
        cv::Vec3f axis_y = plane.normal.cross(axis_x);
        if (!normalize_vec(axis_y)) {
            return reject("reference floor second axis is degenerate");
        }

        // Atlas bounds are estimated only from the RANSAC-refined floor
        // inliers.  A small number of remote point-head rays must not turn
        // the fixed canvas into a giant sparse rectangle.
        std::vector<FloorUVSample> floor_uv;
        floor_uv.reserve(inliers.size());
        for (const int index : inliers) {
            const CandidatePoint& candidate = candidates[static_cast<std::size_t>(index)];
            if (!std::isfinite(candidate.confidence)) {
                continue;
            }
            const cv::Vec3f delta = candidate.point - inlier_center;
            const float height = plane.normal.dot(delta);
            if (std::abs(height) <= floor_band) {
                floor_uv.push_back(FloorUVSample{
                    axis_x.dot(delta), axis_y.dot(delta)});
            }
        }
        if (floor_uv.size() < 3U) {
            return reject("reference floor has too few finite XY samples");
        }
        std::vector<float> floor_u;
        std::vector<float> floor_v;
        floor_u.reserve(floor_uv.size());
        floor_v.reserve(floor_uv.size());
        for (const FloorUVSample& sample : floor_uv) {
            floor_u.push_back(sample.u);
            floor_v.push_back(sample.v);
        }
        const float uv_center_u = median_value(floor_u);
        const float uv_center_v = median_value(floor_v);
        std::vector<float> uv_radii;
        uv_radii.reserve(floor_uv.size());
        for (const FloorUVSample& sample : floor_uv) {
            uv_radii.push_back(std::hypot(sample.u - uv_center_u, sample.v - uv_center_v));
        }
        const float radius_median = median_value(uv_radii);
        std::vector<float> radius_deviations;
        radius_deviations.reserve(uv_radii.size());
        for (const float radius : uv_radii) {
            radius_deviations.push_back(std::abs(radius - radius_median));
        }
        const float radius_mad = median_value(radius_deviations);
        const float radius_limit = radius_median + 4.0f * 1.4826f
            * std::max(radius_mad, kNumericEpsilon);
        std::vector<float> robust_floor_u;
        std::vector<float> robust_floor_v;
        robust_floor_u.reserve(floor_uv.size());
        robust_floor_v.reserve(floor_uv.size());
        for (std::size_t index = 0; index < floor_uv.size(); ++index) {
            if (uv_radii[index] <= radius_limit) {
                robust_floor_u.push_back(floor_uv[index].u);
                robust_floor_v.push_back(floor_uv[index].v);
            }
        }
        if (robust_floor_u.size() < 3U || robust_floor_v.size() < 3U) {
            return reject("reference floor robust XY support is too small");
        }
        const float u_two = percentile_value(robust_floor_u, 0.02f);
        const float u_ninety_eight = percentile_value(robust_floor_u, 0.98f);
        const float v_two = percentile_value(robust_floor_v, 0.02f);
        const float v_ninety_eight = percentile_value(robust_floor_v, 0.98f);
        const float raw_u_span = std::max(u_ninety_eight - u_two, 0.01f * scene_scale);
        const float raw_v_span = std::max(v_ninety_eight - v_two, 0.01f * scene_scale);
        const float u_min = u_two - 0.08f * raw_u_span;
        const float u_max = u_ninety_eight + 0.08f * raw_u_span;
        const float v_min = v_two - 0.08f * raw_v_span;
        const float v_max = v_ninety_eight + 0.08f * raw_v_span;

        reference_initialized_ = true;
        reference_centers_ = current_centers;
        plane_origin_ = inlier_center;
        plane_normal_ = plane.normal;
        axis_x_ = axis_x;
        axis_y_ = axis_y;
        scene_scale_ = scene_scale;
        floor_band_ = floor_band;
        max_object_height_ = 0.90f * reference_camera_height;
        u_min_ = u_min;
        u_max_ = u_max;
        v_min_ = v_min;
        v_max_ = v_max;
        center_u_ = 0.5f * (u_min_ + u_max_);
        center_v_ = 0.5f * (v_min_ + v_max_);
        // The larger expanded floor span covers about 0.9 GUI units. This
        // is also the fixed world-units-per-GUI-unit scale used for Z.
        display_scale_ = std::max(u_max_ - u_min_, v_max_ - v_min_) / 0.9f;
        if (!std::isfinite(display_scale_) || display_scale_ <= kNumericEpsilon) {
            reset();
            return reject("reference display scale is invalid");
        }
        floor_cells_.assign(
            static_cast<std::size_t>(logical_width_)
                * static_cast<std::size_t>(logical_height_),
            FusedSlot{});
        floor_cell_valid_.assign(floor_cells_.size(), 0U);
    } else {
        const float reference_baseline = median_baseline(reference_centers_);
        if (!std::isfinite(reference_baseline) || reference_baseline <= kBaselineEpsilon) {
            return reject("stored reference camera rig baseline is degenerate");
        }
        Sim3 current_to_reference;
        float aligned_rms = 0.0f;
        if (!calculate_sim3(
                current_centers, reference_centers_, current_to_reference, aligned_rms)) {
            return reject("current camera rig Sim(3) is invalid or outside scale gate");
        }
        const float normalized_rms = aligned_rms / reference_baseline;
        if (!std::isfinite(normalized_rms) || normalized_rms >= 0.08f) {
            return reject("current camera rig Sim(3) RMS gate rejected the group");
        }
        transforms = {current_to_reference, current_to_reference, current_to_reference};
    }

    result.scene_scale = scene_scale_;
    result.floor_band = floor_band_;
    const std::size_t cell_count = static_cast<std::size_t>(logical_width_)
        * static_cast<std::size_t>(logical_height_);
    std::vector<float> confidence_values;
    for (const GroupWorldView& view : views) {
        for (int y = 0; y < view.world_confidence.rows; ++y) {
            for (int x = 0; x < view.world_confidence.cols; ++x) {
                const float confidence = view.world_confidence.at<float>(y, x);
                if (std::isfinite(confidence)) {
                    confidence_values.push_back(confidence);
                }
            }
        }
    }
    if (confidence_values.empty()) {
        return reject("group world-point confidence map is empty");
    }
    const float confidence_lower = percentile_value(confidence_values, 0.10f);
    const float confidence_upper = percentile_value(confidence_values, 0.90f);

    // Materialize the three aligned point-head maps once.  The local object
    // continuity test below must compare points in the same (reference)
    // world frame as the final floor/object classification.
    std::array<cv::Mat, 3> aligned_world_points;
    for (int view_index = 0; view_index < 3; ++view_index) {
        const GroupWorldView& view = views[static_cast<std::size_t>(view_index)];
        aligned_world_points[static_cast<std::size_t>(view_index)] = cv::Mat(
            view.world_points.rows, view.world_points.cols, CV_32FC3);
        const Sim3& transform = transforms[static_cast<std::size_t>(view_index)];
        for (int y = 0; y < view.world_points.rows; ++y) {
            for (int x = 0; x < view.world_points.cols; ++x) {
                const cv::Vec3f source = view.world_points.at<cv::Vec3f>(y, x);
                aligned_world_points[static_cast<std::size_t>(view_index)].at<cv::Vec3f>(y, x) =
                    finite_vec(source)
                    ? apply_sim3(transform, source)
                    : cv::Vec3f(
                        std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::quiet_NaN());
            }
        }
    }

    const float gui_scale = static_cast<float>(std::max(canvas_width_, canvas_height_));
    const auto world_to_cell = [&](const float u, const float v, int& logical_x, int& logical_y) {
        if (!std::isfinite(u) || !std::isfinite(v)
            || u < u_min_ || u > u_max_ || v < v_min_ || v > v_max_) {
            return false;
        }
        // This is the exact GUI point_from_slot() convention translated back
        // to a physical pixel. The logical cell is the even-origin 2x2 block
        // so all four layers share one GUI-derived XY neighborhood.
        const float physical_x = canvas_width_ * 0.5f
            + (u - center_u_) / display_scale_ * gui_scale;
        const float physical_y = canvas_height_ * 0.5f
            - (v - center_v_) / display_scale_ * gui_scale;
        const int rounded_x = static_cast<int>(std::lround(physical_x));
        const int rounded_y = static_cast<int>(std::lround(physical_y));
        if (rounded_x < 0 || rounded_x >= canvas_width_
            || rounded_y < 0 || rounded_y >= canvas_height_) {
            return false;
        }
        logical_x = rounded_x / 2;
        logical_y = rounded_y / 2;
        return logical_x >= 0 && logical_x < logical_width_
            && logical_y >= 0 && logical_y < logical_height_;
    };
    const auto slot_for = [&](const int logical_x, const int logical_y, const int layer) {
        const int dx = (layer == 1 || layer == 3) ? 1 : 0;
        const int dy = (layer >= 2) ? 1 : 0;
        const int pixel_x = logical_x * 2 + dx;
        const int pixel_y = logical_y * 2 + dy;
        if (layer < 0 || layer > 3 || pixel_x < 0 || pixel_x >= canvas_width_
            || pixel_y < 0 || pixel_y >= canvas_height_) {
            return std::numeric_limits<std::uint32_t>::max();
        }
        return static_cast<std::uint32_t>(pixel_y * canvas_width_ + pixel_x);
    };

    std::vector<FloorSample> floor_best(cell_count * 3U);
    std::vector<std::vector<ObjectSample>> object_samples(cell_count);
    for (int view_index = 0; view_index < 3; ++view_index) {
        const GroupWorldView& view = views[static_cast<std::size_t>(view_index)];
        const cv::Mat& aligned_points = aligned_world_points[static_cast<std::size_t>(view_index)];
        for (int y = 0; y < view.world_points.rows; ++y) {
            for (int x = 0; x < view.world_points.cols; ++x) {
                const cv::Vec3f point = aligned_points.at<cv::Vec3f>(y, x);
                const float confidence = view.world_confidence.at<float>(y, x);
                if (!finite_vec(point) || !std::isfinite(confidence)) {
                    continue;
                }
                const cv::Vec3f delta = point - plane_origin_;
                const float u = axis_x_.dot(delta);
                const float v = axis_y_.dot(delta);
                const float height = plane_normal_.dot(delta);
                int logical_x = 0;
                int logical_y = 0;
                if (!world_to_cell(u, v, logical_x, logical_y)) {
                    continue;
                }
                const std::size_t cell = static_cast<std::size_t>(logical_y)
                    * static_cast<std::size_t>(logical_width_)
                    + static_cast<std::size_t>(logical_x);
                const float normalized = normalized_confidence(
                    confidence, confidence_lower, confidence_upper);
                const float weight = normalized * border_weight(
                    x, y, view.world_points.cols, view.world_points.rows);
                if (std::abs(height) <= floor_band_) {
                    if (!floor_neighborhood_consistent(
                            aligned_points, view.world_confidence, x, y,
                            plane_origin_, plane_normal_, floor_band_)) {
                        continue;
                    }
                    FloorSample& best = floor_best[
                        static_cast<std::size_t>(view_index) * cell_count + cell];
                    if (!best.valid || weight > best.weight) {
                        best.color = view.rgb.at<cv::Vec3f>(y, x);
                        best.normalized_confidence = normalized;
                        best.weight = weight;
                        best.valid = true;
                    }
                } else if (height > 1.5f * floor_band_
                    && height < max_object_height_
                    && local_object_continuity(
                        aligned_points, view.world_confidence, x, y, point, scene_scale_)) {
                    object_samples[cell].push_back(ObjectSample{
                        height,
                        normalized,
                        std::max(weight, 1e-4f),
                        view.rgb.at<cv::Vec3f>(y, x),
                        view_index});
                }
                // The transition band and points below the plane are
                // intentionally ignored. Neither decision uses RGB.
            }
        }
    }

    // Estimate side-camera exposure gains from actual three-dimensional
    // floor-cell overlap. Saturated channels are excluded independently;
    // dark object/floor pixels are never used as a geometry mask.
    color_gain_[1].fill(1.0f);
    for (const int side_view : {0, 2}) {
        std::array<std::vector<float>, 3> ratios;
        for (std::size_t cell = 0; cell < cell_count; ++cell) {
            const FloorSample& anchor = floor_best[cell_count + cell];
            const FloorSample& side = floor_best[
                static_cast<std::size_t>(side_view) * cell_count + cell];
            if (!anchor.valid || !side.valid) {
                continue;
            }
            for (int channel = 0; channel < 3; ++channel) {
                const float anchor_value = anchor.color[channel];
                const float side_value = side.color[channel];
                if (std::isfinite(anchor_value) && std::isfinite(side_value)
                    && anchor_value >= 0.08f && anchor_value <= 0.92f
                    && side_value >= 0.08f && side_value <= 0.92f
                    && side_value > kNumericEpsilon) {
                    ratios[static_cast<std::size_t>(channel)].push_back(
                        anchor_value / side_value);
                }
            }
        }
        // Keep separate gains for the two physical side cameras. The array
        // is indexed by model slot, so the anchor (slot 1) remains exactly 1.
        const auto previous_gain = color_gain_[static_cast<std::size_t>(side_view)];
        std::array<float, 3> measured = previous_gain;
        for (int channel = 0; channel < 3; ++channel) {
            auto& channel_ratios = ratios[static_cast<std::size_t>(channel)];
            if (channel_ratios.size() < kMinimumGainOverlap) {
                continue;
            }
            measured[static_cast<std::size_t>(channel)] = std::clamp(
                median_value(std::move(channel_ratios)), 0.75f, 1.33f);
            color_gain_[static_cast<std::size_t>(side_view)][static_cast<std::size_t>(channel)] =
                (1.0f - kGainAlpha) * previous_gain[static_cast<std::size_t>(channel)]
                + kGainAlpha * measured[static_cast<std::size_t>(channel)];
        }
        for (int channel = 0; channel < 3; ++channel) {
            color_gain_[static_cast<std::size_t>(side_view)][static_cast<std::size_t>(channel)] =
                std::clamp(
                    color_gain_[static_cast<std::size_t>(side_view)][static_cast<std::size_t>(channel)],
                    0.75f,
                    1.33f);
        }
    }

    // Update persistent floor cells. A cell not seen in this group is left
    // untouched, so a temporary arm occlusion cannot erase static ground.
    for (std::size_t cell = 0; cell < cell_count; ++cell) {
        float color_weight = 0.0f;
        float total_weight = 0.0f;
        float confidence_weight = 0.0f;
        cv::Vec3f color(0.0f, 0.0f, 0.0f);
        for (int view_index = 0; view_index < 3; ++view_index) {
            const FloorSample& sample = floor_best[
                static_cast<std::size_t>(view_index) * cell_count + cell];
            if (!sample.valid || !std::isfinite(sample.weight) || sample.weight <= 0.0f) {
                continue;
            }
            const auto& gain = color_gain_[static_cast<std::size_t>(view_index)];
            const cv::Vec3f corrected(
                std::clamp(sample.color[0] * gain[0], 0.0f, 1.0f),
                std::clamp(sample.color[1] * gain[1], 0.0f, 1.0f),
                std::clamp(sample.color[2] * gain[2], 0.0f, 1.0f));
            // Exact black model padding has no valid color sample. It is
            // skipped here without affecting the geometric floor decision;
            // genuinely dark object geometry remains eligible above.
            const bool black_padding = std::abs(sample.color[0]) <= kNumericEpsilon
                && std::abs(sample.color[1]) <= kNumericEpsilon
                && std::abs(sample.color[2]) <= kNumericEpsilon;
            if (!black_padding && finite_vec(corrected)) {
                color += corrected * sample.weight;
                color_weight += sample.weight;
            }
            total_weight += sample.weight;
            confidence_weight += sample.normalized_confidence * sample.weight;
        }
        if (color_weight > kNumericEpsilon) {
            color *= 1.0f / color_weight;
        }
        if (confidence_weight > kNumericEpsilon && total_weight > kNumericEpsilon) {
            confidence_weight = std::clamp(
                confidence_weight / total_weight, 0.0f, 1.0f);
        }
        if (floor_best[cell].valid || floor_best[cell_count + cell].valid
            || floor_best[2U * cell_count + cell].valid) {
            // A floor cell's XY is fixed by the first reference atlas. Only
            // its observed color/confidence may change from frame to frame.
            const int logical_x = static_cast<int>(cell % static_cast<std::size_t>(logical_width_));
            const int logical_y = static_cast<int>(cell / static_cast<std::size_t>(logical_width_));
            FusedSlot next = floor_cells_[cell];
            next.slot_id = slot_for(logical_x, logical_y, 0);
            next.depth = 0.0f;
            next.confidence = confidence_weight;
            if (color_weight > kNumericEpsilon) {
                next.rgba = color_to_rgba(color);
            }
            next.floor = true;
            floor_cells_[cell] = next;
            floor_cell_valid_[cell] = 1U;
        }
    }
    result.slots.reserve(cell_count / 2U);
    result.occupied_slots.reserve(cell_count / 2U);
    for (std::size_t cell = 0; cell < cell_count; ++cell) {
        if (floor_cell_valid_[cell] != 0U) {
            result.slots.push_back(floor_cells_[cell]);
            result.occupied_slots.push_back(floor_cells_[cell].slot_id);
        }
    }

    // Quantize object points to the same logical XY cells. Production keeps
    // one dominant object surface per cell; layer 2/3 remain reserved by the
    // 2x2 layout for compatibility but are intentionally not written here.
    const float cluster_tolerance = std::max(2.0f * floor_band_, 0.01f * scene_scale_);
    std::vector<HeightCluster> selected_clusters(cell_count);
    std::vector<std::uint8_t> selected_valid(cell_count, 0U);
    for (std::size_t cell = 0; cell < cell_count; ++cell) {
        auto& samples = object_samples[cell];
        if (samples.empty()) {
            continue;
        }
        std::sort(samples.begin(), samples.end(), [](const ObjectSample& lhs, const ObjectSample& rhs) {
            return lhs.height < rhs.height;
        });
        std::vector<HeightCluster> clusters;
        for (const ObjectSample& sample : samples) {
            if (clusters.empty()
                || sample.height - clusters.back().height > cluster_tolerance) {
                HeightCluster cluster;
                cluster.samples.push_back(sample);
                cluster.total_weight = sample.weight;
                cluster.height = sample.height;
                clusters.push_back(std::move(cluster));
            } else {
                HeightCluster& cluster = clusters.back();
                cluster.samples.push_back(sample);
                cluster.total_weight += sample.weight;
                cluster.height = weighted_median_height(cluster.samples);
            }
        }
        if (clusters.empty()) {
            continue;
        }
        std::size_t selected_index = 0U;
        float selected_score = -1.0f;
        for (std::size_t index = 0; index < clusters.size(); ++index) {
            const HeightCluster& cluster = clusters[index];
            const float score = cluster.total_weight
                * std::sqrt(static_cast<float>(cluster.samples.size()));
            if (score > selected_score
                || (score == selected_score && cluster.samples.size()
                    > clusters[selected_index].samples.size())
                || (score == selected_score
                    && cluster.samples.size() == clusters[selected_index].samples.size()
                    && cluster.height < clusters[selected_index].height)) {
                selected_index = index;
                selected_score = score;
            }
        }
        selected_clusters[cell] = std::move(clusters[selected_index]);
        selected_valid[cell] = 1U;
    }

    cv::Mat occupancy(logical_height_, logical_width_, CV_8UC1, cv::Scalar(0));
    for (std::size_t cell = 0; cell < cell_count; ++cell) {
        if (selected_valid[cell] != 0U) {
            const int x = static_cast<int>(cell % static_cast<std::size_t>(logical_width_));
            const int y = static_cast<int>(cell / static_cast<std::size_t>(logical_width_));
            occupancy.at<std::uint8_t>(y, x) = 255U;
        }
    }
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int component_count = cv::connectedComponentsWithStats(
        occupancy, labels, stats, centroids, 8, CV_32S);
    for (int y = 0; y < logical_height_; ++y) {
        for (int x = 0; x < logical_width_; ++x) {
            const int component = labels.at<int>(y, x);
            const std::size_t cell = static_cast<std::size_t>(y) * logical_width_ + x;
            if (selected_valid[cell] != 0U
                && (component <= 0 || component >= component_count
                    || stats.at<int>(component, cv::CC_STAT_AREA) < 8)) {
                selected_valid[cell] = 0U;
            }
        }
    }

    const float height_neighbor_threshold = std::max(
        4.0f * floor_band_, 0.03f * scene_scale_);
    for (int y = 0; y < logical_height_; ++y) {
        for (int x = 0; x < logical_width_; ++x) {
            const std::size_t cell = static_cast<std::size_t>(y) * logical_width_ + x;
            if (selected_valid[cell] == 0U
                || distinct_view_count(selected_clusters[cell]) != 1) {
                continue;
            }
            std::vector<float> neighbor_heights;
            neighbor_heights.reserve(8U);
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) {
                        continue;
                    }
                    const int nx = x + dx;
                    const int ny = y + dy;
                    if (nx < 0 || nx >= logical_width_ || ny < 0 || ny >= logical_height_) {
                        continue;
                    }
                    const std::size_t neighbor_cell =
                        static_cast<std::size_t>(ny) * logical_width_ + nx;
                    if (selected_valid[neighbor_cell] != 0U) {
                        neighbor_heights.push_back(selected_clusters[neighbor_cell].height);
                    }
                }
            }
            if (!neighbor_heights.empty()
                && std::abs(selected_clusters[cell].height
                    - median_value(std::move(neighbor_heights))) > height_neighbor_threshold) {
                selected_valid[cell] = 0U;
            }
        }
    }

    for (std::size_t cell = 0; cell < cell_count; ++cell) {
        if (selected_valid[cell] == 0U) {
            continue;
        }
        const int logical_x = static_cast<int>(cell % static_cast<std::size_t>(logical_width_));
        const int logical_y = static_cast<int>(cell / static_cast<std::size_t>(logical_width_));
        float confidence = 0.0f;
        const cv::Vec3f color = weighted_color(
            selected_clusters[cell].samples, color_gain_, confidence);
        const std::uint32_t slot_id = slot_for(logical_x, logical_y, 1);
        if (slot_id == std::numeric_limits<std::uint32_t>::max()) {
            continue;
        }
        result.slots.push_back(FusedSlot{
            slot_id,
            selected_clusters[cell].height / display_scale_,
            confidence,
            color_to_rgba(color),
            false});
        result.occupied_slots.push_back(slot_id);
    }

    // Brightness is a display-only correction. Geometry and all of the
    // floor/object decisions above have already completed without consulting
    // RGB. Keep the persistent floor cache in pre-gain colour space so a
    // cell that is temporarily occluded is not multiplied again next frame.
    std::vector<float> luminances;
    luminances.reserve(result.slots.size());
    for (const FusedSlot& slot : result.slots) {
        const auto channels = unpack_rgba(slot.rgba);
        if (channels[0] != 0U || channels[1] != 0U || channels[2] != 0U) {
            luminances.push_back(rgba_luminance(slot.rgba));
        }
    }
    if (!luminances.empty()) {
        const float median_luminance = median_value(luminances);
        if (std::isfinite(median_luminance) && median_luminance > kNumericEpsilon) {
            global_color_gain_ = std::clamp(
                0.25f / median_luminance, 1.0f, 1.8f);
        }
    }
    for (FusedSlot& slot : result.slots) {
        slot.rgba = apply_global_color_gain(slot.rgba, global_color_gain_);
    }

    result.accepted = true;
    return result;
}

}  // namespace omnivggt::observer
