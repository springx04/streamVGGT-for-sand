#include "group_world_fusion.hpp"

#if defined(_WIN32)
#include <opencv2/imgcodecs.hpp>
#endif
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace omnivggt::observer {

namespace {

bool finite_vec(const cv::Vec3f& value);

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
    float height = 0.0f;
    bool valid = false;
};

struct ObjectSample {
    float u = 0.0f;
    float v = 0.0f;
    float height = 0.0f;
    float normalized_confidence = 0.0f;
    float weight = 0.0f;
    cv::Vec3f color = cv::Vec3f(0.0f, 0.0f, 0.0f);
    int view = 0;
    bool direct_margin = false;
};

struct ViewObjectObservation {
    bool valid = false;
    int view_id = -1;
    int logical_x = -1;
    int logical_y = -1;
    float u = 0.0f;
    float v = 0.0f;
    float height = 0.0f;
    float confidence = 0.0f;
    float weight = 0.0f;
    cv::Vec3f color = cv::Vec3f(0.0f, 0.0f, 0.0f);
    int sample_count = 0;
    bool direct_margin = false;
};

struct FloorObservation {
    bool valid = false;
    bool near_floor = false;
    int view_id = -1;
    int logical_x = -1;
    int logical_y = -1;
    float height = 0.0f;
    FloorSample sample;
};

struct ObjectSurfaceCandidate {
    bool valid = false;
    int logical_x = -1;
    int logical_y = -1;
    std::uint8_t view_mask = 0U;
    std::array<ViewObjectObservation, 3> observations{};
    float height = 0.0f;
    float score = 0.0f;
    int sample_count = 0;
};

struct FloorRecoveryCandidate {
    bool valid = false;
    int logical_x = -1;
    int logical_y = -1;
    std::uint8_t view_mask = 0U;
    std::array<FloorObservation, 3> observations{};
    float height = 0.0f;
    float score = 0.0f;
};

struct FloorUVSample {
    float u = 0.0f;
    float v = 0.0f;
};

struct ConfidenceStats {
    float q10 = 0.0f;
    float q50 = 0.0f;
    float q90 = 0.0f;
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

enum class ReprojectionRelation {
    Support,
    Occluded,
    Contradict,
    Outside,
    Invalid,
};

struct ReprojectionResult {
    ReprojectionRelation relation = ReprojectionRelation::Invalid;
    float world_error = std::numeric_limits<float>::infinity();
    float depth_error = std::numeric_limits<float>::infinity();
    float pixel_x = 0.0f;
    float pixel_y = 0.0f;
};

struct ReprojectionStats {
    std::size_t atlas_pair = 0U;
    std::size_t bidir_support = 0U;
    std::size_t oneway_support = 0U;
    std::size_t occluded = 0U;
    std::size_t contradict = 0U;
    std::size_t outside = 0U;
    std::size_t invalid = 0U;
};

cv::Vec3f inverse_sim3_point(const Sim3& current_to_reference,
    const cv::Vec3f& reference_point) {
    if (!finite_vec(reference_point)
        || !std::isfinite(current_to_reference.scale)
        || current_to_reference.scale <= kNumericEpsilon) {
        return cv::Vec3f(
            std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::quiet_NaN());
    }
    return current_to_reference.rotation.t()
        * (reference_point - current_to_reference.translation)
        / current_to_reference.scale;
}

ReprojectionResult project_world_point_to_view(
    const cv::Vec3f& point_reference,
    const Sim3& current_to_reference,
    const GroupWorldView& target_view,
    const float world_tolerance,
    const float depth_tolerance) {
    ReprojectionResult result;
    const cv::Vec3f point_current = inverse_sim3_point(
        current_to_reference, point_reference);
    if (!finite_vec(point_current) || !target_view.has_pose
        || !finite_vec(target_view.translation)
        || !finite_quaternion(target_view.quaternion)
        || !std::isfinite(target_view.fov_h) || !std::isfinite(target_view.fov_w)
        || target_view.fov_h <= 0.01f || target_view.fov_w <= 0.01f
        || target_view.fov_h >= 3.13f || target_view.fov_w >= 3.13f
        || target_view.world_points.empty()) {
        return result;
    }

    cv::Matx33f rotation;
    if (!rotation_from_quaternion(target_view.quaternion, rotation)) {
        return result;
    }
    const cv::Vec3f camera_point = rotation * point_current + target_view.translation;
    if (!finite_vec(camera_point) || camera_point[2] <= kNumericEpsilon) {
        return result;
    }

    const float tan_half_fov_w = std::tan(0.5f * target_view.fov_w);
    const float tan_half_fov_h = std::tan(0.5f * target_view.fov_h);
    if (!std::isfinite(tan_half_fov_w) || !std::isfinite(tan_half_fov_h)
        || tan_half_fov_w <= kNumericEpsilon || tan_half_fov_h <= kNumericEpsilon) {
        return result;
    }
    const float width = static_cast<float>(target_view.world_points.cols);
    const float height = static_cast<float>(target_view.world_points.rows);
    const float fx = 0.5f * width / tan_half_fov_w;
    const float fy = 0.5f * height / tan_half_fov_h;
    const float cx = 0.5f * (width - 1.0f);
    const float cy = 0.5f * (height - 1.0f);
    result.pixel_x = fx * camera_point[0] / camera_point[2] + cx;
    result.pixel_y = fy * camera_point[1] / camera_point[2] + cy;
    if (!std::isfinite(result.pixel_x) || !std::isfinite(result.pixel_y)) {
        return result;
    }
    if (result.pixel_x < 0.0f || result.pixel_x > width - 1.0f
        || result.pixel_y < 0.0f || result.pixel_y > height - 1.0f) {
        result.relation = ReprojectionRelation::Outside;
        return result;
    }

    const int center_x = static_cast<int>(std::lround(result.pixel_x));
    const int center_y = static_cast<int>(std::lround(result.pixel_y));
    bool has_target_point = false;
    bool all_target_points_are_closer = true;
    bool has_support = false;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            const int x = center_x + dx;
            const int y = center_y + dy;
            if (x < 0 || y < 0 || x >= target_view.world_points.cols
                || y >= target_view.world_points.rows) {
                continue;
            }
            const cv::Vec3f target_point = target_view.world_points.at<cv::Vec3f>(y, x);
            if (!finite_vec(target_point)) {
                continue;
            }
            const float target_confidence =
                target_view.world_confidence.at<float>(y, x);
            if (!std::isfinite(target_confidence)) {
                continue;
            }
            const cv::Vec3f target_camera_point =
                rotation * target_point + target_view.translation;
            if (!finite_vec(target_camera_point)
                || target_camera_point[2] <= kNumericEpsilon) {
                continue;
            }
            has_target_point = true;
            const float world_error = vector_norm(target_point - point_current);
            const float depth_error =
                std::abs(target_camera_point[2] - camera_point[2]);
            if (std::isfinite(world_error) && std::isfinite(depth_error)) {
                result.world_error = std::min(result.world_error, world_error);
                result.depth_error = std::min(result.depth_error, depth_error);
                if (world_error <= world_tolerance && depth_error <= depth_tolerance) {
                    has_support = true;
                }
            }
            if (!(target_camera_point[2] < camera_point[2] - depth_tolerance)) {
                all_target_points_are_closer = false;
            }
        }
    }
    if (has_support) {
        result.relation = ReprojectionRelation::Support;
    } else if (!has_target_point) {
        result.relation = ReprojectionRelation::Invalid;
    } else if (all_target_points_are_closer) {
        result.relation = ReprojectionRelation::Occluded;
    } else {
        result.relation = ReprojectionRelation::Contradict;
    }
    return result;
}

void record_failed_reprojection(const ReprojectionRelation relation,
    ReprojectionStats& stats) {
    switch (relation) {
    case ReprojectionRelation::Support:
        break;
    case ReprojectionRelation::Occluded:
        ++stats.occluded;
        break;
    case ReprojectionRelation::Contradict:
        ++stats.contradict;
        break;
    case ReprojectionRelation::Outside:
        ++stats.outside;
        break;
    case ReprojectionRelation::Invalid:
        ++stats.invalid;
        break;
    }
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

int support_count(const std::uint8_t view_mask) {
    int count = 0;
    for (int view = 0; view < 3; ++view) {
        if ((view_mask & (static_cast<std::uint8_t>(1U) << view)) != 0U) {
            ++count;
        }
    }
    return count;
}

int consensus_coordinate(std::vector<int> coordinates) {
    if (coordinates.empty()) {
        return -1;
    }
    std::sort(coordinates.begin(), coordinates.end());
    if (coordinates.size() == 2U) {
        return static_cast<int>(std::lround(
            0.5 * static_cast<double>(coordinates[0] + coordinates[1])));
    }
    return coordinates[coordinates.size() / 2U];
}

float weighted_median_pairs(std::vector<std::pair<float, float>> values) {
    values.erase(std::remove_if(values.begin(), values.end(), [](const auto& value) {
        return !std::isfinite(value.first) || !std::isfinite(value.second)
            || value.second <= 0.0f;
    }), values.end());
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
    if (!std::isfinite(total) || total <= 0.0f) {
        return 0.0f;
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
    return weighted_median_pairs(std::move(values));
}

void finalize_object_candidate(ObjectSurfaceCandidate& candidate) {
    std::vector<std::pair<float, float>> heights;
    std::vector<int> x_coordinates;
    std::vector<int> y_coordinates;
    heights.reserve(3U);
    x_coordinates.reserve(3U);
    y_coordinates.reserve(3U);
    candidate.score = 0.0f;
    candidate.sample_count = 0;
    for (int view = 0; view < 3; ++view) {
        const std::uint8_t bit = static_cast<std::uint8_t>(1U) << view;
        if ((candidate.view_mask & bit) == 0U
            || !candidate.observations[static_cast<std::size_t>(view)].valid) {
            continue;
        }
        const ViewObjectObservation& observation =
            candidate.observations[static_cast<std::size_t>(view)];
        heights.emplace_back(observation.height, std::max(observation.weight, 1e-4f));
        x_coordinates.push_back(observation.logical_x);
        y_coordinates.push_back(observation.logical_y);
        candidate.score += std::max(observation.weight, 1e-4f);
        candidate.sample_count += std::max(observation.sample_count, 1);
    }
    candidate.height = weighted_median_pairs(std::move(heights));
    candidate.score *= std::sqrt(static_cast<float>(std::max(candidate.sample_count, 1)));
    if (candidate.logical_x < 0) {
        candidate.logical_x = consensus_coordinate(std::move(x_coordinates));
    }
    if (candidate.logical_y < 0) {
        candidate.logical_y = consensus_coordinate(std::move(y_coordinates));
    }
}

void finalize_floor_candidate(FloorRecoveryCandidate& candidate) {
    std::vector<std::pair<float, float>> heights;
    heights.reserve(3U);
    candidate.score = 0.0f;
    for (int view = 0; view < 3; ++view) {
        const std::uint8_t bit = static_cast<std::uint8_t>(1U) << view;
        if ((candidate.view_mask & bit) == 0U
            || !candidate.observations[static_cast<std::size_t>(view)].valid) {
            continue;
        }
        const FloorObservation& observation =
            candidate.observations[static_cast<std::size_t>(view)];
        heights.emplace_back(
            observation.height, std::max(observation.sample.weight, 1e-4f));
        candidate.score += std::max(observation.sample.weight, 1e-4f);
    }
    candidate.height = weighted_median_pairs(std::move(heights));
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

cv::Vec3f object_candidate_color(
    const ObjectSurfaceCandidate& candidate,
    float& confidence) {
    cv::Vec3f color(0.0f, 0.0f, 0.0f);
    float color_weight = 0.0f;
    float confidence_weight = 0.0f;
    for (int view = 0; view < 3; ++view) {
        const std::uint8_t bit = static_cast<std::uint8_t>(1U) << view;
        if ((candidate.view_mask & bit) == 0U
            || !candidate.observations[static_cast<std::size_t>(view)].valid) {
            continue;
        }
        const ViewObjectObservation& observation =
            candidate.observations[static_cast<std::size_t>(view)];
        const float weight = std::max(observation.weight, 1e-4f);
        if (finite_vec(observation.color)) {
            color += observation.color * weight;
            color_weight += weight;
        }
        confidence += observation.confidence * weight;
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
    accepted_fuse_count_ = 0;
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
        const float sparse_u_min = u_two - 0.08f * raw_u_span;
        const float sparse_u_max = u_ninety_eight + 0.08f * raw_u_span;
        const float sparse_v_min = v_two - 0.08f * raw_v_span;
        const float sparse_v_max = v_ninety_eight + 0.08f * raw_v_span;

        // The balanced/RANSAC samples above remain the baseline extent. On
        // the first group only, use all full-resolution point-head pixels as
        // a second extent source, but keep only real near-plane samples that
        // pass the existing neighborhood consistency gate. This determines
        // representable atlas extent only; it does not create floor cells.
        std::vector<FloorUVSample> full_extent_floor_uv;
        for (int view_index = 0; view_index < 3; ++view_index) {
            const GroupWorldView& view = views[static_cast<std::size_t>(view_index)];
            for (int y = 0; y < view.world_points.rows; ++y) {
                for (int x = 0; x < view.world_points.cols; ++x) {
                    const cv::Vec3f point = view.world_points.at<cv::Vec3f>(y, x);
                    const float confidence = view.world_confidence.at<float>(y, x);
                    if (!finite_vec(point) || !std::isfinite(confidence)) {
                        continue;
                    }
                    const cv::Vec3f delta = point - inlier_center;
                    const float height = plane.normal.dot(delta);
                    if (!std::isfinite(height)
                        || std::abs(height) > 2.5f * floor_band) {
                        continue;
                    }
                    if (!floor_neighborhood_consistent(
                        view.world_points, view.world_confidence, x, y,
                        inlier_center, plane.normal, floor_band)) {
                        continue;
                    }
                    full_extent_floor_uv.push_back(FloorUVSample{
                        axis_x.dot(delta), axis_y.dot(delta)});
                }
            }
        }

        float full_u_min = sparse_u_min;
        float full_u_max = sparse_u_max;
        float full_v_min = sparse_v_min;
        float full_v_max = sparse_v_max;
        std::size_t full_robust_support_count = 0U;
        if (full_extent_floor_uv.size() >= 3U) {
            std::vector<float> full_u;
            std::vector<float> full_v;
            full_u.reserve(full_extent_floor_uv.size());
            full_v.reserve(full_extent_floor_uv.size());
            for (const FloorUVSample& sample : full_extent_floor_uv) {
                full_u.push_back(sample.u);
                full_v.push_back(sample.v);
            }
            const float full_center_u = median_value(full_u);
            const float full_center_v = median_value(full_v);
            std::vector<float> full_radii;
            full_radii.reserve(full_extent_floor_uv.size());
            for (const FloorUVSample& sample : full_extent_floor_uv) {
                full_radii.push_back(std::hypot(
                    sample.u - full_center_u, sample.v - full_center_v));
            }
            const float full_radius_median = median_value(full_radii);
            std::vector<float> full_radius_deviations;
            full_radius_deviations.reserve(full_radii.size());
            for (const float radius : full_radii) {
                full_radius_deviations.push_back(std::abs(
                    radius - full_radius_median));
            }
            const float full_radius_mad = median_value(full_radius_deviations);
            const float full_radius_limit = full_radius_median
                + 4.0f * 1.4826f
                * std::max(full_radius_mad, kNumericEpsilon);
            std::vector<float> full_robust_u;
            std::vector<float> full_robust_v;
            full_robust_u.reserve(full_extent_floor_uv.size());
            full_robust_v.reserve(full_extent_floor_uv.size());
            for (std::size_t index = 0; index < full_extent_floor_uv.size(); ++index) {
                if (full_radii[index] <= full_radius_limit) {
                    full_robust_u.push_back(full_extent_floor_uv[index].u);
                    full_robust_v.push_back(full_extent_floor_uv[index].v);
                }
            }
            full_robust_support_count = full_robust_u.size();
            if (full_robust_u.size() >= 3U && full_robust_v.size() >= 3U) {
                const float full_u_two = percentile_value(full_robust_u, 0.02f);
                const float full_u_ninety_eight =
                    percentile_value(full_robust_u, 0.98f);
                const float full_v_two = percentile_value(full_robust_v, 0.02f);
                const float full_v_ninety_eight =
                    percentile_value(full_robust_v, 0.98f);
                const float full_raw_u_span = std::max(
                    full_u_ninety_eight - full_u_two, 0.01f * scene_scale);
                const float full_raw_v_span = std::max(
                    full_v_ninety_eight - full_v_two, 0.01f * scene_scale);
                full_u_min = full_u_two - 0.08f * full_raw_u_span;
                full_u_max = full_u_ninety_eight + 0.08f * full_raw_u_span;
                full_v_min = full_v_two - 0.08f * full_raw_v_span;
                full_v_max = full_v_ninety_eight + 0.08f * full_raw_v_span;
            }
        }

        // Only expand the baseline. A full-resolution distribution must not
        // remove any region that the sparse baseline already represented.
        const float u_min = std::min(sparse_u_min, full_u_min);
        const float u_max = std::max(sparse_u_max, full_u_max);
        const float v_min = std::min(sparse_v_min, full_v_min);
        const float v_max = std::max(sparse_v_max, full_v_max);
#if defined(_WIN32)
        const float sparse_u_span = sparse_u_max - sparse_u_min;
        const float sparse_v_span = sparse_v_max - sparse_v_min;
        const float final_u_span = u_max - u_min;
        const float final_v_span = v_max - v_min;
        const float extent_target_fill = 0.90f;
        const float extent_gui_scale = static_cast<float>(
            std::max(canvas_width_, canvas_height_));
        const float old_display_scale = std::max(
            sparse_u_span * extent_gui_scale
                / (extent_target_fill * static_cast<float>(canvas_width_)),
            sparse_v_span * extent_gui_scale
                / (extent_target_fill * static_cast<float>(canvas_height_)));
        std::clog << "[WINDOWS_ATLAS_EXTENT] sparse_support=" << robust_floor_u.size()
                  << " full_extent_support=" << full_extent_floor_uv.size()
                  << " full_robust_support=" << full_robust_support_count
                  << " sparse_bounds=[" << sparse_u_min << "," << sparse_u_max
                  << ";" << sparse_v_min << "," << sparse_v_max << "]"
                  << " full_bounds=[" << full_u_min << "," << full_u_max
                  << ";" << full_v_min << "," << full_v_max << "]"
                  << " final_bounds=[" << u_min << "," << u_max
                  << ";" << v_min << "," << v_max << "]"
                  << " old_span=[" << sparse_u_span << "," << sparse_v_span << "]"
                  << " new_span=[" << final_u_span << "," << final_v_span << "]"
                  << " old_display_scale=" << old_display_scale << std::endl;
#endif

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
        // Fit the expanded atlas independently to the physical canvas axes.
        // world_to_cell() uses max(width,height) as its GUI scale for both
        // axes, so a square-canvas scale would under-capacity the shorter
        // physical axis on the actual 770x630 canvas.
        const float target_fill = 0.90f;
        const float initialization_gui_scale = static_cast<float>(
            std::max(canvas_width_, canvas_height_));
        const float u_span = u_max_ - u_min_;
        const float v_span = v_max_ - v_min_;
        const float required_scale_x =
            u_span * initialization_gui_scale
            / (target_fill * static_cast<float>(canvas_width_));
        const float required_scale_y =
            v_span * initialization_gui_scale
            / (target_fill * static_cast<float>(canvas_height_));
        display_scale_ = std::max(required_scale_x, required_scale_y);
#if defined(_WIN32)
        if (accepted_fuse_count_ == 0U) {
            const float old_display_scale = std::max(u_span, v_span) / target_fill;
            const float old_x_capacity = old_display_scale
                * static_cast<float>(canvas_width_) / initialization_gui_scale;
            const float old_y_capacity = old_display_scale
                * static_cast<float>(canvas_height_) / initialization_gui_scale;
            const float new_x_capacity = display_scale_
                * static_cast<float>(canvas_width_) / initialization_gui_scale;
            const float new_y_capacity = display_scale_
                * static_cast<float>(canvas_height_) / initialization_gui_scale;
            std::clog << "[WINDOWS_ATLAS_MAPPING] canvas=" << canvas_width_ << "x"
                      << canvas_height_ << " gui_scale=" << initialization_gui_scale
                      << " u_span=" << u_span << " v_span=" << v_span
                      << " old_display_scale=" << old_display_scale
                      << " old_x_capacity=" << old_x_capacity
                      << " old_y_capacity=" << old_y_capacity
                      << " required_scale_x=" << required_scale_x
                      << " required_scale_y=" << required_scale_y
                      << " new_display_scale=" << display_scale_
                      << " new_x_capacity=" << new_x_capacity
                      << " new_y_capacity=" << new_y_capacity << std::endl;
        }
#endif
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

    std::array<cv::Vec3f, 3> reference_camera_centers{};
    for (int view_index = 0; view_index < 3; ++view_index) {
        reference_camera_centers[static_cast<std::size_t>(view_index)] = apply_sim3(
            transforms[static_cast<std::size_t>(view_index)],
            current_centers[static_cast<std::size_t>(view_index)]);
        if (!finite_vec(reference_camera_centers[static_cast<std::size_t>(view_index)])) {
            return reject("reference-frame camera center is invalid");
        }
    }

    result.scene_scale = scene_scale_;
    result.floor_band = floor_band_;
    const std::size_t cell_count = static_cast<std::size_t>(logical_width_)
        * static_cast<std::size_t>(logical_height_);
    std::array<std::vector<float>, 3> confidence_values_by_view;
    for (int view_index = 0; view_index < 3; ++view_index) {
        const GroupWorldView& view = views[static_cast<std::size_t>(view_index)];
        for (int y = 0; y < view.world_confidence.rows; ++y) {
            for (int x = 0; x < view.world_confidence.cols; ++x) {
                const float confidence = view.world_confidence.at<float>(y, x);
                if (std::isfinite(confidence)) {
                    confidence_values_by_view[static_cast<std::size_t>(view_index)].push_back(
                        confidence);
                }
            }
        }
    }
    const std::size_t finite_confidence_count = confidence_values_by_view[0].size()
        + confidence_values_by_view[1].size()
        + confidence_values_by_view[2].size();
    if (finite_confidence_count == 0U) {
        return reject("group world-point confidence map is empty");
    }
    std::array<ConfidenceStats, 3> confidence_stats{};
    for (int view_index = 0; view_index < 3; ++view_index) {
        const auto& values = confidence_values_by_view[static_cast<std::size_t>(view_index)];
        if (!values.empty()) {
            confidence_stats[static_cast<std::size_t>(view_index)] = ConfidenceStats{
                percentile_value(values, 0.10f),
                percentile_value(values, 0.50f),
                percentile_value(values, 0.90f)};
        }
    }

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
    const auto physical_uv_to_cell = [&](const float u, const float v, int& logical_x, int& logical_y) {
        if (!std::isfinite(u) || !std::isfinite(v)) {
            return false;
        }
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
    const auto world_to_cell = [&](const float u, const float v, int& logical_x, int& logical_y) {
        if (!std::isfinite(u) || !std::isfinite(v)
            || u < u_min_ || u > u_max_ || v < v_min_ || v > v_max_) {
            return false;
        }
        // This is the exact GUI point_from_slot() convention translated back
        // to a physical pixel. The logical cell is the even-origin 2x2 block
        // so all four layers share one GUI-derived XY neighborhood.
        return physical_uv_to_cell(u, v, logical_x, logical_y);
    };
#if defined(_WIN32)
    // Diagnostic-only mirror of world_to_cell(): retain the production
    // return contract while identifying whether a reject came from the
    // atlas UV bounds or from a physical canvas edge.
    const auto world_to_cell_reject_reason = [&](const float u, const float v) {
        if (!std::isfinite(u) || !std::isfinite(v)
            || u < u_min_ || u > u_max_ || v < v_min_ || v > v_max_) {
            return 1; // UV bounds
        }
        const float physical_x = canvas_width_ * 0.5f
            + (u - center_u_) / display_scale_ * gui_scale;
        const float physical_y = canvas_height_ * 0.5f
            - (v - center_v_) / display_scale_ * gui_scale;
        const int rounded_x = static_cast<int>(std::lround(physical_x));
        const int rounded_y = static_cast<int>(std::lround(physical_y));
        if (rounded_x < 0 || rounded_x >= canvas_width_) {
            return 2; // physical X
        }
        if (rounded_y < 0 || rounded_y >= canvas_height_) {
            return 3; // physical Y
        }
        const int mapped_x = rounded_x / 2;
        const int mapped_y = rounded_y / 2;
        if (mapped_x < 0 || mapped_x >= logical_width_
            || mapped_y < 0 || mapped_y >= logical_height_) {
            return 3; // logical Y/canvas quantization edge
        }
        return 0;
    };
    const auto world_to_cell_uv_reject_flags = [&](const float u, const float v) {
        int flags = 0;
        if (std::isfinite(u)) {
            if (u < u_min_) {
                flags |= 1; // u_low
            }
            if (u > u_max_) {
                flags |= 2; // u_high
            }
        }
        if (std::isfinite(v)) {
            if (v < v_min_) {
                flags |= 4; // v_low
            }
            if (v > v_max_) {
                flags |= 8; // v_high
            }
        }
        return flags;
    };
#endif
    const auto cell_to_reference_floor_point = [&](const int logical_x, const int logical_y) {
        const float physical_x = static_cast<float>(logical_x * 2) + 0.5f;
        const float physical_y = static_cast<float>(logical_y * 2) + 0.5f;
        const float u = center_u_
            + (physical_x - 0.5f * static_cast<float>(canvas_width_))
                / gui_scale * display_scale_;
        const float v = center_v_
            - (physical_y - 0.5f * static_cast<float>(canvas_height_))
                / gui_scale * display_scale_;
        return plane_origin_ + axis_x_ * u + axis_y_ * v;
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
    std::vector<FloorSample> near_floor_best(cell_count * 3U);
    std::array<std::vector<std::vector<ObjectSample>>, 3> object_samples_by_view;
    for (auto& samples : object_samples_by_view) {
        samples.resize(cell_count);
    }
#if defined(_WIN32)
    std::size_t raw_strict_plane_band_total = 0U;
    std::size_t raw_strict_uv_reject = 0U;
    std::size_t raw_strict_pixel_x_reject = 0U;
    std::size_t raw_strict_pixel_y_reject = 0U;
    std::array<std::size_t, 4> raw_strict_uv_directional{};
    std::size_t raw_near_plane_total = 0U;
    std::size_t raw_near_uv_reject = 0U;
    std::size_t raw_near_pixel_x_reject = 0U;
    std::size_t raw_near_pixel_y_reject = 0U;
    std::array<std::size_t, 4> raw_near_uv_directional{};
    std::size_t raw_strict_neighborhood_rejected = 0U;
    std::size_t raw_near_neighborhood_rejected = 0U;
    std::array<std::size_t, 3> reliable_strict_uv_reject_by_view{};
    std::array<std::size_t, 3> reliable_near_uv_reject_by_view{};
    std::array<std::size_t, 3> reliable_object_uv_reject_by_view{};
    std::array<std::size_t, 3> reliable_strict_representable_margin_by_view{};
    std::array<std::size_t, 3> reliable_strict_physical_outside_by_view{};
    std::array<std::size_t, 3> reliable_near_representable_margin_by_view{};
    std::array<std::size_t, 3> reliable_near_physical_outside_by_view{};
    std::array<std::size_t, 3> reliable_object_representable_margin_by_view{};
    std::array<std::size_t, 3> reliable_object_physical_outside_by_view{};
    std::array<std::vector<std::uint8_t>, 3> direct_margin_strict_floor_by_view;
    std::array<std::vector<std::uint8_t>, 3> direct_margin_near_floor_by_view;
    std::array<std::vector<std::uint8_t>, 3> direct_margin_object_by_view;
    for (int view_index = 0; view_index < 3; ++view_index) {
        direct_margin_strict_floor_by_view[static_cast<std::size_t>(view_index)].assign(
            cell_count, 0U);
        direct_margin_near_floor_by_view[static_cast<std::size_t>(view_index)].assign(
            cell_count, 0U);
        direct_margin_object_by_view[static_cast<std::size_t>(view_index)].assign(
            cell_count, 0U);
    }
#endif
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
                const float abs_height = std::abs(height);
                const bool strict_plane_band = abs_height <= floor_band_;
                const bool near_plane_band = abs_height <= 2.5f * floor_band_;
#if defined(_WIN32)
                if (strict_plane_band) {
                    ++raw_strict_plane_band_total;
                } else if (near_plane_band) {
                    ++raw_near_plane_total;
                }
#endif
                int logical_x = 0;
                int logical_y = 0;
                int direct_margin_x = 0;
                int direct_margin_y = 0;
                int direct_margin_kind = 0;
                if (!world_to_cell(u, v, logical_x, logical_y)) {
                    const bool uv_bounds_reject = std::isfinite(u) && std::isfinite(v)
                        && (u < u_min_ || u > u_max_ || v < v_min_ || v > v_max_);
                    bool strict_neighborhood_ok = false;
                    bool near_neighborhood_ok = false;
                    bool object_continuity_ok = false;
                    if (uv_bounds_reject) {
                        if (strict_plane_band) {
                            strict_neighborhood_ok = floor_neighborhood_consistent(
                                aligned_points, view.world_confidence, x, y,
                                plane_origin_, plane_normal_, floor_band_);
                        } else if (near_plane_band) {
                            near_neighborhood_ok = floor_neighborhood_consistent(
                                aligned_points, view.world_confidence, x, y,
                                plane_origin_, plane_normal_, floor_band_);
                        } else if (height > 1.5f * floor_band_
                            && height < max_object_height_) {
                            object_continuity_ok = local_object_continuity(
                                aligned_points, view.world_confidence, x, y, point, scene_scale_);
                        }
                        if (physical_uv_to_cell(u, v, direct_margin_x, direct_margin_y)) {
                            if (strict_neighborhood_ok) {
                                direct_margin_kind = 1;
                            } else if (near_neighborhood_ok) {
                                direct_margin_kind = 2;
                            } else if (object_continuity_ok) {
                                direct_margin_kind = 3;
                            }
                        }
                    }
#if defined(_WIN32)
                    const int reject_reason = world_to_cell_reject_reason(u, v);
                    if (strict_plane_band) {
                        if (strict_neighborhood_ok && reject_reason == 1) {
                            ++reliable_strict_uv_reject_by_view[
                                static_cast<std::size_t>(view_index)];
                            if (physical_uv_to_cell(u, v, direct_margin_x, direct_margin_y)) {
                                ++reliable_strict_representable_margin_by_view[
                                    static_cast<std::size_t>(view_index)];
                            } else {
                                ++reliable_strict_physical_outside_by_view[
                                    static_cast<std::size_t>(view_index)];
                            }
                        }
                        if (reject_reason == 1) {
                            ++raw_strict_uv_reject;
                            const int flags = world_to_cell_uv_reject_flags(u, v);
                            if ((flags & 1) != 0) {
                                ++raw_strict_uv_directional[0];
                            }
                            if ((flags & 2) != 0) {
                                ++raw_strict_uv_directional[1];
                            }
                            if ((flags & 4) != 0) {
                                ++raw_strict_uv_directional[2];
                            }
                            if ((flags & 8) != 0) {
                                ++raw_strict_uv_directional[3];
                            }
                        } else if (reject_reason == 2) {
                            ++raw_strict_pixel_x_reject;
                        } else if (reject_reason == 3) {
                            ++raw_strict_pixel_y_reject;
                        }
                    } else if (near_plane_band) {
                        if (near_neighborhood_ok && reject_reason == 1) {
                            ++reliable_near_uv_reject_by_view[
                                static_cast<std::size_t>(view_index)];
                            if (physical_uv_to_cell(u, v, direct_margin_x, direct_margin_y)) {
                                ++reliable_near_representable_margin_by_view[
                                    static_cast<std::size_t>(view_index)];
                            } else {
                                ++reliable_near_physical_outside_by_view[
                                    static_cast<std::size_t>(view_index)];
                            }
                        }
                        if (reject_reason == 1) {
                            ++raw_near_uv_reject;
                            const int flags = world_to_cell_uv_reject_flags(u, v);
                            if ((flags & 1) != 0) {
                                ++raw_near_uv_directional[0];
                            }
                            if ((flags & 2) != 0) {
                                ++raw_near_uv_directional[1];
                            }
                            if ((flags & 4) != 0) {
                                ++raw_near_uv_directional[2];
                            }
                            if ((flags & 8) != 0) {
                                ++raw_near_uv_directional[3];
                            }
                        } else if (reject_reason == 2) {
                            ++raw_near_pixel_x_reject;
                        } else if (reject_reason == 3) {
                            ++raw_near_pixel_y_reject;
                        }
                    } else if (height > 1.5f * floor_band_
                        && height < max_object_height_
                        && local_object_continuity(
                            aligned_points, view.world_confidence, x, y, point, scene_scale_)) {
                        if (object_continuity_ok && reject_reason == 1) {
                            ++reliable_object_uv_reject_by_view[
                                static_cast<std::size_t>(view_index)];
                            if (physical_uv_to_cell(u, v, direct_margin_x, direct_margin_y)) {
                                ++reliable_object_representable_margin_by_view[
                                    static_cast<std::size_t>(view_index)];
                            } else {
                                ++reliable_object_physical_outside_by_view[
                                    static_cast<std::size_t>(view_index)];
                            }
                        }
                    }
#endif
                    if (direct_margin_kind == 0) {
                        continue;
                    }
                    logical_x = direct_margin_x;
                    logical_y = direct_margin_y;
                }
                const std::size_t cell = static_cast<std::size_t>(logical_y)
                    * static_cast<std::size_t>(logical_width_)
                    + static_cast<std::size_t>(logical_x);
#if defined(_WIN32)
                if (direct_margin_kind == 1) {
                    direct_margin_strict_floor_by_view[static_cast<std::size_t>(view_index)][cell] = 1U;
                } else if (direct_margin_kind == 2) {
                    direct_margin_near_floor_by_view[static_cast<std::size_t>(view_index)][cell] = 1U;
                } else if (direct_margin_kind == 3) {
                    direct_margin_object_by_view[static_cast<std::size_t>(view_index)][cell] = 1U;
                }
#endif
                const ConfidenceStats& confidence_range =
                    confidence_stats[static_cast<std::size_t>(view_index)];
                const float normalized = normalized_confidence(
                    confidence, confidence_range.q10, confidence_range.q90);
                const float weight = normalized * border_weight(
                    x, y, view.world_points.cols, view.world_points.rows);
                if (abs_height <= floor_band_) {
                    if (direct_margin_kind != 1 && !floor_neighborhood_consistent(
                        aligned_points, view.world_confidence, x, y,
                        plane_origin_, plane_normal_, floor_band_)) {
#if defined(_WIN32)
                        ++raw_strict_neighborhood_rejected;
#endif
                        continue;
                    }
                    FloorSample& best = floor_best[
                        static_cast<std::size_t>(view_index) * cell_count + cell];
                    if (!best.valid || weight > best.weight) {
                        best.color = view.rgb.at<cv::Vec3f>(y, x);
                        best.normalized_confidence = normalized;
                        best.weight = weight;
                        best.height = height;
                        best.valid = true;
                    }
                } else if (abs_height <= 2.5f * floor_band_) {
                    if (direct_margin_kind != 2 && !floor_neighborhood_consistent(
                        aligned_points, view.world_confidence, x, y,
                        plane_origin_, plane_normal_, floor_band_)) {
#if defined(_WIN32)
                        ++raw_near_neighborhood_rejected;
#endif
                        continue;
                    }
                    FloorSample& best = near_floor_best[
                        static_cast<std::size_t>(view_index) * cell_count + cell];
                    if (!best.valid || weight > best.weight) {
                        best.color = view.rgb.at<cv::Vec3f>(y, x);
                        best.normalized_confidence = normalized;
                        best.weight = weight;
                        best.height = height;
                        best.valid = true;
                    }
                } else if (height > 1.5f * floor_band_
                    && height < max_object_height_
                    ) {
                    if (direct_margin_kind == 3 || local_object_continuity(
                        aligned_points, view.world_confidence, x, y, point, scene_scale_)) {
                        object_samples_by_view[static_cast<std::size_t>(view_index)][cell].push_back(
                            ObjectSample{
                            u,
                            v,
                            height,
                            normalized,
                            std::max(weight, 1e-4f),
                            view.rgb.at<cv::Vec3f>(y, x),
                            view_index,
                            direct_margin_kind == 3});
                    }
                }
                // Points outside the strict/near-floor bands and positive
                // object band are intentionally ignored. Neither decision
                // uses RGB.
            }
        }
    }

#if defined(_WIN32)
    if (accepted_fuse_count_ == 0U) {
        std::clog << "[WINDOWS_FLOOR_EDGE_DIAG] strict_plane_band_total="
                  << raw_strict_plane_band_total
                  << " strict_uv_reject=" << raw_strict_uv_reject
                  << " strict_uv_u_low=" << raw_strict_uv_directional[0]
                  << " strict_uv_u_high=" << raw_strict_uv_directional[1]
                  << " strict_uv_v_low=" << raw_strict_uv_directional[2]
                  << " strict_uv_v_high=" << raw_strict_uv_directional[3]
                  << " strict_pixel_x_reject=" << raw_strict_pixel_x_reject
                  << " strict_pixel_y_reject=" << raw_strict_pixel_y_reject
                  << " near_plane_total=" << raw_near_plane_total
                  << " near_uv_reject=" << raw_near_uv_reject
                  << " near_uv_u_low=" << raw_near_uv_directional[0]
                  << " near_uv_u_high=" << raw_near_uv_directional[1]
                  << " near_uv_v_low=" << raw_near_uv_directional[2]
                  << " near_uv_v_high=" << raw_near_uv_directional[3]
                  << " near_pixel_x_reject=" << raw_near_pixel_x_reject
                  << " near_pixel_y_reject=" << raw_near_pixel_y_reject
                  << " strict_neighborhood_rejected="
                  << raw_strict_neighborhood_rejected
                  << " near_neighborhood_rejected="
                  << raw_near_neighborhood_rejected << std::endl;
        std::clog << "[WINDOWS_DIRECT_UV] reliable_strict_view0="
                  << reliable_strict_uv_reject_by_view[0]
                  << " reliable_strict_view1="
                  << reliable_strict_uv_reject_by_view[1]
                  << " reliable_strict_view2="
                  << reliable_strict_uv_reject_by_view[2]
                  << " reliable_near_view0="
                  << reliable_near_uv_reject_by_view[0]
                  << " reliable_near_view1="
                  << reliable_near_uv_reject_by_view[1]
                  << " reliable_near_view2="
                  << reliable_near_uv_reject_by_view[2]
                  << " reliable_object_view0="
                  << reliable_object_uv_reject_by_view[0]
                  << " reliable_object_view1="
                  << reliable_object_uv_reject_by_view[1]
                  << " reliable_object_view2="
                  << reliable_object_uv_reject_by_view[2]
                  << " strict_representable_margin_view0="
                  << reliable_strict_representable_margin_by_view[0]
                  << " strict_representable_margin_view1="
                  << reliable_strict_representable_margin_by_view[1]
                  << " strict_representable_margin_view2="
                  << reliable_strict_representable_margin_by_view[2]
                  << " strict_physical_outside_view0="
                  << reliable_strict_physical_outside_by_view[0]
                  << " strict_physical_outside_view1="
                  << reliable_strict_physical_outside_by_view[1]
                  << " strict_physical_outside_view2="
                  << reliable_strict_physical_outside_by_view[2]
                  << " near_representable_margin_view0="
                  << reliable_near_representable_margin_by_view[0]
                  << " near_representable_margin_view1="
                  << reliable_near_representable_margin_by_view[1]
                  << " near_representable_margin_view2="
                  << reliable_near_representable_margin_by_view[2]
                  << " near_physical_outside_view0="
                  << reliable_near_physical_outside_by_view[0]
                  << " near_physical_outside_view1="
                  << reliable_near_physical_outside_by_view[1]
                  << " near_physical_outside_view2="
                  << reliable_near_physical_outside_by_view[2]
                  << " object_representable_margin_view0="
                  << reliable_object_representable_margin_by_view[0]
                  << " object_representable_margin_view1="
                  << reliable_object_representable_margin_by_view[1]
                  << " object_representable_margin_view2="
                  << reliable_object_representable_margin_by_view[2]
                  << " object_physical_outside_view0="
                  << reliable_object_physical_outside_by_view[0]
                  << " object_physical_outside_view1="
                  << reliable_object_physical_outside_by_view[1]
                  << " object_physical_outside_view2="
                  << reliable_object_physical_outside_by_view[2] << std::endl;
    }
#endif

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

    // A near-floor point is not allowed to become floor by itself. It can
    // recover a persistent floor cell only when another view observes the
    // same nearby logical XY at a compatible height.
    const float floor_recovery_height_tolerance = 2.0f * floor_band_;
    std::array<std::vector<FloorObservation>, 3> floor_observations_by_view;
    std::array<std::vector<int>, 3> floor_observation_index_by_view;
    for (int view_index = 0; view_index < 3; ++view_index) {
        floor_observation_index_by_view[static_cast<std::size_t>(view_index)].assign(
            cell_count, -1);
        for (std::size_t cell = 0; cell < cell_count; ++cell) {
            const FloorSample& strict_sample = floor_best[
                static_cast<std::size_t>(view_index) * cell_count + cell];
            const FloorSample& near_sample = near_floor_best[
                static_cast<std::size_t>(view_index) * cell_count + cell];
            const FloorSample* sample = strict_sample.valid
                ? &strict_sample
                : (near_sample.valid ? &near_sample : nullptr);
            if (sample == nullptr) {
                continue;
            }
            const int logical_x = static_cast<int>(
                cell % static_cast<std::size_t>(logical_width_));
            const int logical_y = static_cast<int>(
                cell / static_cast<std::size_t>(logical_width_));
            FloorObservation observation;
            observation.valid = true;
            observation.near_floor = !strict_sample.valid;
            observation.view_id = view_index;
            observation.logical_x = logical_x;
            observation.logical_y = logical_y;
            observation.height = sample->height;
            observation.sample = *sample;
            auto& observations = floor_observations_by_view[
                static_cast<std::size_t>(view_index)];
            floor_observation_index_by_view[static_cast<std::size_t>(view_index)][cell] =
                static_cast<int>(observations.size());
            observations.push_back(std::move(observation));
        }
    }

    std::vector<FloorRecoveryCandidate> floor_recovery_best(cell_count);
    const auto consider_floor_recovery = [&](FloorRecoveryCandidate candidate) {
        if (!candidate.valid || candidate.logical_x < 0 || candidate.logical_x >= logical_width_
            || candidate.logical_y < 0 || candidate.logical_y >= logical_height_
            || support_count(candidate.view_mask) < 2) {
            return;
        }
        const std::size_t cell = static_cast<std::size_t>(candidate.logical_y)
            * static_cast<std::size_t>(logical_width_) + static_cast<std::size_t>(candidate.logical_x);
        FloorRecoveryCandidate& current = floor_recovery_best[cell];
        if (!current.valid) {
            current = std::move(candidate);
            return;
        }
        if (std::abs(current.height - candidate.height)
            <= floor_recovery_height_tolerance) {
            FloorRecoveryCandidate merged = current;
            for (int view = 0; view < 3; ++view) {
                const std::uint8_t bit = static_cast<std::uint8_t>(1U) << view;
                if ((candidate.view_mask & bit) == 0U) {
                    continue;
                }
                if ((merged.view_mask & bit) == 0U
                    || candidate.observations[static_cast<std::size_t>(view)].sample.weight
                        > merged.observations[static_cast<std::size_t>(view)].sample.weight) {
                    merged.observations[static_cast<std::size_t>(view)] =
                        candidate.observations[static_cast<std::size_t>(view)];
                    merged.view_mask |= bit;
                }
            }
            finalize_floor_candidate(merged);
            const bool better = support_count(merged.view_mask) > support_count(current.view_mask)
                || (support_count(merged.view_mask) == support_count(current.view_mask)
                    && merged.score > current.score);
            if (better) {
                current = std::move(merged);
            }
        } else if (candidate.score > current.score) {
            current = std::move(candidate);
        }
    };
    for (int first_view = 0; first_view < 3; ++first_view) {
        for (int second_view = first_view + 1; second_view < 3; ++second_view) {
            const auto& first_observations = floor_observations_by_view[
                static_cast<std::size_t>(first_view)];
            const auto& second_observations = floor_observations_by_view[
                static_cast<std::size_t>(second_view)];
            const auto& second_index = floor_observation_index_by_view[
                static_cast<std::size_t>(second_view)];
            for (const FloorObservation& first : first_observations) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int second_x = first.logical_x + dx;
                        const int second_y = first.logical_y + dy;
                        if (second_x < 0 || second_x >= logical_width_
                            || second_y < 0 || second_y >= logical_height_) {
                            continue;
                        }
                        const std::size_t second_cell = static_cast<std::size_t>(second_y)
                            * static_cast<std::size_t>(logical_width_)
                            + static_cast<std::size_t>(second_x);
                        const int second_index_value = second_index[second_cell];
                        if (second_index_value < 0) {
                            continue;
                        }
                        const FloorObservation& second = second_observations[
                            static_cast<std::size_t>(second_index_value)];
                        if (!first.near_floor && !second.near_floor) {
                            continue;
                        }
                        if (std::abs(first.height - second.height)
                            > floor_recovery_height_tolerance) {
                            continue;
                        }
                        FloorRecoveryCandidate candidate;
                        candidate.valid = true;
                        candidate.logical_x = consensus_coordinate({
                            first.logical_x, second.logical_x});
                        candidate.logical_y = consensus_coordinate({
                            first.logical_y, second.logical_y});
                        candidate.view_mask = static_cast<std::uint8_t>(
                            (static_cast<std::uint8_t>(1U) << first_view)
                            | (static_cast<std::uint8_t>(1U) << second_view));
                        candidate.observations[static_cast<std::size_t>(first_view)] = first;
                        candidate.observations[static_cast<std::size_t>(second_view)] = second;
                        finalize_floor_candidate(candidate);
                        consider_floor_recovery(std::move(candidate));
                    }
                }
            }
        }
    }

    // Update persistent floor cells. A cell not seen in this group is left
    // untouched, so a temporary arm occlusion cannot erase static ground.
    // Keep a separate current-group observation mask: a completion cell may
    // persist, but it must not become geometry evidence for the next F3 ring.
    std::vector<std::uint8_t> current_observed_floor_mask(cell_count, 0U);
    for (std::size_t cell = 0; cell < cell_count; ++cell) {
        std::array<const FloorSample*, 3> contributors{};
        bool has_strict_floor = false;
        for (int view_index = 0; view_index < 3; ++view_index) {
            const FloorSample& sample = floor_best[
                static_cast<std::size_t>(view_index) * cell_count + cell];
            if (sample.valid) {
                contributors[static_cast<std::size_t>(view_index)] = &sample;
                has_strict_floor = true;
            }
        }
        if (!has_strict_floor && floor_recovery_best[cell].valid) {
            const FloorRecoveryCandidate& recovery = floor_recovery_best[cell];
            for (int view_index = 0; view_index < 3; ++view_index) {
                const std::uint8_t bit = static_cast<std::uint8_t>(1U) << view_index;
                if ((recovery.view_mask & bit) != 0U) {
                    contributors[static_cast<std::size_t>(view_index)] =
                        &recovery.observations[static_cast<std::size_t>(view_index)].sample;
                }
            }
        }

        float color_weight = 0.0f;
        float total_weight = 0.0f;
        float confidence_weight = 0.0f;
        cv::Vec3f color(0.0f, 0.0f, 0.0f);
        for (int view_index = 0; view_index < 3; ++view_index) {
            const FloorSample* sample = contributors[static_cast<std::size_t>(view_index)];
            if (sample == nullptr || !std::isfinite(sample->weight) || sample->weight <= 0.0f) {
                continue;
            }
            const auto& gain = color_gain_[static_cast<std::size_t>(view_index)];
            const cv::Vec3f corrected(
                std::clamp(sample->color[0] * gain[0], 0.0f, 1.0f),
                std::clamp(sample->color[1] * gain[1], 0.0f, 1.0f),
                std::clamp(sample->color[2] * gain[2], 0.0f, 1.0f));
            // Exact black model padding has no valid color sample. It is
            // skipped here without affecting the geometric floor decision;
            // genuinely dark object geometry remains eligible above.
            const bool black_padding = std::abs(sample->color[0]) <= kNumericEpsilon
                && std::abs(sample->color[1]) <= kNumericEpsilon
                && std::abs(sample->color[2]) <= kNumericEpsilon;
            if (!black_padding && finite_vec(corrected)) {
                color += corrected * sample->weight;
                color_weight += sample->weight;
            }
            total_weight += sample->weight;
            confidence_weight += sample->normalized_confidence * sample->weight;
        }
        if (color_weight > kNumericEpsilon) {
            color *= 1.0f / color_weight;
        }
        if (confidence_weight > kNumericEpsilon && total_weight > kNumericEpsilon) {
            confidence_weight = std::clamp(
                confidence_weight / total_weight, 0.0f, 1.0f);
        }
        bool has_floor_observation = false;
        for (const FloorSample* contributor : contributors) {
            if (contributor != nullptr && contributor->valid) {
                has_floor_observation = true;
                break;
            }
        }
        if (has_floor_observation) {
            // A floor cell's XY is fixed by the first reference atlas. Only
            // its observed color/confidence may change from frame to frame.
            const int logical_x = static_cast<int>(
                cell % static_cast<std::size_t>(logical_width_));
            const int logical_y = static_cast<int>(
                cell / static_cast<std::size_t>(logical_width_));
            current_observed_floor_mask[cell] = 1U;
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
    // Keep each view's direct strict/near-floor visibility separate from the
    // current observed-floor result.  The latter remains governed by the
    // frozen strict/two-view policy in this diagnostic pass.
    std::array<std::vector<std::uint8_t>, 3> direct_floor_presence_by_view;
    for (int view_index = 0; view_index < 3; ++view_index) {
        auto& direct_presence = direct_floor_presence_by_view[
            static_cast<std::size_t>(view_index)];
        direct_presence.assign(cell_count, 0U);
        for (std::size_t cell = 0; cell < cell_count; ++cell) {
            const FloorSample& strict_sample = floor_best[
                static_cast<std::size_t>(view_index) * cell_count + cell];
            const FloorSample& near_sample = near_floor_best[
                static_cast<std::size_t>(view_index) * cell_count + cell];
            if (strict_sample.valid || near_sample.valid) {
                direct_presence[cell] = 1U;
            }
        }
    }
#if defined(_WIN32)
    const auto resize_logical_mask = [&](const cv::Mat& mask) {
        cv::Mat resized;
        cv::resize(mask, resized, cv::Size(canvas_width_, canvas_height_),
            0.0, 0.0, cv::INTER_NEAREST);
        return resized;
    };
    const auto make_view_mask_visual = [&](const std::array<cv::Mat, 3>& masks) {
        cv::Mat visual(logical_height_, logical_width_, CV_8UC3,
            cv::Scalar(0, 0, 0));
        for (int y = 0; y < logical_height_; ++y) {
            for (int x = 0; x < logical_width_; ++x) {
                cv::Vec3b& pixel = visual.at<cv::Vec3b>(y, x);
                if (masks[0].at<std::uint8_t>(y, x) != 0U) {
                    pixel[2] = 255U; // view0 = red
                }
                if (masks[1].at<std::uint8_t>(y, x) != 0U) {
                    pixel[1] = 255U; // view1 = green
                }
                if (masks[2].at<std::uint8_t>(y, x) != 0U) {
                    pixel[0] = 255U; // view2 = blue
                }
            }
        }
        return resize_logical_mask(visual);
    };
    if (accepted_fuse_count_ == 0U) {
        std::array<cv::Mat, 3> direct_margin_floor_masks;
        std::array<cv::Mat, 3> direct_margin_object_masks;
        std::array<std::size_t, 3> direct_margin_strict_counts{};
        std::array<std::size_t, 3> direct_margin_near_counts{};
        std::array<std::size_t, 3> direct_margin_object_counts{};
        for (int view_index = 0; view_index < 3; ++view_index) {
            const std::size_t view = static_cast<std::size_t>(view_index);
            direct_margin_strict_counts[view] = static_cast<std::size_t>(std::count(
                direct_margin_strict_floor_by_view[view].begin(),
                direct_margin_strict_floor_by_view[view].end(),
                static_cast<std::uint8_t>(1U)));
            direct_margin_near_counts[view] = static_cast<std::size_t>(std::count(
                direct_margin_near_floor_by_view[view].begin(),
                direct_margin_near_floor_by_view[view].end(),
                static_cast<std::uint8_t>(1U)));
            direct_margin_object_counts[view] = static_cast<std::size_t>(std::count(
                direct_margin_object_by_view[view].begin(),
                direct_margin_object_by_view[view].end(),
                static_cast<std::uint8_t>(1U)));
            const cv::Mat strict_mask(logical_height_, logical_width_, CV_8UC1,
                direct_margin_strict_floor_by_view[view].data());
            const cv::Mat near_mask(logical_height_, logical_width_, CV_8UC1,
                direct_margin_near_floor_by_view[view].data());
            cv::bitwise_or(strict_mask, near_mask, direct_margin_floor_masks[view]);
            direct_margin_object_masks[view] = cv::Mat(
                logical_height_, logical_width_, CV_8UC1,
                direct_margin_object_by_view[view].data()).clone();
        }
        try {
            if (!cv::imwrite("tmp/direct_margin_floor.png",
                    make_view_mask_visual(direct_margin_floor_masks))
                || !cv::imwrite("tmp/direct_margin_object.png",
                    make_view_mask_visual(direct_margin_object_masks))) {
                std::clog << "[WINDOWS_DIRECT_MARGIN_MASK] write_failed=1" << std::endl;
            }
        } catch (const cv::Exception&) {
            std::clog << "[WINDOWS_DIRECT_MARGIN_MASK] write_failed=1" << std::endl;
        }
        std::clog << "[WINDOWS_DIRECT_MARGIN] strict_view0="
                  << direct_margin_strict_counts[0]
                  << " strict_view1=" << direct_margin_strict_counts[1]
                  << " strict_view2=" << direct_margin_strict_counts[2]
                  << " near_view0=" << direct_margin_near_counts[0]
                  << " near_view1=" << direct_margin_near_counts[1]
                  << " near_view2=" << direct_margin_near_counts[2]
                  << " object_view0=" << direct_margin_object_counts[0]
                  << " object_view1=" << direct_margin_object_counts[1]
                  << " object_view2=" << direct_margin_object_counts[2]
                  << " strict_total=" << direct_margin_strict_counts[0]
                        + direct_margin_strict_counts[1]
                        + direct_margin_strict_counts[2]
                  << " near_total=" << direct_margin_near_counts[0]
                        + direct_margin_near_counts[1]
                        + direct_margin_near_counts[2]
                  << " object_total=" << direct_margin_object_counts[0]
                        + direct_margin_object_counts[1]
                        + direct_margin_object_counts[2]
                  << std::endl;
        std::array<cv::Mat, 3> direct_floor_masks;
        std::array<std::size_t, 3> strict_counts{};
        std::array<std::size_t, 3> near_counts{};
        std::array<std::size_t, 3> direct_counts{};
        std::array<std::size_t, 3> direct_only_counts{};
        std::array<std::size_t, 3> direct_only_kept_counts{};
        std::array<std::size_t, 3> direct_only_lost_counts{};
        std::array<std::size_t, 8> direct_pattern_counts{};
        cv::Mat direct_floor_union(logical_height_, logical_width_, CV_8UC1,
            cv::Scalar(0));
        cv::Mat lost_floor_direct_before(logical_height_, logical_width_, CV_8UC1,
            cv::Scalar(0));
        std::size_t direct_union_count = 0U;
        std::size_t lost_direct_count = 0U;
        for (int view_index = 0; view_index < 3; ++view_index) {
            direct_floor_masks[static_cast<std::size_t>(view_index)] = cv::Mat(
                logical_height_, logical_width_, CV_8UC1, cv::Scalar(0));
        }
        for (std::size_t cell = 0; cell < cell_count; ++cell) {
            const int x = static_cast<int>(cell % static_cast<std::size_t>(logical_width_));
            const int y = static_cast<int>(cell / static_cast<std::size_t>(logical_width_));
            std::uint8_t pattern = 0U;
            for (int view_index = 0; view_index < 3; ++view_index) {
                const FloorSample& strict_sample = floor_best[
                    static_cast<std::size_t>(view_index) * cell_count + cell];
                const FloorSample& near_sample = near_floor_best[
                    static_cast<std::size_t>(view_index) * cell_count + cell];
                if (strict_sample.valid) {
                    ++strict_counts[static_cast<std::size_t>(view_index)];
                }
                if (near_sample.valid) {
                    ++near_counts[static_cast<std::size_t>(view_index)];
                }
                if (direct_floor_presence_by_view[static_cast<std::size_t>(view_index)][cell]
                    != 0U) {
                    direct_floor_masks[static_cast<std::size_t>(view_index)].at<
                        std::uint8_t>(y, x) = 255U;
                    ++direct_counts[static_cast<std::size_t>(view_index)];
                    pattern |= static_cast<std::uint8_t>(1U) << view_index;
                }
            }
            if (pattern == 0U) {
                continue;
            }
            ++direct_pattern_counts[pattern];
            direct_floor_union.at<std::uint8_t>(y, x) = 255U;
            ++direct_union_count;
            if (support_count(pattern) == 1) {
                for (int view_index = 0; view_index < 3; ++view_index) {
                    if ((pattern & (static_cast<std::uint8_t>(1U) << view_index)) == 0U) {
                        continue;
                    }
                    ++direct_only_counts[static_cast<std::size_t>(view_index)];
                    if (current_observed_floor_mask[cell] != 0U) {
                        ++direct_only_kept_counts[static_cast<std::size_t>(view_index)];
                    } else {
                        ++direct_only_lost_counts[static_cast<std::size_t>(view_index)];
                    }
                }
            }
            if (current_observed_floor_mask[cell] == 0U) {
                lost_floor_direct_before.at<std::uint8_t>(y, x) = 255U;
                ++lost_direct_count;
            }
        }
        try {
            if (!cv::imwrite("tmp/direct_floor_union.png",
                    make_view_mask_visual(direct_floor_masks))
                || !cv::imwrite("tmp/lost_floor_direct_before.png",
                    resize_logical_mask(lost_floor_direct_before))) {
                std::clog << "[WINDOWS_DIRECT_FLOOR_MASK] write_failed=1" << std::endl;
            }
        } catch (const cv::Exception&) {
            std::clog << "[WINDOWS_DIRECT_FLOOR_MASK] write_failed=1" << std::endl;
        }
        std::clog << "[WINDOWS_DIRECT_FLOOR] strict_view0=" << strict_counts[0]
                  << " strict_view1=" << strict_counts[1]
                  << " strict_view2=" << strict_counts[2]
                  << " near_view0=" << near_counts[0]
                  << " near_view1=" << near_counts[1]
                  << " near_view2=" << near_counts[2]
                  << " direct_view0=" << direct_counts[0]
                  << " direct_view1=" << direct_counts[1]
                  << " direct_view2=" << direct_counts[2]
                  << " union=" << direct_union_count
                  << " only_view0=" << direct_only_counts[0]
                  << " only_view1=" << direct_only_counts[1]
                  << " only_view2=" << direct_only_counts[2]
                  << " only_view0_kept=" << direct_only_kept_counts[0]
                  << " only_view1_kept=" << direct_only_kept_counts[1]
                  << " only_view2_kept=" << direct_only_kept_counts[2]
                  << " only_view0_lost=" << direct_only_lost_counts[0]
                  << " only_view1_lost=" << direct_only_lost_counts[1]
                  << " only_view2_lost=" << direct_only_lost_counts[2]
                  << " lost_before_f4=" << lost_direct_count
                  << " pattern1=" << direct_pattern_counts[1]
                  << " pattern2=" << direct_pattern_counts[2]
                  << " pattern3=" << direct_pattern_counts[3]
                  << " pattern4=" << direct_pattern_counts[4]
                  << " pattern5=" << direct_pattern_counts[5]
                  << " pattern6=" << direct_pattern_counts[6]
                  << " pattern7=" << direct_pattern_counts[7] << std::endl;
    }
#endif
#if defined(_WIN32)
    if (accepted_fuse_count_ == 0U) {
        std::vector<std::pair<int, int>> roundtrip_cells;
        roundtrip_cells.reserve(32U);
        const float roundtrip_u_margin = 0.02f * (u_max_ - u_min_);
        const float roundtrip_v_margin = 0.02f * (v_max_ - v_min_);
        const std::array<std::pair<float, float>, 5> fixed_uv{{
            {center_u_, center_v_},
            {u_min_ + roundtrip_u_margin, v_min_ + roundtrip_v_margin},
            {u_max_ - roundtrip_u_margin, v_min_ + roundtrip_v_margin},
            {u_min_ + roundtrip_u_margin, v_max_ - roundtrip_v_margin},
            {u_max_ - roundtrip_u_margin, v_max_ - roundtrip_v_margin}}};
        for (const auto& uv : fixed_uv) {
            int logical_x = 0;
            int logical_y = 0;
            if (world_to_cell(uv.first, uv.second, logical_x, logical_y)) {
                roundtrip_cells.emplace_back(logical_x, logical_y);
            }
        }
        std::size_t occupied_samples = 0U;
        for (std::size_t cell = 0; cell < cell_count && occupied_samples < 16U; ++cell) {
            if (current_observed_floor_mask[cell] == 0U) {
                continue;
            }
            roundtrip_cells.emplace_back(
                static_cast<int>(cell % static_cast<std::size_t>(logical_width_)),
                static_cast<int>(cell / static_cast<std::size_t>(logical_width_)));
            ++occupied_samples;
        }
        std::size_t roundtrip_mapped = 0U;
        std::size_t roundtrip_passed = 0U;
        std::size_t roundtrip_failures = 0U;
        std::size_t roundtrip_max_dx = 0U;
        std::size_t roundtrip_max_dy = 0U;
        for (const auto& cell : roundtrip_cells) {
            const cv::Vec3f point = cell_to_reference_floor_point(cell.first, cell.second);
            const cv::Vec3f delta = point - plane_origin_;
            const float u = axis_x_.dot(delta);
            const float v = axis_y_.dot(delta);
            int mapped_x = 0;
            int mapped_y = 0;
            if (!physical_uv_to_cell(u, v, mapped_x, mapped_y)) {
                ++roundtrip_failures;
                continue;
            }
            ++roundtrip_mapped;
            const std::size_t dx = static_cast<std::size_t>(
                std::abs(mapped_x - cell.first));
            const std::size_t dy = static_cast<std::size_t>(
                std::abs(mapped_y - cell.second));
            roundtrip_max_dx = std::max(roundtrip_max_dx, dx);
            roundtrip_max_dy = std::max(roundtrip_max_dy, dy);
            if (dx <= 1U && dy <= 1U) {
                ++roundtrip_passed;
            } else {
                ++roundtrip_failures;
            }
        }
        const bool roundtrip_ok = roundtrip_failures == 0U
            && roundtrip_mapped == roundtrip_cells.size();
        std::clog << "[WINDOWS_ATLAS_ROUNDTRIP] cells=" << roundtrip_cells.size()
                  << " mapped=" << roundtrip_mapped
                  << " passed=" << roundtrip_passed
                  << " failures=" << roundtrip_failures
                  << " max_abs_dx=" << roundtrip_max_dx
                  << " max_abs_dy=" << roundtrip_max_dy
                  << " status=" << (roundtrip_ok ? "PASS" : "FAIL") << std::endl;
    }
#endif

    const float reprojection_world_tolerance = std::max(
        3.0f * floor_band_, 0.012f * scene_scale_);
    const float reprojection_depth_tolerance = std::max(
        3.0f * floor_band_, 0.015f * scene_scale_);

    // F4 is the direct three-view visibility union.  It uses only an
    // existing near-floor observation from one source view; the other views
    // can support, occlude, lie outside, or be invalid.  Only Contradict is a
    // veto.  This promotion happens before every inferred floor tier.
    std::vector<FloorSample> f4_best(cell_count);
    std::vector<int> f4_source_view(cell_count, -1);
#if defined(_WIN32)
    std::array<std::size_t, 3> f4_candidate_counts{};
    std::array<std::size_t, 3> f4_rejected_contradict_counts{};
    std::array<std::size_t, 3> f4_accepted_counts{};
#endif
    for (int source_view = 0; source_view < 3; ++source_view) {
        const std::size_t source_offset = static_cast<std::size_t>(source_view) * cell_count;
        for (std::size_t cell = 0; cell < cell_count; ++cell) {
            if (current_observed_floor_mask[cell] != 0U) {
                continue;
            }
            const FloorSample& sample = near_floor_best[source_offset + cell];
            if (!sample.valid) {
                continue;
            }
#if defined(_WIN32)
            ++f4_candidate_counts[static_cast<std::size_t>(source_view)];
#endif
            const int logical_x = static_cast<int>(
                cell % static_cast<std::size_t>(logical_width_));
            const int logical_y = static_cast<int>(
                cell / static_cast<std::size_t>(logical_width_));
            const cv::Vec3f plane_point = cell_to_reference_floor_point(logical_x, logical_y);
            bool contradicted = false;
            for (int target_view = 0; target_view < 3; ++target_view) {
                if (target_view == source_view) {
                    continue;
                }
                const ReprojectionResult reprojection = project_world_point_to_view(
                    plane_point,
                    transforms[static_cast<std::size_t>(target_view)],
                    views[static_cast<std::size_t>(target_view)],
                    reprojection_world_tolerance,
                    reprojection_depth_tolerance);
                if (reprojection.relation == ReprojectionRelation::Contradict) {
                    contradicted = true;
                    break;
                }
            }
            if (contradicted) {
#if defined(_WIN32)
                ++f4_rejected_contradict_counts[static_cast<std::size_t>(source_view)];
#endif
                continue;
            }
            if (!f4_best[cell].valid || sample.weight > f4_best[cell].weight) {
                f4_best[cell] = sample;
                f4_source_view[cell] = source_view;
            }
        }
    }
    std::size_t f4_accepted_total = 0U;
    for (std::size_t cell = 0; cell < cell_count; ++cell) {
        if (current_observed_floor_mask[cell] != 0U || !f4_best[cell].valid) {
            continue;
        }
        const int source_view = f4_source_view[cell];
        if (source_view < 0 || source_view >= 3) {
            continue;
        }
        const FloorSample& sample = f4_best[cell];
        const auto& gain = color_gain_[static_cast<std::size_t>(source_view)];
        const cv::Vec3f corrected_color(
            std::clamp(sample.color[0] * gain[0], 0.0f, 1.0f),
            std::clamp(sample.color[1] * gain[1], 0.0f, 1.0f),
            std::clamp(sample.color[2] * gain[2], 0.0f, 1.0f));
        const int logical_x = static_cast<int>(
            cell % static_cast<std::size_t>(logical_width_));
        const int logical_y = static_cast<int>(
            cell / static_cast<std::size_t>(logical_width_));
        FusedSlot next = floor_cells_[cell];
        next.slot_id = slot_for(logical_x, logical_y, 0);
        next.depth = 0.0f;
        next.confidence = std::clamp(sample.normalized_confidence, 0.0f, 1.0f);
        if (finite_vec(corrected_color)) {
            next.rgba = color_to_rgba(corrected_color);
        }
        next.floor = true;
        floor_cells_[cell] = next;
        floor_cell_valid_[cell] = 1U;
        current_observed_floor_mask[cell] = 1U;
        ++f4_accepted_total;
#if defined(_WIN32)
        ++f4_accepted_counts[static_cast<std::size_t>(source_view)];
#endif
    }
#if defined(_WIN32)
    if (accepted_fuse_count_ == 0U) {
        std::size_t direct_floor_after_f4 = 0U;
        std::size_t lost_floor_after_f4 = 0U;
        std::array<std::size_t, 3> floor_only_kept_after_f4{};
        std::array<std::size_t, 3> floor_only_lost_after_f4{};
        cv::Mat lost_floor_direct_after(logical_height_, logical_width_, CV_8UC1,
            cv::Scalar(0));
        for (std::size_t cell = 0; cell < cell_count; ++cell) {
            bool direct = false;
            std::uint8_t pattern = 0U;
            for (int view_index = 0; view_index < 3; ++view_index) {
                if (direct_floor_presence_by_view[static_cast<std::size_t>(view_index)][cell]
                    != 0U) {
                    direct = true;
                    pattern |= static_cast<std::uint8_t>(1U) << view_index;
                }
            }
            if (!direct) {
                continue;
            }
            if (current_observed_floor_mask[cell] != 0U) {
                ++direct_floor_after_f4;
            } else {
                ++lost_floor_after_f4;
                const int x = static_cast<int>(cell % static_cast<std::size_t>(logical_width_));
                const int y = static_cast<int>(cell / static_cast<std::size_t>(logical_width_));
                lost_floor_direct_after.at<std::uint8_t>(y, x) = 255U;
            }
            if (support_count(pattern) == 1) {
                const int view_index = pattern == 1U ? 0 : (pattern == 2U ? 1 : 2);
                if (current_observed_floor_mask[cell] != 0U) {
                    ++floor_only_kept_after_f4[static_cast<std::size_t>(view_index)];
                } else {
                    ++floor_only_lost_after_f4[static_cast<std::size_t>(view_index)];
                }
            }
        }
        try {
            if (!cv::imwrite("tmp/lost_floor_direct_after.png",
                    resize_logical_mask(lost_floor_direct_after))) {
                std::clog << "[WINDOWS_DIRECT_FLOOR_MASK] write_failed=1" << std::endl;
            }
        } catch (const cv::Exception&) {
            std::clog << "[WINDOWS_DIRECT_FLOOR_MASK] write_failed=1" << std::endl;
        }
        std::clog << "[WINDOWS_F4] candidates_view0=" << f4_candidate_counts[0]
                  << " candidates_view1=" << f4_candidate_counts[1]
                  << " candidates_view2=" << f4_candidate_counts[2]
                  << " rejected_contradict_view0=" << f4_rejected_contradict_counts[0]
                  << " rejected_contradict_view1=" << f4_rejected_contradict_counts[1]
                  << " rejected_contradict_view2=" << f4_rejected_contradict_counts[2]
                  << " accepted_view0=" << f4_accepted_counts[0]
                  << " accepted_view1=" << f4_accepted_counts[1]
                  << " accepted_view2=" << f4_accepted_counts[2]
                  << " accepted_total=" << f4_accepted_total
                  << " direct_floor_after_f4=" << direct_floor_after_f4
                  << " lost_floor_after_f4=" << lost_floor_after_f4
                  << " only_view0_kept_after_f4=" << floor_only_kept_after_f4[0]
                  << " only_view1_kept_after_f4=" << floor_only_kept_after_f4[1]
                  << " only_view2_kept_after_f4=" << floor_only_kept_after_f4[2]
                  << " only_view0_lost_after_f4=" << floor_only_lost_after_f4[0]
                  << " only_view1_lost_after_f4=" << floor_only_lost_after_f4[1]
                  << " only_view2_lost_after_f4=" << floor_only_lost_after_f4[2]
                  << std::endl;
    }
#endif

    result.slots.reserve(cell_count / 2U);
    result.occupied_slots.reserve(cell_count / 2U);
    for (std::size_t cell = 0; cell < cell_count; ++cell) {
        if (floor_cell_valid_[cell] != 0U) {
            result.slots.push_back(floor_cells_[cell]);
            result.occupied_slots.push_back(floor_cells_[cell].slot_id);
        }
    }

    // First reduce each view to at most one robust object observation per
    // logical cell. The views are deliberately kept separate until the
    // cross-view consistency pass below; mixing them first was able to make a
    // single-view curtain look like a supported surface.
    std::array<std::vector<ViewObjectObservation>, 3> object_observations_by_view;
    std::array<std::vector<int>, 3> object_observation_index_by_view;
#if defined(_WIN32)
    std::array<cv::Mat, 3> object_raw_masks;
    for (int view_index = 0; view_index < 3; ++view_index) {
        object_raw_masks[static_cast<std::size_t>(view_index)] = cv::Mat(
            logical_height_, logical_width_, CV_8UC1, cv::Scalar(0));
    }
#endif
    std::size_t object_pre_consistency = 0U;
    for (int view_index = 0; view_index < 3; ++view_index) {
        object_observation_index_by_view[static_cast<std::size_t>(view_index)].assign(
            cell_count, -1);
        const auto& per_cell_samples = object_samples_by_view[
            static_cast<std::size_t>(view_index)];
        for (std::size_t cell = 0; cell < cell_count; ++cell) {
            const auto& samples = per_cell_samples[cell];
            if (samples.empty()) {
                continue;
            }
            ViewObjectObservation observation;
            observation.valid = true;
            observation.view_id = view_index;
            observation.logical_x = static_cast<int>(
                cell % static_cast<std::size_t>(logical_width_));
            observation.logical_y = static_cast<int>(
                cell / static_cast<std::size_t>(logical_width_));
            std::vector<float> u_values;
            std::vector<float> v_values;
            std::vector<float> confidence_values;
            u_values.reserve(samples.size());
            v_values.reserve(samples.size());
            confidence_values.reserve(samples.size());
            observation.weight = 0.0f;
            observation.direct_margin = false;
            for (const ObjectSample& sample : samples) {
                u_values.push_back(sample.u);
                v_values.push_back(sample.v);
                confidence_values.push_back(sample.normalized_confidence);
                observation.weight += std::max(sample.weight, 1e-4f);
                observation.direct_margin = observation.direct_margin || sample.direct_margin;
            }
            observation.u = median_value(std::move(u_values));
            observation.v = median_value(std::move(v_values));
            observation.height = weighted_median_height(samples);
            observation.confidence = median_value(std::move(confidence_values));
            float ignored_confidence = 0.0f;
            observation.color = weighted_color(samples, color_gain_, ignored_confidence);
            observation.sample_count = static_cast<int>(samples.size());
            observation.valid = finite_vec(observation.color)
                && std::isfinite(observation.u)
                && std::isfinite(observation.v)
                && std::isfinite(observation.height)
                && std::isfinite(observation.confidence)
                && std::isfinite(observation.weight)
                && observation.weight > 0.0f;
            if (!observation.valid) {
                continue;
            }
            auto& observations = object_observations_by_view[
                static_cast<std::size_t>(view_index)];
            object_observation_index_by_view[static_cast<std::size_t>(view_index)][cell] =
                static_cast<int>(observations.size());
            observations.push_back(std::move(observation));
#if defined(_WIN32)
            object_raw_masks[static_cast<std::size_t>(view_index)].at<std::uint8_t>(
                static_cast<int>(cell / static_cast<std::size_t>(logical_width_)),
                static_cast<int>(cell % static_cast<std::size_t>(logical_width_))) = 255U;
#endif
            ++object_pre_consistency;
        }
    }

#if defined(_WIN32)
    std::array<std::vector<std::uint8_t>, 3> direct_object_presence_by_view;
    for (int view_index = 0; view_index < 3; ++view_index) {
        auto& direct_presence = direct_object_presence_by_view[
            static_cast<std::size_t>(view_index)];
        direct_presence.assign(cell_count, 0U);
        for (const ViewObjectObservation& observation : object_observations_by_view[
                static_cast<std::size_t>(view_index)]) {
            const std::size_t cell = static_cast<std::size_t>(observation.logical_y)
                * static_cast<std::size_t>(logical_width_)
                + static_cast<std::size_t>(observation.logical_x);
            if (cell < cell_count) {
                direct_presence[cell] = 1U;
            }
        }
    }
#endif

    const float object_cross_view_height_tolerance = std::max(
        3.0f * floor_band_, 0.02f * scene_scale_);

#if defined(_WIN32)
    // Audit every reduced single-view object observation against both other
    // point heads.  This is diagnostic-only: the existing relation rules and
    // Contradict vetoes below remain unchanged.
    std::array<std::array<std::size_t, 25>, 3> object_relation_pattern_counts{};
    std::array<std::array<std::size_t, 3>, 3> object_contradict_counts_by_target{};
    std::size_t object_contradict_total = 0U;
    std::array<std::vector<float>, 9> object_contradict_world_errors;
    std::array<std::vector<float>, 9> object_contradict_depth_errors;
    std::vector<float> object_contradict_world_errors_all;
    std::vector<float> object_contradict_depth_errors_all;
    std::array<cv::Mat, 3> object_contradict_masks;
    std::array<cv::Mat, 3> object_noncontradict_masks;
    for (int view_index = 0; view_index < 3; ++view_index) {
        object_contradict_masks[static_cast<std::size_t>(view_index)] = cv::Mat(
            logical_height_, logical_width_, CV_8UC1, cv::Scalar(0));
        object_noncontradict_masks[static_cast<std::size_t>(view_index)] = cv::Mat(
            logical_height_, logical_width_, CV_8UC1, cv::Scalar(0));
        const auto& observations = object_observations_by_view[
            static_cast<std::size_t>(view_index)];
        for (const ViewObjectObservation& observation : observations) {
            const cv::Vec3f point_reference = plane_origin_
                + axis_x_ * observation.u
                + axis_y_ * observation.v
                + plane_normal_ * observation.height;
            std::array<int, 2> relation_codes{};
            bool has_contradict = false;
            int relation_slot = 0;
            for (int target_view = 0; target_view < 3; ++target_view) {
                if (target_view == view_index) {
                    continue;
                }
                const ReprojectionResult reprojection = project_world_point_to_view(
                    point_reference,
                    transforms[static_cast<std::size_t>(view_index)],
                    views[static_cast<std::size_t>(target_view)],
                    reprojection_world_tolerance,
                    reprojection_depth_tolerance);
                int relation_code = 2; // Invalid
                switch (reprojection.relation) {
                case ReprojectionRelation::Support:
                    relation_code = 0;
                    break;
                case ReprojectionRelation::Occluded:
                    relation_code = 1;
                    break;
                case ReprojectionRelation::Invalid:
                    relation_code = 2;
                    break;
                case ReprojectionRelation::Outside:
                    relation_code = 3;
                    break;
                case ReprojectionRelation::Contradict:
                    relation_code = 4;
                    has_contradict = true;
                    ++object_contradict_total;
                    ++object_contradict_counts_by_target[
                        static_cast<std::size_t>(view_index)][
                            static_cast<std::size_t>(target_view)];
                    if (std::isfinite(reprojection.world_error)) {
                        object_contradict_world_errors[static_cast<std::size_t>(
                            view_index * 3 + target_view)].push_back(
                            reprojection.world_error);
                        object_contradict_world_errors_all.push_back(
                            reprojection.world_error);
                    }
                    if (std::isfinite(reprojection.depth_error)) {
                        object_contradict_depth_errors[static_cast<std::size_t>(
                            view_index * 3 + target_view)].push_back(
                            reprojection.depth_error);
                        object_contradict_depth_errors_all.push_back(
                            reprojection.depth_error);
                    }
                    break;
                }
                relation_codes[static_cast<std::size_t>(relation_slot)] = relation_code;
                ++relation_slot;
            }
            const int first_relation = std::min(relation_codes[0], relation_codes[1]);
            const int second_relation = std::max(relation_codes[0], relation_codes[1]);
            ++object_relation_pattern_counts[static_cast<std::size_t>(view_index)][
                first_relation * 5 + second_relation];
            const int x = observation.logical_x;
            const int y = observation.logical_y;
            if (has_contradict) {
                object_contradict_masks[static_cast<std::size_t>(view_index)].at<
                    std::uint8_t>(y, x) = 255U;
            } else {
                object_noncontradict_masks[static_cast<std::size_t>(view_index)].at<
                    std::uint8_t>(y, x) = 255U;
            }
        }
    }
    const auto relation_label = [](const int relation) {
        constexpr std::array<const char*, 5> labels{{"S", "O", "I", "X", "C"}};
        return labels[static_cast<std::size_t>(relation)];
    };
    const auto log_quantiles = [](const char* prefix, const std::vector<float>& values) {
        std::clog << prefix << "_n=" << values.size()
                  << "_q10=" << percentile_value(values, 0.10f)
                  << "_q50=" << percentile_value(values, 0.50f)
                  << "_q90=" << percentile_value(values, 0.90f)
                  << "_q99=" << percentile_value(values, 0.99f);
    };
    if (accepted_fuse_count_ == 0U) {
        for (int view_index = 0; view_index < 3; ++view_index) {
            std::clog << "[WINDOWS_OBJECT_RELATIONS] source_view=" << view_index;
            for (int first_relation = 0; first_relation < 5; ++first_relation) {
                for (int second_relation = first_relation; second_relation < 5;
                    ++second_relation) {
                    const std::size_t count = object_relation_pattern_counts[
                        static_cast<std::size_t>(view_index)][first_relation * 5
                        + second_relation];
                    std::clog << " " << relation_label(first_relation)
                              << relation_label(second_relation) << "=" << count;
                }
            }
            std::clog << std::endl;
        }
        std::clog << "[WINDOWS_OBJECT_CONTRADICT] source_view0="
                  << object_contradict_counts_by_target[0][1]
                        + object_contradict_counts_by_target[0][2]
                  << " source_view1="
                  << object_contradict_counts_by_target[1][0]
                        + object_contradict_counts_by_target[1][2]
                  << " source_view2="
                  << object_contradict_counts_by_target[2][0]
                        + object_contradict_counts_by_target[2][1]
                  << " total=" << object_contradict_total
                  << " target_0_to_1=" << object_contradict_counts_by_target[0][1]
                  << " target_0_to_2=" << object_contradict_counts_by_target[0][2]
                  << " target_1_to_0=" << object_contradict_counts_by_target[1][0]
                  << " target_1_to_2=" << object_contradict_counts_by_target[1][2]
                  << " target_2_to_0=" << object_contradict_counts_by_target[2][0]
                  << " target_2_to_1=" << object_contradict_counts_by_target[2][1]
                  << std::endl;
        std::clog << "[WINDOWS_OBJECT_CONTRADICT_ERROR] aggregate ";
        log_quantiles("world_error", object_contradict_world_errors_all);
        std::clog << " ";
        log_quantiles("depth_error", object_contradict_depth_errors_all);
        std::clog << std::endl;
        for (int source_view = 0; source_view < 3; ++source_view) {
            for (int target_view = 0; target_view < 3; ++target_view) {
                if (source_view == target_view) {
                    continue;
                }
                const std::size_t pair = static_cast<std::size_t>(
                    source_view * 3 + target_view);
                std::clog << "[WINDOWS_OBJECT_CONTRADICT_ERROR] pair_"
                          << source_view << "_to_" << target_view << " ";
                log_quantiles("world_error", object_contradict_world_errors[pair]);
                std::clog << " ";
                log_quantiles("depth_error", object_contradict_depth_errors[pair]);
                std::clog << std::endl;
            }
        }
        try {
            for (int view_index = 0; view_index < 3; ++view_index) {
                const std::string contradict_path = "tmp/object_contradict_view"
                    + std::to_string(view_index) + ".png";
                const std::string noncontradict_path = "tmp/object_noncontradict_view"
                    + std::to_string(view_index) + ".png";
                if (!cv::imwrite(contradict_path, resize_logical_mask(
                        object_contradict_masks[static_cast<std::size_t>(view_index)]))
                    || !cv::imwrite(noncontradict_path, resize_logical_mask(
                        object_noncontradict_masks[static_cast<std::size_t>(view_index)]))) {
                    std::clog << "[WINDOWS_OBJECT_CONTRADICT_MASK] write_failed=1"
                              << std::endl;
                }
            }
        } catch (const cv::Exception&) {
            std::clog << "[WINDOWS_OBJECT_CONTRADICT_MASK] write_failed=1"
                      << std::endl;
        }
    }
#endif

    std::vector<ObjectSurfaceCandidate> object_best(cell_count);
#if defined(_WIN32)
    // Diagnostics only: record which frozen object tier supplied the final
    // candidate. This vector never participates in candidate selection.
    std::vector<std::uint8_t> object_tier_provenance(cell_count, 0U);
#endif
    std::array<std::vector<std::uint8_t>, 3> object_matched_by_view;
    for (int view_index = 0; view_index < 3; ++view_index) {
        object_matched_by_view[static_cast<std::size_t>(view_index)].assign(
            cell_count, 0U);
    }
    std::vector<std::uint8_t> atlas_support2_cells(cell_count, 0U);
    std::vector<std::uint8_t> atlas_support3_cells(cell_count, 0U);
    ReprojectionStats reprojection_stats;

    const auto object_candidate_is_better = [](const ObjectSurfaceCandidate& candidate,
        const ObjectSurfaceCandidate& current) {
        const int candidate_support = support_count(candidate.view_mask);
        const int current_support = support_count(current.view_mask);
        return candidate_support > current_support
            || (candidate_support == current_support && candidate.score > current.score)
            || (candidate_support == current_support
                && candidate.score == current.score
                && candidate.sample_count > current.sample_count)
            || (candidate_support == current_support
                && candidate.score == current.score
                && candidate.sample_count == current.sample_count
                && candidate.height < current.height);
    };
    const auto consider_object_candidate = [&](ObjectSurfaceCandidate candidate) {
        if (!candidate.valid || candidate.logical_x < 0 || candidate.logical_x >= logical_width_
            || candidate.logical_y < 0 || candidate.logical_y >= logical_height_
            || support_count(candidate.view_mask) < 2) {
            return;
        }
        const std::size_t cell = static_cast<std::size_t>(candidate.logical_y)
            * static_cast<std::size_t>(logical_width_) + static_cast<std::size_t>(candidate.logical_x);
        ObjectSurfaceCandidate& current = object_best[cell];
        if (!current.valid) {
            current = std::move(candidate);
            return;
        }
        if (std::abs(current.height - candidate.height)
            <= object_cross_view_height_tolerance) {
            ObjectSurfaceCandidate merged = current;
            for (int view = 0; view < 3; ++view) {
                const std::uint8_t bit = static_cast<std::uint8_t>(1U) << view;
                if ((candidate.view_mask & bit) == 0U) {
                    continue;
                }
                if ((merged.view_mask & bit) == 0U
                    || candidate.observations[static_cast<std::size_t>(view)].weight
                        > merged.observations[static_cast<std::size_t>(view)].weight) {
                    merged.observations[static_cast<std::size_t>(view)] =
                        candidate.observations[static_cast<std::size_t>(view)];
                    merged.view_mask |= bit;
                }
            }
            finalize_object_candidate(merged);
            merged.logical_x = current.logical_x;
            merged.logical_y = current.logical_y;
            if (object_candidate_is_better(merged, current)) {
                current = std::move(merged);
            }
        } else if (object_candidate_is_better(candidate, current)) {
            current = std::move(candidate);
        }
    };

    const auto find_object_observation = [&](const int view_index,
        const int logical_x, const int logical_y) -> const ViewObjectObservation* {
        if (view_index < 0 || view_index >= 3 || logical_x < 0 || logical_x >= logical_width_
            || logical_y < 0 || logical_y >= logical_height_) {
            return nullptr;
        }
        const std::size_t cell = static_cast<std::size_t>(logical_y)
            * static_cast<std::size_t>(logical_width_) + static_cast<std::size_t>(logical_x);
        const int index = object_observation_index_by_view[
            static_cast<std::size_t>(view_index)][cell];
        if (index < 0) {
            return nullptr;
        }
        const auto& observations = object_observations_by_view[
            static_cast<std::size_t>(view_index)];
        return &observations[static_cast<std::size_t>(index)];
    };
    const auto mark_object_matched = [&](const ViewObjectObservation& observation) {
        if (observation.view_id < 0 || observation.view_id >= 3) {
            return;
        }
        const std::size_t cell = static_cast<std::size_t>(observation.logical_y)
            * static_cast<std::size_t>(logical_width_)
            + static_cast<std::size_t>(observation.logical_x);
        object_matched_by_view[static_cast<std::size_t>(observation.view_id)][cell] = 1U;
    };

    const auto representative_reference_point = [&](const ViewObjectObservation& observation) {
        return plane_origin_
            + axis_x_ * observation.u
            + axis_y_ * observation.v
            + plane_normal_ * observation.height;
    };

    struct ObjectPairProposal {
        int first_view = -1;
        int second_view = -1;
        const ViewObjectObservation* first = nullptr;
        const ViewObjectObservation* second = nullptr;
        ReprojectionResult first_to_second;
        ReprojectionResult second_to_first;
        bool bidirectional_support = false;
    };

    // Atlas proximity is only a proposal.  Each compatible observation pair
    // is evaluated once in both directions against the target point head.
    std::vector<ObjectPairProposal> pair_proposals;
    for (int first_view = 0; first_view < 3; ++first_view) {
        for (int second_view = first_view + 1; second_view < 3; ++second_view) {
            for (const ViewObjectObservation& first : object_observations_by_view[
                    static_cast<std::size_t>(first_view)]) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const ViewObjectObservation* second = find_object_observation(
                            second_view, first.logical_x + dx, first.logical_y + dy);
                        if (second == nullptr
                            || std::abs(first.height - second->height)
                                > object_cross_view_height_tolerance) {
                            continue;
                        }

                        ObjectPairProposal proposal;
                        proposal.first_view = first_view;
                        proposal.second_view = second_view;
                        proposal.first = &first;
                        proposal.second = second;
                        proposal.first_to_second = project_world_point_to_view(
                            representative_reference_point(first),
                            transforms[static_cast<std::size_t>(first_view)],
                            views[static_cast<std::size_t>(second_view)],
                            reprojection_world_tolerance,
                            reprojection_depth_tolerance);
                        proposal.second_to_first = project_world_point_to_view(
                            representative_reference_point(*second),
                            transforms[static_cast<std::size_t>(second_view)],
                            views[static_cast<std::size_t>(first_view)],
                            reprojection_world_tolerance,
                            reprojection_depth_tolerance);
                        const bool first_support = proposal.first_to_second.relation
                            == ReprojectionRelation::Support;
                        const bool second_support = proposal.second_to_first.relation
                            == ReprojectionRelation::Support;
                        proposal.bidirectional_support = first_support && second_support;
                        ++reprojection_stats.atlas_pair;
                        if (proposal.bidirectional_support) {
                            ++reprojection_stats.bidir_support;
                        } else {
                            if (first_support != second_support) {
                                ++reprojection_stats.oneway_support;
                            }
                            record_failed_reprojection(
                                proposal.first_to_second.relation, reprojection_stats);
                            record_failed_reprojection(
                                proposal.second_to_first.relation, reprojection_stats);
                        }
                        atlas_support2_cells[static_cast<std::size_t>(
                            consensus_coordinate({first.logical_x, second->logical_x})
                            + logical_width_ * consensus_coordinate({
                                first.logical_y, second->logical_y}))] = 1U;
                        mark_object_matched(first);
                        mark_object_matched(*second);
                        pair_proposals.push_back(std::move(proposal));
                    }
                }
            }
        }
    }

    const auto find_pair_proposal = [&](const int first_view,
        const ViewObjectObservation& first, const int second_view,
        const ViewObjectObservation& second) -> const ObjectPairProposal* {
        const int low_view = std::min(first_view, second_view);
        const int high_view = std::max(first_view, second_view);
        const ViewObjectObservation* low_observation = first_view == low_view
            ? &first : &second;
        const ViewObjectObservation* high_observation = first_view == low_view
            ? &second : &first;
        for (const ObjectPairProposal& proposal : pair_proposals) {
            if (proposal.first_view == low_view && proposal.second_view == high_view
                && proposal.first != nullptr && proposal.second != nullptr
                && proposal.first->logical_x == low_observation->logical_x
                && proposal.first->logical_y == low_observation->logical_y
                && proposal.second->logical_x == high_observation->logical_x
                && proposal.second->logical_y == high_observation->logical_y) {
                return &proposal;
            }
        }
        return nullptr;
    };

    const auto make_pair_candidate = [](const ObjectPairProposal& proposal) {
        ObjectSurfaceCandidate pair;
        pair.valid = true;
        pair.logical_x = consensus_coordinate({
            proposal.first->logical_x, proposal.second->logical_x});
        pair.logical_y = consensus_coordinate({
            proposal.first->logical_y, proposal.second->logical_y});
        pair.view_mask = static_cast<std::uint8_t>(
            (static_cast<std::uint8_t>(1U) << proposal.first_view)
            | (static_cast<std::uint8_t>(1U) << proposal.second_view));
        pair.observations[static_cast<std::size_t>(proposal.first_view)] = *proposal.first;
        pair.observations[static_cast<std::size_t>(proposal.second_view)] = *proposal.second;
        finalize_object_candidate(pair);
        return pair;
    };

    // A three-view candidate is accepted only when the base pair and at
    // least one of its two edges to the third view are both bidirectional
    // Support.  The pair is kept local until all third-view checks finish.
    for (const ObjectPairProposal& proposal : pair_proposals) {
        if (!proposal.bidirectional_support) {
            continue;
        }
        ObjectSurfaceCandidate pair = make_pair_candidate(proposal);
        const int third_view = 3 - proposal.first_view - proposal.second_view;
        const int min_x = std::max(
            proposal.first->logical_x - 1, proposal.second->logical_x - 1);
        const int max_x = std::min(
            proposal.first->logical_x + 1, proposal.second->logical_x + 1);
        const int min_y = std::max(
            proposal.first->logical_y - 1, proposal.second->logical_y - 1);
        const int max_y = std::min(
            proposal.first->logical_y + 1, proposal.second->logical_y + 1);
        const ViewObjectObservation* best_third = nullptr;
        int best_edge_count = 1;
        float best_third_score = -std::numeric_limits<float>::infinity();
        for (int third_y = min_y; third_y <= max_y; ++third_y) {
            for (int third_x = min_x; third_x <= max_x; ++third_x) {
                const ViewObjectObservation* third = find_object_observation(
                    third_view, third_x, third_y);
                if (third == nullptr
                    || std::abs(third->height - pair.height)
                        > object_cross_view_height_tolerance) {
                    continue;
                }
                const ObjectPairProposal* first_third = find_pair_proposal(
                    proposal.first_view, *proposal.first, third_view, *third);
                const ObjectPairProposal* second_third = find_pair_proposal(
                    proposal.second_view, *proposal.second, third_view, *third);
                const int edge_count = 1
                    + (first_third != nullptr && first_third->bidirectional_support ? 1 : 0)
                    + (second_third != nullptr && second_third->bidirectional_support ? 1 : 0);
                if (edge_count < 2) {
                    continue;
                }
                const float third_score = third->weight
                    + 0.001f * static_cast<float>(third->sample_count);
                if (best_third == nullptr || edge_count > best_edge_count
                    || (edge_count == best_edge_count && third_score > best_third_score)) {
                    best_third = third;
                    best_edge_count = edge_count;
                    best_third_score = third_score;
                }
            }
        }

        if (best_third != nullptr) {
            ObjectSurfaceCandidate triple = pair;
            triple.logical_x = -1;
            triple.logical_y = -1;
            triple.view_mask |= static_cast<std::uint8_t>(
                static_cast<std::uint8_t>(1U) << third_view);
            triple.observations[static_cast<std::size_t>(third_view)] = *best_third;
            finalize_object_candidate(triple);
            const std::size_t triple_cell = static_cast<std::size_t>(triple.logical_y)
                * static_cast<std::size_t>(logical_width_)
                + static_cast<std::size_t>(triple.logical_x);
            if (triple_cell < cell_count) {
                atlas_support3_cells[triple_cell] = 1U;
            }
            mark_object_matched(*best_third);
            consider_object_candidate(std::move(triple));
#if defined(_WIN32)
            if (triple_cell < cell_count && object_best[triple_cell].valid) {
                object_tier_provenance[triple_cell] = 1U;
            }
#endif
        } else {
            // No third view was needed to validate this strict pair.  The
            // candidate is moved only after all optional third-view work.
            const std::size_t pair_cell = static_cast<std::size_t>(pair.logical_y)
                * static_cast<std::size_t>(logical_width_)
                + static_cast<std::size_t>(pair.logical_x);
            consider_object_candidate(std::move(pair));
#if defined(_WIN32)
            if (pair_cell < cell_count && object_best[pair_cell].valid) {
                object_tier_provenance[pair_cell] = 1U;
            }
#endif
        }
    }

    // Fallback candidates use the same pair proposals as the strict pass.
    // Only a one-way reprojection Support paired with Occluded is accepted;
    // all other relations remain rejected.  The fallback is kept separate
    // from the strict result and can only fill cells that strict left empty.
    std::vector<ObjectSurfaceCandidate> fallback_best(cell_count);
    for (const ObjectPairProposal& proposal : pair_proposals) {
        const bool first_support = proposal.first_to_second.relation
            == ReprojectionRelation::Support;
        const bool second_support = proposal.second_to_first.relation
            == ReprojectionRelation::Support;
        const bool first_occluded = proposal.first_to_second.relation
            == ReprojectionRelation::Occluded;
        const bool second_occluded = proposal.second_to_first.relation
            == ReprojectionRelation::Occluded;
        if (!((first_support && second_occluded)
                || (second_support && first_occluded))) {
            continue;
        }
        ObjectSurfaceCandidate fallback = make_pair_candidate(proposal);
        if (!fallback.valid
            || fallback.logical_x < 0 || fallback.logical_x >= logical_width_
            || fallback.logical_y < 0 || fallback.logical_y >= logical_height_
            || support_count(fallback.view_mask) != 2) {
            continue;
        }
        const std::size_t cell = static_cast<std::size_t>(fallback.logical_y)
            * static_cast<std::size_t>(logical_width_)
            + static_cast<std::size_t>(fallback.logical_x);
        if (cell >= cell_count || object_best[cell].valid) {
            continue;
        }
        ObjectSurfaceCandidate& current = fallback_best[cell];
        if (!current.valid || object_candidate_is_better(fallback, current)) {
            current = std::move(fallback);
        }
    }
    for (std::size_t cell = 0; cell < cell_count; ++cell) {
        if (object_best[cell].valid || !fallback_best[cell].valid) {
            continue;
        }
        object_best[cell] = std::move(fallback_best[cell]);
#if defined(_WIN32)
        object_tier_provenance[cell] = 2U;
#endif
    }

    // Tier 3 is deliberately limited to the Phase-A-supported case: a
    // two-view atlas candidate with no reprojection Support and no
    // Contradict relation. It is built from the same robust per-view
    // observations and pair proposals as the first two tiers. The independent
    // best array ensures Tier 3 can only fill a cell left empty by Tier 1/2.
    std::vector<ObjectSurfaceCandidate> atlas_consensus_best(cell_count);
    for (const ObjectPairProposal& proposal : pair_proposals) {
        const bool first_support = proposal.first_to_second.relation
            == ReprojectionRelation::Support;
        const bool second_support = proposal.second_to_first.relation
            == ReprojectionRelation::Support;
        const bool first_contradict = proposal.first_to_second.relation
            == ReprojectionRelation::Contradict;
        const bool second_contradict = proposal.second_to_first.relation
            == ReprojectionRelation::Contradict;
        if (first_support || second_support || first_contradict || second_contradict) {
            continue;
        }
        ObjectSurfaceCandidate atlas_candidate = make_pair_candidate(proposal);
        if (!atlas_candidate.valid
            || atlas_candidate.logical_x < 0
            || atlas_candidate.logical_x >= logical_width_
            || atlas_candidate.logical_y < 0
            || atlas_candidate.logical_y >= logical_height_
            || support_count(atlas_candidate.view_mask) != 2) {
            continue;
        }
        const std::size_t cell = static_cast<std::size_t>(atlas_candidate.logical_y)
            * static_cast<std::size_t>(logical_width_)
            + static_cast<std::size_t>(atlas_candidate.logical_x);
        ObjectSurfaceCandidate& current = atlas_consensus_best[cell];
        if (!current.valid || object_candidate_is_better(atlas_candidate, current)) {
            current = std::move(atlas_candidate);
        }
    }
    for (std::size_t cell = 0; cell < cell_count; ++cell) {
        if (object_best[cell].valid || !atlas_consensus_best[cell].valid) {
            continue;
        }
        object_best[cell] = std::move(atlas_consensus_best[cell]);
#if defined(_WIN32)
        object_tier_provenance[cell] = 3U;
#endif
    }

    std::vector<std::uint8_t> single_support_cells(cell_count, 0U);
    for (int view_index = 0; view_index < 3; ++view_index) {
        const auto& observations = object_observations_by_view[
            static_cast<std::size_t>(view_index)];
        const auto& matched = object_matched_by_view[static_cast<std::size_t>(view_index)];
        for (const ViewObjectObservation& observation : observations) {
            const std::size_t cell = static_cast<std::size_t>(observation.logical_y)
                * static_cast<std::size_t>(logical_width_)
                + static_cast<std::size_t>(observation.logical_x);
            if (matched[cell] == 0U) {
                single_support_cells[cell] = 1U;
            }
        }
    }
    const std::size_t atlas_support1 = static_cast<std::size_t>(std::count(
        single_support_cells.begin(), single_support_cells.end(), static_cast<std::uint8_t>(1U)));
    const std::size_t atlas_support2 = static_cast<std::size_t>(std::count(
        atlas_support2_cells.begin(), atlas_support2_cells.end(), static_cast<std::uint8_t>(1U)));
    const std::size_t atlas_support3 = static_cast<std::size_t>(std::count(
        atlas_support3_cells.begin(), atlas_support3_cells.end(), static_cast<std::uint8_t>(1U)));
    std::size_t reprojection_support2 = 0U;
    std::size_t reprojection_support3 = 0U;
    for (const ObjectSurfaceCandidate& candidate : object_best) {
        if (!candidate.valid) {
            continue;
        }
        if (support_count(candidate.view_mask) == 2) {
            ++reprojection_support2;
        } else if (support_count(candidate.view_mask) == 3) {
            ++reprojection_support3;
        }
    }

    // Connected components run after cross-view matching and the reprojection-
    // aware fallback. The existing minimum of eight logical cells is retained
    // to reject isolated noise.
    cv::Mat occupancy(logical_height_, logical_width_, CV_8UC1, cv::Scalar(0));
    for (std::size_t cell = 0; cell < cell_count; ++cell) {
        if (object_best[cell].valid) {
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
            if (object_best[cell].valid
                && (component <= 0 || component >= component_count
                    || stats.at<int>(component, cv::CC_STAT_AREA) < 8)) {
                object_best[cell].valid = false;
            }
        }
    }

    // Keep the post-component Tier1/2/3 result as the only trusted seed set.
    // Tier4 candidates may touch this snapshot, but they must never use one
    // another as a seed while the connected components are being expanded.
    std::vector<std::uint8_t> trusted_object_seed_mask(cell_count, 0U);
    for (std::size_t cell = 0; cell < cell_count; ++cell) {
        if (object_best[cell].valid) {
            trusted_object_seed_mask[cell] = 1U;
        }
    }

    const float height_neighbor_threshold = std::max(
        4.0f * floor_band_, 0.03f * scene_scale_);
    std::array<cv::Mat, 3> tier4_raw_masks;
    std::array<cv::Mat, 3> tier4_relation_anchor_masks;
    for (int source_view = 0; source_view < 3; ++source_view) {
        tier4_raw_masks[static_cast<std::size_t>(source_view)] = cv::Mat(
            logical_height_, logical_width_, CV_8UC1, cv::Scalar(0));
        tier4_relation_anchor_masks[static_cast<std::size_t>(source_view)] = cv::Mat(
            logical_height_, logical_width_, CV_8UC1, cv::Scalar(0));
    }
    std::vector<ObjectSurfaceCandidate> tier4_best(cell_count);
#if defined(_WIN32)
    std::size_t tier4_rejected_contradict = 0U;
    std::array<std::size_t, 3> tier4_component_counts{};
    std::array<std::size_t, 3> tier4_relation_anchor_cell_counts{};
    std::array<std::size_t, 3> tier4_relation_anchor_component_counts{};
    std::array<std::size_t, 3> tier4_seed_anchor_component_counts{};
    std::array<std::size_t, 3> tier4_both_anchor_component_counts{};
    std::array<std::size_t, 3> tier4_relation_only_component_counts{};
    std::array<std::size_t, 3> tier4_seed_only_component_counts{};
    std::array<std::size_t, 3> tier4_no_anchor_component_counts{};
    std::size_t tier4_rejected_no_anchor = 0U;
    std::size_t tier4_accepted_before_height_neighbor = 0U;
#endif

    const auto make_tier4_candidate = [&](const int source_view,
        const int logical_x, const int logical_y) {
        ObjectSurfaceCandidate candidate;
        const ViewObjectObservation* observation = find_object_observation(
            source_view, logical_x, logical_y);
        if (observation == nullptr) {
            return candidate;
        }
        candidate.valid = true;
        candidate.logical_x = logical_x;
        candidate.logical_y = logical_y;
        candidate.view_mask = static_cast<std::uint8_t>(1U) << source_view;
        candidate.observations[static_cast<std::size_t>(source_view)] = *observation;
        finalize_object_candidate(candidate);
        return candidate;
    };

    // Construct raw Tier4-v2 candidates only from observations that survived
    // the existing per-view object classifier.  Contradict remains a hard
    // veto, while Outside/Invalid are retained for component-level anchors.
    for (int source_view = 0; source_view < 3; ++source_view) {
        const auto& observations = object_observations_by_view[
            static_cast<std::size_t>(source_view)];
        for (const ViewObjectObservation& observation : observations) {
            const std::size_t cell = static_cast<std::size_t>(observation.logical_y)
                * static_cast<std::size_t>(logical_width_)
                + static_cast<std::size_t>(observation.logical_x);
            if (cell >= cell_count || object_best[cell].valid) {
                continue;
            }
            const cv::Vec3f point_reference = representative_reference_point(observation);
            bool has_relation_anchor = false;
            bool has_contradict = false;
            for (int target_view = 0; target_view < 3; ++target_view) {
                if (target_view == source_view) {
                    continue;
                }
                const ReprojectionResult reprojection = project_world_point_to_view(
                    point_reference,
                    transforms[static_cast<std::size_t>(source_view)],
                    views[static_cast<std::size_t>(target_view)],
                    reprojection_world_tolerance,
                    reprojection_depth_tolerance);
                if (reprojection.relation == ReprojectionRelation::Contradict) {
                    has_contradict = true;
                } else if (reprojection.relation == ReprojectionRelation::Support
                    || reprojection.relation == ReprojectionRelation::Occluded) {
                    has_relation_anchor = true;
                }
            }
            if (has_contradict) {
#if defined(_WIN32)
                ++tier4_rejected_contradict;
#endif
                continue;
            }
            tier4_raw_masks[static_cast<std::size_t>(source_view)].at<std::uint8_t>(
                observation.logical_y, observation.logical_x) = 255U;
            if (has_relation_anchor) {
                tier4_relation_anchor_masks[static_cast<std::size_t>(source_view)].at<
                    std::uint8_t>(observation.logical_y, observation.logical_x) = 255U;
#if defined(_WIN32)
                ++tier4_relation_anchor_cell_counts[static_cast<std::size_t>(source_view)];
#endif
            }
        }
    }

    // Evaluate each source-view mask independently.  A component is accepted
    // by relation anchor A or trusted-seed anchor B; neither anchor may be
    // synthesized by another Tier4-v2 component.
    for (int source_view = 0; source_view < 3; ++source_view) {
        cv::Mat labels;
        cv::Mat stats;
        cv::Mat centroids;
        const int component_count = cv::connectedComponentsWithStats(
            tier4_raw_masks[static_cast<std::size_t>(source_view)],
            labels, stats, centroids, 8, CV_32S);
#if defined(_WIN32)
        tier4_component_counts[static_cast<std::size_t>(source_view)] =
            component_count > 0 ? static_cast<std::size_t>(component_count - 1) : 0U;
#endif
        for (int component = 1; component < component_count; ++component) {
            bool component_has_relation_anchor = false;
            bool touches_trusted_seed = false;
            for (int y = 0; y < logical_height_
                && (!touches_trusted_seed || !component_has_relation_anchor); ++y) {
                for (int x = 0; x < logical_width_
                    && (!touches_trusted_seed || !component_has_relation_anchor); ++x) {
                    if (labels.at<int>(y, x) != component) {
                        continue;
                    }
                    if (tier4_relation_anchor_masks[static_cast<std::size_t>(source_view)].at<
                            std::uint8_t>(y, x) != 0U) {
                        component_has_relation_anchor = true;
                    }
                    const ViewObjectObservation* observation = find_object_observation(
                        source_view, x, y);
                    if (observation == nullptr) {
                        continue;
                    }
                    for (int dy = -1; dy <= 1 && !touches_trusted_seed; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            if (dx == 0 && dy == 0) {
                                continue;
                            }
                            const int nx = x + dx;
                            const int ny = y + dy;
                            if (nx < 0 || nx >= logical_width_
                                || ny < 0 || ny >= logical_height_) {
                                continue;
                            }
                            const std::size_t neighbor_cell =
                                static_cast<std::size_t>(ny) * logical_width_ + nx;
                            if (trusted_object_seed_mask[neighbor_cell] == 0U) {
                                continue;
                            }
                            if (std::abs(observation->height
                                    - object_best[neighbor_cell].height)
                                <= height_neighbor_threshold) {
                                touches_trusted_seed = true;
                                break;
                            }
                        }
                    }
                }
            }
            const bool component_has_seed_anchor = touches_trusted_seed;
#if defined(_WIN32)
            if (component_has_relation_anchor) {
                ++tier4_relation_anchor_component_counts[
                    static_cast<std::size_t>(source_view)];
            }
            if (component_has_seed_anchor) {
                ++tier4_seed_anchor_component_counts[static_cast<std::size_t>(source_view)];
            }
            if (component_has_relation_anchor && component_has_seed_anchor) {
                ++tier4_both_anchor_component_counts[static_cast<std::size_t>(source_view)];
            } else if (component_has_relation_anchor) {
                ++tier4_relation_only_component_counts[static_cast<std::size_t>(source_view)];
            } else if (component_has_seed_anchor) {
                ++tier4_seed_only_component_counts[static_cast<std::size_t>(source_view)];
            } else {
                ++tier4_no_anchor_component_counts[static_cast<std::size_t>(source_view)];
            }
#endif
            if (!component_has_relation_anchor && !component_has_seed_anchor) {
#if defined(_WIN32)
                ++tier4_rejected_no_anchor;
#endif
                continue;
            }
            for (int y = 0; y < logical_height_; ++y) {
                for (int x = 0; x < logical_width_; ++x) {
                    if (labels.at<int>(y, x) != component) {
                        continue;
                    }
                    const std::size_t cell = static_cast<std::size_t>(y)
                        * static_cast<std::size_t>(logical_width_)
                        + static_cast<std::size_t>(x);
                    if (object_best[cell].valid) {
                        continue;
                    }
                    ObjectSurfaceCandidate candidate = make_tier4_candidate(
                        source_view, x, y);
                    if (!candidate.valid || (candidate.view_mask &
                            (static_cast<std::uint8_t>(1U) << source_view)) == 0U) {
                        continue;
                    }
                    if (!tier4_best[cell].valid
                        || object_candidate_is_better(candidate, tier4_best[cell])) {
                        tier4_best[cell] = std::move(candidate);
                    }
                }
            }
        }
    }

    // Tier4 only fills cells still empty after the frozen tiers. No Tier4
    // candidate is exposed as a trusted seed during this pass.
    for (std::size_t cell = 0; cell < cell_count; ++cell) {
        if (object_best[cell].valid || !tier4_best[cell].valid) {
            continue;
        }
        object_best[cell] = std::move(tier4_best[cell]);
#if defined(_WIN32)
        object_tier_provenance[cell] = 4U;
        ++tier4_accepted_before_height_neighbor;
#endif
    }

    for (int y = 0; y < logical_height_; ++y) {
        for (int x = 0; x < logical_width_; ++x) {
            const std::size_t cell = static_cast<std::size_t>(y) * logical_width_ + x;
            if (!object_best[cell].valid
                || support_count(object_best[cell].view_mask) != 1) {
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
                    if (trusted_object_seed_mask[neighbor_cell] != 0U
                        && object_best[neighbor_cell].valid) {
                        neighbor_heights.push_back(object_best[neighbor_cell].height);
                    }
                }
            }
            if (!neighbor_heights.empty()
                && std::abs(object_best[cell].height
                    - median_value(std::move(neighbor_heights))) > height_neighbor_threshold) {
                object_best[cell].valid = false;
            }
        }
    }

#if defined(_WIN32)
    if (accepted_fuse_count_ == 0U) {
        cv::Mat final_seed_mask(logical_height_, logical_width_, CV_8UC1,
            cv::Scalar(0));
        std::array<cv::Mat, 3> direct_margin_final_masks;
        std::array<std::size_t, 3> direct_margin_final_counts{};
        for (int view_index = 0; view_index < 3; ++view_index) {
            direct_margin_final_masks[static_cast<std::size_t>(view_index)] = cv::Mat(
                logical_height_, logical_width_, CV_8UC1, cv::Scalar(0));
        }
        std::array<std::size_t, 3> raw_object_counts{};
        std::size_t final_seed_count = 0U;
        for (int y = 0; y < logical_height_; ++y) {
            for (int x = 0; x < logical_width_; ++x) {
                const std::size_t cell = static_cast<std::size_t>(y)
                    * static_cast<std::size_t>(logical_width_) + static_cast<std::size_t>(x);
                for (int view_index = 0; view_index < 3; ++view_index) {
                    if (object_raw_masks[static_cast<std::size_t>(view_index)].at<
                            std::uint8_t>(y, x) != 0U) {
                        ++raw_object_counts[static_cast<std::size_t>(view_index)];
                    }
                }
                if (trusted_object_seed_mask[cell] != 0U) {
                    final_seed_mask.at<std::uint8_t>(y, x) = 255U;
                    ++final_seed_count;
                }
                if (!object_best[cell].valid) {
                    continue;
                }
                for (int view_index = 0; view_index < 3; ++view_index) {
                    const std::uint8_t bit = static_cast<std::uint8_t>(1U) << view_index;
                    if ((object_best[cell].view_mask & bit) == 0U
                        || !object_best[cell].observations[
                            static_cast<std::size_t>(view_index)].direct_margin) {
                        continue;
                    }
                    direct_margin_final_masks[static_cast<std::size_t>(view_index)].at<
                        std::uint8_t>(y, x) = 255U;
                    ++direct_margin_final_counts[static_cast<std::size_t>(view_index)];
                }
            }
        }
        try {
            for (int view_index = 0; view_index < 3; ++view_index) {
                const std::string path = "tmp/object_raw_view"
                    + std::to_string(view_index) + ".png";
                if (!cv::imwrite(path, object_raw_masks[
                        static_cast<std::size_t>(view_index)])) {
                    std::clog << "[WINDOWS_OBJECT_PROVENANCE_MASK] write_failed=1"
                              << std::endl;
                }
            }
            if (!cv::imwrite("tmp/object_final_seed.png", final_seed_mask)) {
                std::clog << "[WINDOWS_OBJECT_PROVENANCE_MASK] write_failed=1"
                          << std::endl;
            }
            if (!cv::imwrite("tmp/direct_margin_final_object.png",
                    make_view_mask_visual(direct_margin_final_masks))) {
                std::clog << "[WINDOWS_DIRECT_MARGIN_MASK] write_failed=1"
                          << std::endl;
            }
        } catch (const cv::Exception&) {
            std::clog << "[WINDOWS_OBJECT_PROVENANCE_MASK] write_failed=1"
                      << std::endl;
        }
        std::clog << "[WINDOWS_OBJECT_PROVENANCE] raw_view0="
                  << raw_object_counts[0]
                  << " raw_view1=" << raw_object_counts[1]
                  << " raw_view2=" << raw_object_counts[2]
                  << " final_seed=" << final_seed_count << std::endl;
        std::clog << "[WINDOWS_DIRECT_MARGIN_FINAL] object_view0="
                  << direct_margin_final_counts[0]
                  << " object_view1=" << direct_margin_final_counts[1]
                  << " object_view2=" << direct_margin_final_counts[2]
                  << " object_total=" << direct_margin_final_counts[0]
                        + direct_margin_final_counts[1]
                        + direct_margin_final_counts[2]
                  << std::endl;
    }
#endif

#if defined(_WIN32)
    if (accepted_fuse_count_ == 0U) {
        std::array<cv::Mat, 3> tier4_final_masks;
        std::array<std::size_t, 3> tier4_raw_counts{};
        std::array<std::size_t, 3> tier4_final_counts{};
        std::size_t tier4_final_count = 0U;
        for (int view_index = 0; view_index < 3; ++view_index) {
            tier4_final_masks[static_cast<std::size_t>(view_index)] = cv::Mat(
                logical_height_, logical_width_, CV_8UC1, cv::Scalar(0));
            tier4_raw_counts[static_cast<std::size_t>(view_index)] = static_cast<std::size_t>(
                cv::countNonZero(tier4_raw_masks[static_cast<std::size_t>(view_index)]));
        }
        for (std::size_t cell = 0; cell < cell_count; ++cell) {
            if (!object_best[cell].valid || object_tier_provenance[cell] != 4U) {
                continue;
            }
            ++tier4_final_count;
            for (int view_index = 0; view_index < 3; ++view_index) {
                const std::uint8_t bit = static_cast<std::uint8_t>(1U) << view_index;
                if ((object_best[cell].view_mask & bit) == 0U) {
                    continue;
                }
                const int x = static_cast<int>(cell % static_cast<std::size_t>(logical_width_));
                const int y = static_cast<int>(cell / static_cast<std::size_t>(logical_width_));
                tier4_final_masks[static_cast<std::size_t>(view_index)].at<
                    std::uint8_t>(y, x) = 255U;
                ++tier4_final_counts[static_cast<std::size_t>(view_index)];
                break;
            }
        }
        const auto make_tier4_visual = [&](const std::array<cv::Mat, 3>& masks) {
            cv::Mat visual(logical_height_, logical_width_, CV_8UC3,
                cv::Scalar(0, 0, 0));
            for (int y = 0; y < logical_height_; ++y) {
                for (int x = 0; x < logical_width_; ++x) {
                    cv::Vec3b& pixel = visual.at<cv::Vec3b>(y, x);
                    if (masks[0].at<std::uint8_t>(y, x) != 0U) {
                        pixel[2] = 255U; // view0 = red
                    }
                    if (masks[1].at<std::uint8_t>(y, x) != 0U) {
                        pixel[1] = 255U; // view1 = green
                    }
                    if (masks[2].at<std::uint8_t>(y, x) != 0U) {
                        pixel[0] = 255U; // view2 = blue
                    }
                }
            }
            cv::Mat resized;
            cv::resize(visual, resized, cv::Size(canvas_width_, canvas_height_),
                0.0, 0.0, cv::INTER_NEAREST);
            return resized;
        };
        try {
            if (!cv::imwrite("tmp/object_tier4_raw.png", make_tier4_visual(
                    tier4_raw_masks))
                || !cv::imwrite("tmp/object_tier4_accepted.png", make_tier4_visual(
                    tier4_final_masks))) {
                std::clog << "[WINDOWS_OBJECT_TIER4_MASK] write_failed=1"
                          << std::endl;
            }
        } catch (const cv::Exception&) {
            std::clog << "[WINDOWS_OBJECT_TIER4_MASK] write_failed=1"
                      << std::endl;
        }
        std::clog << "[WINDOWS_OBJECT_TIER4] raw_view0=" << tier4_raw_counts[0]
                  << " raw_view1=" << tier4_raw_counts[1]
                  << " raw_view2=" << tier4_raw_counts[2]
                  << " components_view0=" << tier4_component_counts[0]
                  << " components_view1=" << tier4_component_counts[1]
                  << " components_view2=" << tier4_component_counts[2]
                  << " relation_anchor_cells_view0=" << tier4_relation_anchor_cell_counts[0]
                  << " relation_anchor_cells_view1=" << tier4_relation_anchor_cell_counts[1]
                  << " relation_anchor_cells_view2=" << tier4_relation_anchor_cell_counts[2]
                  << " relation_anchor_components_view0="
                  << tier4_relation_anchor_component_counts[0]
                  << " relation_anchor_components_view1="
                  << tier4_relation_anchor_component_counts[1]
                  << " relation_anchor_components_view2="
                  << tier4_relation_anchor_component_counts[2]
                  << " seed_anchor_components_view0="
                  << tier4_seed_anchor_component_counts[0]
                  << " seed_anchor_components_view1="
                  << tier4_seed_anchor_component_counts[1]
                  << " seed_anchor_components_view2="
                  << tier4_seed_anchor_component_counts[2]
                  << " relation_only_view0=" << tier4_relation_only_component_counts[0]
                  << " relation_only_view1=" << tier4_relation_only_component_counts[1]
                  << " relation_only_view2=" << tier4_relation_only_component_counts[2]
                  << " seed_only_view0=" << tier4_seed_only_component_counts[0]
                  << " seed_only_view1=" << tier4_seed_only_component_counts[1]
                  << " seed_only_view2=" << tier4_seed_only_component_counts[2]
                  << " both_view0=" << tier4_both_anchor_component_counts[0]
                  << " both_view1=" << tier4_both_anchor_component_counts[1]
                  << " both_view2=" << tier4_both_anchor_component_counts[2]
                  << " rejected_contradict=" << tier4_rejected_contradict
                  << " rejected_no_anchor=" << tier4_rejected_no_anchor
                  << " accepted_before_height_neighbor="
                  << tier4_accepted_before_height_neighbor
                  << " accepted_final=" << tier4_final_count
                  << " final_view0=" << tier4_final_counts[0]
                  << " final_view1=" << tier4_final_counts[1]
                  << " final_view2=" << tier4_final_counts[2] << std::endl;
    }
#endif

#if defined(_WIN32)
    // Final direct-visibility accounting for the Tier4-v2 pipeline.  The
    // before-v2 numbers are emitted by the diagnostic-only run immediately
    // preceding this change; this block records the after-v2 retention.
    if (accepted_fuse_count_ == 0U) {
        std::array<std::size_t, 3> direct_object_counts{};
        std::array<std::size_t, 3> direct_object_only_counts{};
        std::array<std::size_t, 3> direct_object_only_kept_counts{};
        std::array<std::size_t, 3> direct_object_only_lost_counts{};
        std::array<std::size_t, 8> direct_object_pattern_counts{};
        cv::Mat direct_object_union(logical_height_, logical_width_, CV_8UC1,
            cv::Scalar(0));
        cv::Mat lost_object_direct_after(logical_height_, logical_width_, CV_8UC1,
            cv::Scalar(0));
        std::size_t direct_union_count = 0U;
        std::size_t lost_direct_count = 0U;
        for (std::size_t cell = 0; cell < cell_count; ++cell) {
            const int x = static_cast<int>(cell % static_cast<std::size_t>(logical_width_));
            const int y = static_cast<int>(cell / static_cast<std::size_t>(logical_width_));
            std::uint8_t pattern = 0U;
            for (int view_index = 0; view_index < 3; ++view_index) {
                if (direct_object_presence_by_view[static_cast<std::size_t>(view_index)][cell]
                    == 0U) {
                    continue;
                }
                ++direct_object_counts[static_cast<std::size_t>(view_index)];
                pattern |= static_cast<std::uint8_t>(1U) << view_index;
            }
            if (pattern == 0U) {
                continue;
            }
            ++direct_object_pattern_counts[pattern];
            direct_object_union.at<std::uint8_t>(y, x) = 255U;
            ++direct_union_count;
            if (support_count(pattern) == 1) {
                for (int view_index = 0; view_index < 3; ++view_index) {
                    if ((pattern & (static_cast<std::uint8_t>(1U) << view_index)) == 0U) {
                        continue;
                    }
                    ++direct_object_only_counts[static_cast<std::size_t>(view_index)];
                    if (object_best[cell].valid) {
                        ++direct_object_only_kept_counts[static_cast<std::size_t>(view_index)];
                    } else {
                        ++direct_object_only_lost_counts[static_cast<std::size_t>(view_index)];
                    }
                }
            }
            if (!object_best[cell].valid) {
                lost_object_direct_after.at<std::uint8_t>(y, x) = 255U;
                ++lost_direct_count;
            }
        }
        std::clog << "[WINDOWS_DIRECT_OBJECT_CANDIDATE] view0=" << direct_object_counts[0]
                  << " view1=" << direct_object_counts[1]
                  << " view2=" << direct_object_counts[2]
                  << " candidate_union=" << direct_union_count
                  << " only_view0=" << direct_object_only_counts[0]
                  << " only_view1=" << direct_object_only_counts[1]
                  << " only_view2=" << direct_object_only_counts[2]
                  << " only_view0_kept=" << direct_object_only_kept_counts[0]
                  << " only_view1_kept=" << direct_object_only_kept_counts[1]
                  << " only_view2_kept=" << direct_object_only_kept_counts[2]
                  << " only_view0_lost=" << direct_object_only_lost_counts[0]
                  << " only_view1_lost=" << direct_object_only_lost_counts[1]
                  << " only_view2_lost=" << direct_object_only_lost_counts[2]
                  << " candidate_not_selected_final=" << lost_direct_count
                  << " pattern1=" << direct_object_pattern_counts[1]
                  << " pattern2=" << direct_object_pattern_counts[2]
                  << " pattern3=" << direct_object_pattern_counts[3]
                  << " pattern4=" << direct_object_pattern_counts[4]
                  << " pattern5=" << direct_object_pattern_counts[5]
                  << " pattern6=" << direct_object_pattern_counts[6]
                  << " pattern7=" << direct_object_pattern_counts[7] << std::endl;
        try {
            std::array<cv::Mat, 3> direct_object_masks;
            for (int view_index = 0; view_index < 3; ++view_index) {
                direct_object_masks[static_cast<std::size_t>(view_index)] =
                    object_raw_masks[static_cast<std::size_t>(view_index)];
            }
            if (!cv::imwrite("tmp/direct_object_union.png",
                    make_view_mask_visual(direct_object_masks))
                || !cv::imwrite("tmp/lost_object_direct_after.png",
                    resize_logical_mask(lost_object_direct_after))) {
                std::clog << "[WINDOWS_DIRECT_OBJECT_MASK] write_failed=1" << std::endl;
            }
        } catch (const cv::Exception&) {
            std::clog << "[WINDOWS_DIRECT_OBJECT_MASK] write_failed=1" << std::endl;
        }
    }
#endif

    // Complete only empty floor cells after the final object filtering. The
    // fitted plane supplies geometry, while the existing multi-view
    // reprojection relation decides whether that geometry is justified.
    std::vector<std::uint8_t> final_object_mask(cell_count, 0U);
    for (std::size_t cell = 0; cell < cell_count; ++cell) {
        if (object_best[cell].valid) {
            final_object_mask[cell] = 1U;
        }
    }

    std::vector<std::uint8_t> exterior_empty(cell_count, 0U);
    std::vector<std::size_t> flood_queue;
    flood_queue.reserve(cell_count);
    const auto enqueue_empty_boundary = [&](const int x, const int y) {
        const std::size_t cell = static_cast<std::size_t>(y) * logical_width_ + x;
        if (current_observed_floor_mask[cell] == 0U && exterior_empty[cell] == 0U) {
            exterior_empty[cell] = 1U;
            flood_queue.push_back(cell);
        }
    };
    for (int x = 0; x < logical_width_; ++x) {
        enqueue_empty_boundary(x, 0);
        enqueue_empty_boundary(x, logical_height_ - 1);
    }
    for (int y = 1; y + 1 < logical_height_; ++y) {
        enqueue_empty_boundary(0, y);
        enqueue_empty_boundary(logical_width_ - 1, y);
    }
    for (std::size_t queue_index = 0U; queue_index < flood_queue.size(); ++queue_index) {
        const std::size_t cell = flood_queue[queue_index];
        const int x = static_cast<int>(cell % static_cast<std::size_t>(logical_width_));
        const int y = static_cast<int>(cell / static_cast<std::size_t>(logical_width_));
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
                enqueue_empty_boundary(nx, ny);
            }
        }
    }

    std::vector<std::uint8_t> floor_completion_tier(cell_count, 0U);
    std::vector<cv::Vec3f> floor_completion_colors(
        cell_count, cv::Vec3f(0.0f, 0.0f, 0.0f));
    std::vector<float> floor_completion_confidences(cell_count, 0.0f);
    std::vector<std::uint8_t> floor_completion_color_valid(cell_count, 0U);
#if defined(_WIN32)
    std::size_t completion_empty_before = 0U;
    std::size_t completion_empty_in_bounds = 0U;
    std::size_t completion_interior_holes = 0U;
    std::size_t completion_interior_holes_in_bounds = 0U;
    std::size_t completion_final_object_cells = 0U;
    std::size_t completion_relation_support = 0U;
    std::size_t completion_relation_occluded = 0U;
    std::size_t completion_relation_contradict = 0U;
    std::size_t completion_relation_outside = 0U;
    std::size_t completion_relation_invalid = 0U;
    std::size_t completion_f1_candidates = 0U;
    std::size_t completion_f2_candidates = 0U;
    std::size_t completion_f3_candidates = 0U;
    std::size_t completion_f3_object_underlay = 0U;
    std::size_t completion_f3_interior_hole = 0U;
    std::size_t completion_object_underlay_candidates = 0U;
    std::size_t completion_blocked_by_contradict = 0U;
    std::size_t completion_rejected_outside_invalid = 0U;
    std::size_t completion_rejected_gate = 0U;
#endif
    for (int y = 0; y < logical_height_; ++y) {
        for (int x = 0; x < logical_width_; ++x) {
            const std::size_t cell = static_cast<std::size_t>(y) * logical_width_ + x;
            if (floor_cell_valid_[cell] != 0U) {
                continue;
            }
#if defined(_WIN32)
            if (accepted_fuse_count_ == 0U) {
                ++completion_empty_before;
                if (exterior_empty[cell] == 0U) {
                    ++completion_interior_holes;
                }
            }
#endif
            const cv::Vec3f plane_point = cell_to_reference_floor_point(x, y);
            const cv::Vec3f delta = plane_point - plane_origin_;
            const float u = axis_x_.dot(delta);
            const float v = axis_y_.dot(delta);
            if (u < u_min_ || u > u_max_ || v < v_min_ || v > v_max_) {
                continue;
            }
#if defined(_WIN32)
            if (accepted_fuse_count_ == 0U) {
                ++completion_empty_in_bounds;
                if (exterior_empty[cell] == 0U) {
                    ++completion_interior_holes_in_bounds;
                }
                if (final_object_mask[cell] != 0U) {
                    ++completion_final_object_cells;
                }
            }
#endif

            std::array<ReprojectionResult, 3> reprojections{};
            std::size_t support = 0U;
            std::size_t occluded = 0U;
            std::size_t contradict = 0U;
            std::size_t outside = 0U;
            std::size_t invalid = 0U;
            for (int view_index = 0; view_index < 3; ++view_index) {
                ReprojectionResult& reprojection =
                    reprojections[static_cast<std::size_t>(view_index)];
                reprojection = project_world_point_to_view(
                    plane_point, transforms[static_cast<std::size_t>(view_index)],
                    views[static_cast<std::size_t>(view_index)],
                    reprojection_world_tolerance, reprojection_depth_tolerance);
                switch (reprojection.relation) {
                case ReprojectionRelation::Support:
                    ++support;
#if defined(_WIN32)
                    if (accepted_fuse_count_ == 0U) {
                        ++completion_relation_support;
                    }
#endif
                    break;
                case ReprojectionRelation::Occluded:
                    ++occluded;
#if defined(_WIN32)
                    if (accepted_fuse_count_ == 0U) {
                        ++completion_relation_occluded;
                    }
#endif
                    break;
                case ReprojectionRelation::Contradict:
                    ++contradict;
#if defined(_WIN32)
                    if (accepted_fuse_count_ == 0U) {
                        ++completion_relation_contradict;
                    }
#endif
                    break;
                case ReprojectionRelation::Outside:
                    ++outside;
#if defined(_WIN32)
                    if (accepted_fuse_count_ == 0U) {
                        ++completion_relation_outside;
                    }
#endif
                    break;
                case ReprojectionRelation::Invalid:
                    ++invalid;
#if defined(_WIN32)
                    if (accepted_fuse_count_ == 0U) {
                        ++completion_relation_invalid;
                    }
#endif
                    break;
                }
            }

            std::uint8_t tier = 0U;
            if (contradict == 0U && support >= 2U) {
                tier = 1U;
            } else if (contradict == 0U && support >= 1U
                && support + occluded >= 2U) {
                tier = 2U;
            } else if (contradict == 0U && support == 0U && occluded >= 2U
                && (final_object_mask[cell] != 0U || exterior_empty[cell] == 0U)) {
                tier = 3U;
            }
#if defined(_WIN32)
            if (accepted_fuse_count_ == 0U) {
                if (contradict > 0U) {
                    ++completion_blocked_by_contradict;
                } else if (tier == 1U) {
                    ++completion_f1_candidates;
                } else if (tier == 2U) {
                    ++completion_f2_candidates;
                } else if (tier == 3U) {
                    ++completion_f3_candidates;
                    if (final_object_mask[cell] != 0U) {
                        ++completion_f3_object_underlay;
                    }
                    if (exterior_empty[cell] == 0U) {
                        ++completion_f3_interior_hole;
                    }
                } else if (outside + invalid > 0U) {
                    ++completion_rejected_outside_invalid;
                } else {
                    ++completion_rejected_gate;
                }
                if (tier != 0U && final_object_mask[cell] != 0U) {
                    ++completion_object_underlay_candidates;
                }
            }
#endif
            if (tier == 0U) {
                continue;
            }
            floor_completion_tier[cell] = tier;

            if (tier == 1U || tier == 2U) {
                cv::Vec3f color(0.0f, 0.0f, 0.0f);
                float color_weight = 0.0f;
                float confidence_weight = 0.0f;
                float total_weight = 0.0f;
                for (int view_index = 0; view_index < 3; ++view_index) {
                    const ReprojectionResult& reprojection =
                        reprojections[static_cast<std::size_t>(view_index)];
                    if (reprojection.relation != ReprojectionRelation::Support) {
                        continue;
                    }
                    const GroupWorldView& view = views[static_cast<std::size_t>(view_index)];
                    const int pixel_x = std::clamp(
                        static_cast<int>(std::lround(reprojection.pixel_x)),
                        0, view.rgb.cols - 1);
                    const int pixel_y = std::clamp(
                        static_cast<int>(std::lround(reprojection.pixel_y)),
                        0, view.rgb.rows - 1);
                    const cv::Vec3f sample_color = view.rgb.at<cv::Vec3f>(pixel_y, pixel_x);
                    const float sample_confidence =
                        view.world_confidence.at<float>(pixel_y, pixel_x);
                    if (!finite_vec(sample_color) || !std::isfinite(sample_confidence)) {
                        continue;
                    }
                    const ConfidenceStats& confidence_range =
                        confidence_stats[static_cast<std::size_t>(view_index)];
                    const float normalized = normalized_confidence(
                        sample_confidence, confidence_range.q10, confidence_range.q90);
                    const float weight = std::max(
                        normalized * border_weight(
                            pixel_x, pixel_y, view.rgb.cols, view.rgb.rows),
                        1e-4f);
                    const auto& gain = color_gain_[static_cast<std::size_t>(view_index)];
                    const cv::Vec3f corrected(
                        std::clamp(sample_color[0] * gain[0], 0.0f, 1.0f),
                        std::clamp(sample_color[1] * gain[1], 0.0f, 1.0f),
                        std::clamp(sample_color[2] * gain[2], 0.0f, 1.0f));
                    if (!finite_vec(corrected)) {
                        continue;
                    }
                    color += corrected * weight;
                    color_weight += weight;
                    total_weight += weight;
                    confidence_weight += normalized * weight;
                }
                if (color_weight > kNumericEpsilon) {
                    floor_completion_colors[cell] = color * (1.0f / color_weight);
                    floor_completion_color_valid[cell] = 1U;
                }
                if (total_weight > kNumericEpsilon) {
                    floor_completion_confidences[cell] = std::clamp(
                        confidence_weight / total_weight, 0.0f, 1.0f);
                }
            }
        }
    }

    // Completion geometry is already accepted above. Texture lookup is a
    // separate pass: find the nearest observed floor source over the entire
    // logical grid, without using the lookup to create geometry.
    std::vector<int> nearest_floor_source(cell_count, -1);
    std::vector<std::size_t> nearest_floor_queue;
    nearest_floor_queue.reserve(cell_count);
    for (std::size_t cell = 0; cell < cell_count; ++cell) {
        if (current_observed_floor_mask[cell] == 0U
            || floor_cell_valid_[cell] == 0U) {
            continue;
        }
        const auto channels = unpack_rgba(floor_cells_[cell].rgba);
        if (channels[0] == 0U && channels[1] == 0U && channels[2] == 0U) {
            continue;
        }
        nearest_floor_source[cell] = static_cast<int>(cell);
        nearest_floor_queue.push_back(cell);
    }
    if (nearest_floor_queue.empty()) {
        for (std::size_t cell = 0; cell < cell_count; ++cell) {
            if (floor_cell_valid_[cell] == 0U) {
                continue;
            }
            const auto channels = unpack_rgba(floor_cells_[cell].rgba);
            if (channels[0] == 0U && channels[1] == 0U && channels[2] == 0U) {
                continue;
            }
            nearest_floor_source[cell] = static_cast<int>(cell);
            nearest_floor_queue.push_back(cell);
        }
    }
    for (std::size_t queue_index = 0U; queue_index < nearest_floor_queue.size(); ++queue_index) {
        const std::size_t cell = nearest_floor_queue[queue_index];
        const int x = static_cast<int>(cell % static_cast<std::size_t>(logical_width_));
        const int y = static_cast<int>(cell / static_cast<std::size_t>(logical_width_));
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
                const std::size_t neighbor =
                    static_cast<std::size_t>(ny) * logical_width_ + nx;
                if (nearest_floor_source[neighbor] >= 0) {
                    continue;
                }
                nearest_floor_source[neighbor] = nearest_floor_source[cell];
                nearest_floor_queue.push_back(neighbor);
            }
        }
    }

#if defined(_WIN32)
    std::size_t completion_observed_floor = static_cast<std::size_t>(std::count(
        floor_cell_valid_.begin(), floor_cell_valid_.end(), static_cast<std::uint8_t>(1U)));
    std::size_t completion_f1_added = 0U;
    std::size_t completion_f2_added = 0U;
    std::size_t completion_f3_added = 0U;
    std::size_t completion_geometry_without_direct_rgb = 0U;
    std::size_t completion_geometry_dropped_due_rgb = 0U;
    std::size_t completion_direct_rgb_cells = 0U;
    std::size_t completion_propagated_rgb_cells = 0U;
    std::size_t completion_without_texture_source = 0U;
#endif
    for (std::size_t cell = 0; cell < cell_count; ++cell) {
        const std::uint8_t tier = floor_completion_tier[cell];
        if (tier == 0U || floor_cell_valid_[cell] != 0U) {
            continue;
        }
        std::uint32_t rgba = 0U;
        float confidence = 0.0f;
        const bool has_direct_color = (tier == 1U || tier == 2U)
            && floor_completion_color_valid[cell] != 0U;
        if (tier == 1U || tier == 2U) {
            if (has_direct_color) {
                rgba = color_to_rgba(floor_completion_colors[cell]);
                confidence = floor_completion_confidences[cell];
            } else if (nearest_floor_source[cell] >= 0) {
                const std::size_t source = static_cast<std::size_t>(nearest_floor_source[cell]);
                rgba = floor_cells_[source].rgba;
                confidence = floor_cells_[source].confidence;
            }
        } else {
            if (nearest_floor_source[cell] >= 0) {
                const std::size_t source = static_cast<std::size_t>(nearest_floor_source[cell]);
                rgba = floor_cells_[source].rgba;
                confidence = floor_cells_[source].confidence;
            }
        }
#if defined(_WIN32)
        const bool has_propagated_color = !has_direct_color
            && nearest_floor_source[cell] >= 0;
        if (!has_direct_color) {
            ++completion_geometry_without_direct_rgb;
        }
        if (has_direct_color) {
            ++completion_direct_rgb_cells;
        } else if (has_propagated_color) {
            ++completion_propagated_rgb_cells;
        } else {
            ++completion_without_texture_source;
        }
#endif
        const int logical_x = static_cast<int>(
            cell % static_cast<std::size_t>(logical_width_));
        const int logical_y = static_cast<int>(
            cell / static_cast<std::size_t>(logical_width_));
        const std::uint32_t slot_id = slot_for(logical_x, logical_y, 0);
        if (slot_id == std::numeric_limits<std::uint32_t>::max()) {
            continue;
        }
        FusedSlot next = floor_cells_[cell];
        next.slot_id = slot_id;
        next.depth = 0.0f;
        next.confidence = confidence;
        next.rgba = rgba;
        next.floor = true;
        floor_cells_[cell] = next;
        floor_cell_valid_[cell] = 1U;
        result.slots.push_back(next);
        result.occupied_slots.push_back(slot_id);
#if defined(_WIN32)
        if (tier == 1U) {
            ++completion_f1_added;
        } else if (tier == 2U) {
            ++completion_f2_added;
        } else {
            ++completion_f3_added;
        }
#endif
    }
#if defined(_WIN32)
    const std::size_t completion_floor_total = static_cast<std::size_t>(std::count(
        floor_cell_valid_.begin(), floor_cell_valid_.end(), static_cast<std::uint8_t>(1U)));
    std::clog << "[WINDOWS_FLOOR_COMPLETION_RESULT] group=" << accepted_fuse_count_
              << " observed_floor=" << completion_observed_floor
              << " F1_added=" << completion_f1_added
              << " F2_added=" << completion_f2_added
              << " F3_added=" << completion_f3_added
              << " floor_total=" << completion_floor_total << std::endl;
    if (accepted_fuse_count_ == 0U) {
        std::clog << "[WINDOWS_FLOOR_COMPLETION_DIAG] empty_before="
                  << completion_empty_before
                  << " empty_in_bounds=" << completion_empty_in_bounds
                  << " interior_holes=" << completion_interior_holes
                  << " interior_holes_in_bounds="
                  << completion_interior_holes_in_bounds
                  << " final_object_cells=" << completion_final_object_cells
                  << " object_underlay_candidates="
                  << completion_object_underlay_candidates << std::endl;
        std::clog << "[WINDOWS_FLOOR_COMPLETION_RELATION] Support="
                  << completion_relation_support
                  << " Occluded=" << completion_relation_occluded
                  << " Contradict=" << completion_relation_contradict
                  << " Outside=" << completion_relation_outside
                  << " Invalid=" << completion_relation_invalid << std::endl;
        std::clog << "[WINDOWS_FLOOR_COMPLETION_GATE] F1="
                  << completion_f1_candidates
                  << " F2=" << completion_f2_candidates
                  << " F3=" << completion_f3_candidates
                  << " F3_object_underlay=" << completion_f3_object_underlay
                  << " F3_interior_hole=" << completion_f3_interior_hole
                  << " blocked_by_contradict="
                  << completion_blocked_by_contradict
                  << " rejected_outside_invalid="
                  << completion_rejected_outside_invalid
                  << " rejected_gate=" << completion_rejected_gate << std::endl;
    }
#endif

    struct RayPlaneCellSupport {
        std::array<std::size_t, 3> support_pixels{};
        std::array<std::size_t, 3> contradict_pixels{};
        std::array<cv::Vec3f, 3> direct_floor_color_sum{};
        std::array<float, 3> direct_floor_color_weight{};
        std::array<float, 3> direct_floor_confidence_sum{};
    };
    std::vector<RayPlaneCellSupport> ray_plane_support(cell_count);
#if defined(_WIN32)
    std::size_t ray_valid_pixels = 0U;
    std::size_t ray_plane_intersections = 0U;
    std::size_t ray_physical_gate_rejected = 0U;
    std::size_t ray_uv_reject = 0U;
    std::size_t ray_pixel_x_reject = 0U;
    std::size_t ray_pixel_y_reject = 0U;
    std::array<std::size_t, 4> ray_uv_directional{};
    std::size_t ray_support_pixels = 0U;
    std::size_t ray_contradict_pixels = 0U;
#endif
    for (int view_index = 0; view_index < 3; ++view_index) {
        const GroupWorldView& view = views[static_cast<std::size_t>(view_index)];
        const cv::Mat& aligned_points = aligned_world_points[static_cast<std::size_t>(view_index)];
        const cv::Vec3f camera = reference_camera_centers[static_cast<std::size_t>(view_index)];
        for (int y = 0; y < view.world_points.rows; ++y) {
            for (int x = 0; x < view.world_points.cols; ++x) {
                const cv::Vec3f point = aligned_points.at<cv::Vec3f>(y, x);
                if (!finite_vec(point) || !finite_vec(camera)) {
                    continue;
                }
                const cv::Vec3f ray = point - camera;
                const float ray_length = vector_norm(ray);
                if (!std::isfinite(ray_length) || ray_length <= kNumericEpsilon) {
                    continue;
                }
#if defined(_WIN32)
                ++ray_valid_pixels;
#endif
                const float denominator = plane_normal_.dot(ray);
                const float numerator = plane_normal_.dot(plane_origin_ - camera);
                if (!std::isfinite(denominator) || !std::isfinite(numerator)
                    || std::abs(denominator) <= kNumericEpsilon) {
                    continue;
                }
                const float t_plane = numerator / denominator;
                if (!std::isfinite(t_plane) || t_plane <= 0.0f) {
                    continue;
                }
                const cv::Vec3f plane_point = camera + ray * t_plane;
                if (!finite_vec(plane_point)) {
                    continue;
                }
#if defined(_WIN32)
                ++ray_plane_intersections;
#endif
                const cv::Vec3f point_delta = point - plane_origin_;
                const float height = plane_normal_.dot(point_delta);
                if (!std::isfinite(height)
                    || height < -2.5f * floor_band_
                    || height >= max_object_height_) {
#if defined(_WIN32)
                    ++ray_physical_gate_rejected;
#endif
                    continue;
                }
                const cv::Vec3f plane_delta = plane_point - plane_origin_;
                const float plane_u = axis_x_.dot(plane_delta);
                const float plane_v = axis_y_.dot(plane_delta);
                int logical_x = 0;
                int logical_y = 0;
                if (!world_to_cell(plane_u, plane_v, logical_x, logical_y)) {
#if defined(_WIN32)
                    const int reject_reason = world_to_cell_reject_reason(plane_u, plane_v);
                    if (reject_reason == 1) {
                        ++ray_uv_reject;
                        const int flags = world_to_cell_uv_reject_flags(plane_u, plane_v);
                        if ((flags & 1) != 0) {
                            ++ray_uv_directional[0];
                        }
                        if ((flags & 2) != 0) {
                            ++ray_uv_directional[1];
                        }
                        if ((flags & 4) != 0) {
                            ++ray_uv_directional[2];
                        }
                        if ((flags & 8) != 0) {
                            ++ray_uv_directional[3];
                        }
                    } else if (reject_reason == 2) {
                        ++ray_pixel_x_reject;
                    } else if (reject_reason == 3) {
                        ++ray_pixel_y_reject;
                    }
#endif
                    continue;
                }
                const float observed_distance = vector_norm(point - camera);
                const float plane_distance = vector_norm(plane_point - camera);
                if (!std::isfinite(observed_distance) || !std::isfinite(plane_distance)) {
                    continue;
                }
                const std::size_t cell = static_cast<std::size_t>(logical_y)
                    * static_cast<std::size_t>(logical_width_)
                    + static_cast<std::size_t>(logical_x);
                RayPlaneCellSupport& support = ray_plane_support[cell];
                const bool is_support = plane_distance
                    >= observed_distance - reprojection_depth_tolerance;
                if (is_support) {
                    ++support.support_pixels[static_cast<std::size_t>(view_index)];
#if defined(_WIN32)
                    ++ray_support_pixels;
#endif
                } else {
                    ++support.contradict_pixels[static_cast<std::size_t>(view_index)];
#if defined(_WIN32)
                    ++ray_contradict_pixels;
#endif
                    continue;
                }

                const float confidence = view.world_confidence.at<float>(y, x);
                const cv::Vec3f sample_color = view.rgb.at<cv::Vec3f>(y, x);
                if (std::abs(height) <= 2.5f * floor_band_
                    && std::isfinite(confidence) && finite_vec(sample_color)) {
                    const ConfidenceStats& confidence_range =
                        confidence_stats[static_cast<std::size_t>(view_index)];
                    const float normalized = normalized_confidence(
                        confidence, confidence_range.q10, confidence_range.q90);
                    const float weight = std::max(
                        normalized * border_weight(
                            x, y, view.rgb.cols, view.rgb.rows),
                        1e-4f);
                    const auto& gain = color_gain_[static_cast<std::size_t>(view_index)];
                    const cv::Vec3f corrected(
                        std::clamp(sample_color[0] * gain[0], 0.0f, 1.0f),
                        std::clamp(sample_color[1] * gain[1], 0.0f, 1.0f),
                        std::clamp(sample_color[2] * gain[2], 0.0f, 1.0f));
                    if (finite_vec(corrected)) {
                        support.direct_floor_color_sum[static_cast<std::size_t>(view_index)]
                            += corrected * weight;
                        support.direct_floor_color_weight[static_cast<std::size_t>(view_index)]
                            += weight;
                        support.direct_floor_confidence_sum[static_cast<std::size_t>(view_index)]
                            += normalized * weight;
                    }
                }
            }
        }
    }

    struct RayCellClassification {
        std::size_t support_views = 0U;
        std::size_t contradict_views = 0U;
        int sole_support_view = -1;
    };
    const auto classify_ray_cell = [](const RayPlaneCellSupport& support) {
        RayCellClassification classification;
        for (int view_index = 0; view_index < 3; ++view_index) {
            const std::size_t view = static_cast<std::size_t>(view_index);
            if (support.support_pixels[view] > support.contradict_pixels[view]
                && support.support_pixels[view] > 0U) {
                ++classification.support_views;
                classification.sole_support_view = view_index;
            } else if (support.contradict_pixels[view]
                > support.support_pixels[view]
                && support.contradict_pixels[view] > 0U) {
                ++classification.contradict_views;
            }
        }
        if (classification.support_views != 1U) {
            classification.sole_support_view = -1;
        }
        return classification;
    };

#if defined(_WIN32)
    if (accepted_fuse_count_ == 0U) {
        std::array<std::size_t, 3> ray_support_cells{};
        std::size_t ray_support_cells_overlap_observed_floor = 0U;
        std::size_t ray_support_cells_empty = 0U;
        std::array<std::size_t, 4> empty_support_view_counts{};
        std::size_t single_view_empty_total = 0U;
        std::size_t single_view_direct_floor_total = 0U;
        std::size_t single_view_no_direct_floor_total = 0U;
        std::size_t ray_r1_candidates = 0U;
        std::size_t ray_r2_candidates = 0U;
        cv::Mat support_views2_mask(logical_height_, logical_width_, CV_8UC1, cv::Scalar(0));
        cv::Mat ray_candidates_mask(logical_height_, logical_width_, CV_8UC1, cv::Scalar(0));
        for (int y = 0; y < logical_height_; ++y) {
            for (int x = 0; x < logical_width_; ++x) {
                const std::size_t cell = static_cast<std::size_t>(y)
                    * static_cast<std::size_t>(logical_width_)
                    + static_cast<std::size_t>(x);
                const RayPlaneCellSupport& support = ray_plane_support[cell];
                const RayCellClassification classification = classify_ray_cell(support);
                const std::size_t support_views = classification.support_views;
                const std::size_t contradict_views = classification.contradict_views;
                for (int view_index = 0; view_index < 3; ++view_index) {
                    const std::size_t view = static_cast<std::size_t>(view_index);
                    if (support.support_pixels[view] > support.contradict_pixels[view]
                        && support.support_pixels[view] > 0U) {
                        ++ray_support_cells[view];
                    }
                }
                if (support_views >= 2U) {
                    support_views2_mask.at<std::uint8_t>(y, x) = 255U;
                }
                if (support_views > 0U && current_observed_floor_mask[cell] != 0U) {
                    ++ray_support_cells_overlap_observed_floor;
                }
                if (support_views > 0U && floor_cell_valid_[cell] == 0U) {
                    ++ray_support_cells_empty;
                }
                if (floor_cell_valid_[cell] != 0U) {
                    continue;
                }
                ++empty_support_view_counts[std::min<std::size_t>(support_views, 3U)];
                if (support_views == 1U) {
                    ++single_view_empty_total;
                    const std::size_t sole_view = static_cast<std::size_t>(
                        classification.sole_support_view);
                    if (support.direct_floor_color_weight[sole_view]
                        > kNumericEpsilon) {
                        ++single_view_direct_floor_total;
                    } else {
                        ++single_view_no_direct_floor_total;
                    }
                }
                const bool interior_hole = exterior_empty[cell] == 0U;
                const bool no_contradiction = contradict_views == 0U;
                const bool r1 = no_contradiction && support_views >= 2U;
                const bool r2 = no_contradiction && support_views == 1U
                    && (final_object_mask[cell] != 0U || interior_hole);
                if (r1) {
                    ++ray_r1_candidates;
                }
                if (r2) {
                    ++ray_r2_candidates;
                }
                if (r1 || r2) {
                    ray_candidates_mask.at<std::uint8_t>(y, x) = 255U;
                }
            }
        }
        cv::Mat support_views2_visual;
        cv::Mat ray_candidates_visual;
        cv::resize(support_views2_mask, support_views2_visual,
            cv::Size(canvas_width_, canvas_height_), 0.0, 0.0, cv::INTER_NEAREST);
        cv::resize(ray_candidates_mask, ray_candidates_visual,
            cv::Size(canvas_width_, canvas_height_), 0.0, 0.0, cv::INTER_NEAREST);
        try {
            cv::imwrite("tmp/ray_plane_support_views2.png", support_views2_visual);
            cv::imwrite("tmp/ray_plane_R1_R2_candidates.png", ray_candidates_visual);
        } catch (const cv::Exception&) {
            std::clog << "[WINDOWS_RAY_PLANE_MASK] write_failed=1" << std::endl;
        }
        std::clog << "[WINDOWS_RAY_PLANE_DIAG] valid_ray_pixels="
                  << ray_valid_pixels
                  << " plane_intersections=" << ray_plane_intersections
                  << " physical_gate_rejected=" << ray_physical_gate_rejected
                  << " support_pixels=" << ray_support_pixels
                  << " contradict_pixels=" << ray_contradict_pixels
                  << " ray_uv_reject=" << ray_uv_reject
                  << " ray_uv_u_low=" << ray_uv_directional[0]
                  << " ray_uv_u_high=" << ray_uv_directional[1]
                  << " ray_uv_v_low=" << ray_uv_directional[2]
                  << " ray_uv_v_high=" << ray_uv_directional[3]
                  << " ray_pixel_x_reject=" << ray_pixel_x_reject
                  << " ray_pixel_y_reject=" << ray_pixel_y_reject
                  << " intersections_outside_atlas="
                  << (ray_uv_reject + ray_pixel_x_reject + ray_pixel_y_reject)
                  << std::endl;
        std::clog << "[WINDOWS_RAY_PLANE_CELLS] support_view0=" << ray_support_cells[0]
                  << " support_view1=" << ray_support_cells[1]
                  << " support_view2=" << ray_support_cells[2]
                  << " overlap_observed_floor="
                  << ray_support_cells_overlap_observed_floor
                  << " support_empty=" << ray_support_cells_empty << std::endl;
        std::clog << "[WINDOWS_RAY_PLANE_EMPTY] support_views0="
                  << empty_support_view_counts[0]
                  << " support_views1=" << empty_support_view_counts[1]
                  << " support_views2=" << empty_support_view_counts[2]
                  << " support_views3=" << empty_support_view_counts[3] << std::endl;
        std::clog << "[WINDOWS_RAY_PLANE_CANDIDATES] R1=" << ray_r1_candidates
                  << " R2=" << ray_r2_candidates << std::endl;
        std::clog << "[WINDOWS_RAY_R3_PROVENANCE] single_view_empty_total="
                  << single_view_empty_total
                  << " direct_floor_total=" << single_view_direct_floor_total
                  << " no_direct_floor_total="
                  << single_view_no_direct_floor_total << std::endl;
    }
#endif

    std::vector<std::uint8_t> ray_completion_tier(cell_count, 0U);
    std::vector<cv::Vec3f> ray_completion_colors(
        cell_count, cv::Vec3f(0.0f, 0.0f, 0.0f));
    std::vector<float> ray_completion_confidences(cell_count, 0.0f);
    std::vector<std::uint8_t> ray_completion_color_valid(cell_count, 0U);
    std::size_t ray_r1_candidate_count = 0U;
    std::size_t ray_r2_candidate_count = 0U;
    for (int y = 0; y < logical_height_; ++y) {
        for (int x = 0; x < logical_width_; ++x) {
            const std::size_t cell = static_cast<std::size_t>(y)
                * static_cast<std::size_t>(logical_width_)
                + static_cast<std::size_t>(x);
            if (floor_cell_valid_[cell] != 0U) {
                continue;
            }
            const RayPlaneCellSupport& support = ray_plane_support[cell];
            const RayCellClassification classification = classify_ray_cell(support);
            const std::size_t support_views = classification.support_views;
            const std::size_t contradict_views = classification.contradict_views;
            const bool no_contradiction = contradict_views == 0U;
            const bool interior_hole = exterior_empty[cell] == 0U;
            const bool r1 = no_contradiction && support_views >= 2U;
            const bool r2 = no_contradiction && support_views == 1U
                && (final_object_mask[cell] != 0U || interior_hole);
            if (r1) {
                ray_completion_tier[cell] = 1U;
                ++ray_r1_candidate_count;
            } else if (r2) {
                ray_completion_tier[cell] = 2U;
                ++ray_r2_candidate_count;
            } else {
                continue;
            }

            cv::Vec3f color(0.0f, 0.0f, 0.0f);
            float color_weight = 0.0f;
            float confidence_weight = 0.0f;
            for (int view_index = 0; view_index < 3; ++view_index) {
                const std::size_t view = static_cast<std::size_t>(view_index);
                if (!(support.support_pixels[view] > support.contradict_pixels[view]
                    && support.support_pixels[view] > 0U)) {
                    continue;
                }
                const float weight = support.direct_floor_color_weight[view];
                if (!std::isfinite(weight) || weight <= kNumericEpsilon) {
                    continue;
                }
                color += support.direct_floor_color_sum[view];
                color_weight += weight;
                confidence_weight += support.direct_floor_confidence_sum[view];
            }
            if (color_weight > kNumericEpsilon && finite_vec(color)) {
                ray_completion_colors[cell] = color * (1.0f / color_weight);
                ray_completion_confidences[cell] = std::clamp(
                    confidence_weight / color_weight, 0.0f, 1.0f);
                ray_completion_color_valid[cell] = 1U;
            }
        }
    }

#if defined(_WIN32)
    std::array<std::size_t, 3> ray_r3_raw_view_counts{};
    std::array<std::size_t, 3> ray_r3_accepted_view_counts{};
    std::array<std::size_t, 3> ray_r3_component_counts{};
    std::array<std::size_t, 3> ray_r3_touching_component_counts{};
    std::array<std::size_t, 3> ray_r3_disconnected_component_counts{};
    std::size_t ray_r3_candidate_count = 0U;
    std::size_t ray_r3_added = 0U;
    std::size_t ray_r3_missing_direct_rgb = 0U;
#endif
    std::size_t ray_r1_added = 0U;
    std::size_t ray_r2_added = 0U;
    for (std::size_t cell = 0; cell < cell_count; ++cell) {
        const std::uint8_t tier = ray_completion_tier[cell];
        if (tier == 0U || tier == 3U || floor_cell_valid_[cell] != 0U) {
            continue;
        }
        std::uint32_t rgba = 0U;
        float confidence = 0.0f;
        const bool has_direct_color = ray_completion_color_valid[cell] != 0U;
        if (has_direct_color) {
            rgba = color_to_rgba(ray_completion_colors[cell]);
            confidence = ray_completion_confidences[cell];
        } else if (nearest_floor_source[cell] >= 0) {
            const std::size_t source = static_cast<std::size_t>(nearest_floor_source[cell]);
            rgba = floor_cells_[source].rgba;
            confidence = floor_cells_[source].confidence;
        }
#if defined(_WIN32)
        const bool has_propagated_color = !has_direct_color
            && nearest_floor_source[cell] >= 0;
        if (!has_direct_color) {
            ++completion_geometry_without_direct_rgb;
        }
        if (has_direct_color) {
            ++completion_direct_rgb_cells;
        } else if (has_propagated_color) {
            ++completion_propagated_rgb_cells;
        } else {
            ++completion_without_texture_source;
        }
#endif
        const int logical_x = static_cast<int>(
            cell % static_cast<std::size_t>(logical_width_));
        const int logical_y = static_cast<int>(
            cell / static_cast<std::size_t>(logical_width_));
        const std::uint32_t slot_id = slot_for(logical_x, logical_y, 0);
        if (slot_id == std::numeric_limits<std::uint32_t>::max()) {
            continue;
        }
        FusedSlot next = floor_cells_[cell];
        next.slot_id = slot_id;
        next.depth = 0.0f;
        next.confidence = confidence;
        next.rgba = rgba;
        next.floor = true;
        floor_cells_[cell] = next;
        floor_cell_valid_[cell] = 1U;
        result.slots.push_back(next);
        result.occupied_slots.push_back(slot_id);
        if (tier == 1U) {
            ++ray_r1_added;
        } else {
            ++ray_r2_added;
        }
    }

    // Ray-R3 is intentionally a separate completion tier. It uses only
    // single-view ray support that also has a direct near-floor sample, then
    // requires that the same source view's component touches a trusted
    // current-group floor seed. R1/R2 and their tolerances are unchanged.
#if defined(_WIN32)
    const std::size_t ray_floor_before_r3 = static_cast<std::size_t>(std::count(
        floor_cell_valid_.begin(), floor_cell_valid_.end(), static_cast<std::uint8_t>(1U)));
    const float ray_logical_cell_world_size =
        2.0f * display_scale_ / gui_scale;
    const float ray_floor_world_area_before =
        static_cast<float>(ray_floor_before_r3)
        * ray_logical_cell_world_size * ray_logical_cell_world_size;
#endif
    std::vector<std::uint8_t> trusted_floor_seed_mask(cell_count, 0U);
    for (std::size_t cell = 0; cell < cell_count; ++cell) {
        if (current_observed_floor_mask[cell] != 0U
            || floor_completion_tier[cell] != 0U
            || ray_completion_tier[cell] == 1U) {
            trusted_floor_seed_mask[cell] = 1U;
        }
    }

    std::array<cv::Mat, 3> r3_raw_masks;
    for (cv::Mat& mask : r3_raw_masks) {
        mask = cv::Mat(logical_height_, logical_width_, CV_8UC1, cv::Scalar(0));
    }
    std::vector<std::uint8_t> r3_source_view(cell_count, 255U);
    for (int y = 0; y < logical_height_; ++y) {
        for (int x = 0; x < logical_width_; ++x) {
            const std::size_t cell = static_cast<std::size_t>(y)
                * static_cast<std::size_t>(logical_width_)
                + static_cast<std::size_t>(x);
            if (floor_cell_valid_[cell] != 0U
                || ray_completion_tier[cell] != 0U) {
                continue;
            }
            const RayPlaneCellSupport& support = ray_plane_support[cell];
            const RayCellClassification classification = classify_ray_cell(support);
            if (classification.support_views != 1U
                || classification.contradict_views != 0U
                || classification.sole_support_view < 0) {
                continue;
            }
            const std::size_t sole_view = static_cast<std::size_t>(
                classification.sole_support_view);
            const float direct_weight = support.direct_floor_color_weight[sole_view];
            if (!std::isfinite(direct_weight) || direct_weight <= kNumericEpsilon) {
                continue;
            }
            r3_raw_masks[sole_view].at<std::uint8_t>(y, x) = 255U;
            r3_source_view[cell] = static_cast<std::uint8_t>(sole_view);
#if defined(_WIN32)
            ++ray_r3_raw_view_counts[sole_view];
#endif
        }
    }

    std::array<cv::Mat, 3> r3_accepted_masks;
    for (cv::Mat& mask : r3_accepted_masks) {
        mask = cv::Mat(logical_height_, logical_width_, CV_8UC1, cv::Scalar(0));
    }
    for (int view_index = 0; view_index < 3; ++view_index) {
        cv::Mat labels;
        cv::Mat stats;
        cv::Mat centroids;
        const int component_count = cv::connectedComponentsWithStats(
            r3_raw_masks[static_cast<std::size_t>(view_index)], labels, stats,
            centroids, 8, CV_32S);
        std::vector<std::uint8_t> component_touches_seed(
            static_cast<std::size_t>(std::max(component_count, 0)), 0U);
        for (int y = 0; y < logical_height_; ++y) {
            for (int x = 0; x < logical_width_; ++x) {
                const int component = labels.at<int>(y, x);
                if (component <= 0
                    || component >= component_count
                    || component_touches_seed[static_cast<std::size_t>(component)] != 0U) {
                    continue;
                }
                bool touches_seed = false;
                for (int dy = -1; dy <= 1 && !touches_seed; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0) {
                            continue;
                        }
                        const int nx = x + dx;
                        const int ny = y + dy;
                        if (nx < 0 || nx >= logical_width_
                            || ny < 0 || ny >= logical_height_) {
                            continue;
                        }
                        const std::size_t neighbor = static_cast<std::size_t>(ny)
                            * static_cast<std::size_t>(logical_width_)
                            + static_cast<std::size_t>(nx);
                        if (trusted_floor_seed_mask[neighbor] != 0U) {
                            touches_seed = true;
                            break;
                        }
                    }
                }
                if (touches_seed) {
                    component_touches_seed[static_cast<std::size_t>(component)] = 1U;
                }
            }
        }

#if defined(_WIN32)
        const std::size_t component_total = component_count > 0
            ? static_cast<std::size_t>(component_count - 1) : 0U;
        ray_r3_component_counts[static_cast<std::size_t>(view_index)] = component_total;
        for (std::size_t component = 1U; component < component_touches_seed.size();
             ++component) {
            if (component_touches_seed[component] != 0U) {
                ++ray_r3_touching_component_counts[static_cast<std::size_t>(view_index)];
            } else {
                ++ray_r3_disconnected_component_counts[static_cast<std::size_t>(view_index)];
            }
        }
#endif
        for (int y = 0; y < logical_height_; ++y) {
            for (int x = 0; x < logical_width_; ++x) {
                const int component = labels.at<int>(y, x);
                if (component <= 0
                    || component >= component_count
                    || component_touches_seed[static_cast<std::size_t>(component)] == 0U) {
                    continue;
                }
                const std::size_t cell = static_cast<std::size_t>(y)
                    * static_cast<std::size_t>(logical_width_)
                    + static_cast<std::size_t>(x);
                const std::size_t sole_view = static_cast<std::size_t>(
                    r3_source_view[cell]);
                if (sole_view >= 3U) {
                    continue;
                }
                ray_completion_tier[cell] = 3U;
                const float direct_weight =
                    ray_plane_support[cell].direct_floor_color_weight[sole_view];
                const cv::Vec3f direct_color_sum =
                    ray_plane_support[cell].direct_floor_color_sum[sole_view];
                const float direct_confidence_sum =
                    ray_plane_support[cell].direct_floor_confidence_sum[sole_view];
                if (std::isfinite(direct_weight)
                    && direct_weight > kNumericEpsilon
                    && finite_vec(direct_color_sum)
                    && std::isfinite(direct_confidence_sum)) {
                    ray_completion_colors[cell] = direct_color_sum
                        * (1.0f / direct_weight);
                    ray_completion_confidences[cell] = std::clamp(
                        direct_confidence_sum / direct_weight, 0.0f, 1.0f);
                    ray_completion_color_valid[cell] = 1U;
                }
                r3_accepted_masks[static_cast<std::size_t>(view_index)].at<
                    std::uint8_t>(y, x) = 255U;
#if defined(_WIN32)
                ++ray_r3_candidate_count;
                ++ray_r3_accepted_view_counts[static_cast<std::size_t>(view_index)];
#endif
            }
        }
    }

#if defined(_WIN32)
    if (accepted_fuse_count_ == 0U) {
        const auto make_r3_visual = [&](const std::array<cv::Mat, 3>& masks) {
            cv::Mat visual(logical_height_, logical_width_, CV_8UC3,
                cv::Scalar(0, 0, 0));
            for (int y = 0; y < logical_height_; ++y) {
                for (int x = 0; x < logical_width_; ++x) {
                    cv::Vec3b& pixel = visual.at<cv::Vec3b>(y, x);
                    if (masks[0].at<std::uint8_t>(y, x) != 0U) {
                        pixel[2] = 255U; // view0 = red
                    }
                    if (masks[1].at<std::uint8_t>(y, x) != 0U) {
                        pixel[1] = 255U; // view1 = green
                    }
                    if (masks[2].at<std::uint8_t>(y, x) != 0U) {
                        pixel[0] = 255U; // view2 = blue
                    }
                }
            }
            cv::Mat resized;
            cv::resize(visual, resized, cv::Size(canvas_width_, canvas_height_),
                0.0, 0.0, cv::INTER_NEAREST);
            return resized;
        };
        const cv::Mat raw_visual = make_r3_visual(r3_raw_masks);
        const cv::Mat accepted_visual = make_r3_visual(r3_accepted_masks);
        try {
            if (!cv::imwrite("tmp/ray_R3_raw_direct_floor.png", raw_visual)
                || !cv::imwrite("tmp/ray_R3_accepted.png", accepted_visual)) {
                std::clog << "[WINDOWS_RAY_R3_MASK] write_failed=1" << std::endl;
            }
        } catch (const cv::Exception&) {
            std::clog << "[WINDOWS_RAY_R3_MASK] write_failed=1" << std::endl;
        }
        std::clog << "[WINDOWS_RAY_R3] raw_view0=" << ray_r3_raw_view_counts[0]
                  << " raw_view1=" << ray_r3_raw_view_counts[1]
                  << " raw_view2=" << ray_r3_raw_view_counts[2]
                  << " candidates=" << ray_r3_candidate_count << std::endl;
        std::clog << "[WINDOWS_RAY_R3_COMPONENTS] view0="
                  << ray_r3_component_counts[0]
                  << " view1=" << ray_r3_component_counts[1]
                  << " view2=" << ray_r3_component_counts[2]
                  << " touching_view0=" << ray_r3_touching_component_counts[0]
                  << " touching_view1=" << ray_r3_touching_component_counts[1]
                  << " touching_view2=" << ray_r3_touching_component_counts[2]
                  << " disconnected_view0="
                  << ray_r3_disconnected_component_counts[0]
                  << " disconnected_view1="
                  << ray_r3_disconnected_component_counts[1]
                  << " disconnected_view2="
                  << ray_r3_disconnected_component_counts[2] << std::endl;
    }
#endif

    for (std::size_t cell = 0; cell < cell_count; ++cell) {
        const std::uint8_t tier = ray_completion_tier[cell];
        if (tier != 3U || floor_cell_valid_[cell] != 0U) {
            continue;
        }
        if (ray_completion_color_valid[cell] == 0U) {
#if defined(_WIN32)
            ++ray_r3_missing_direct_rgb;
#endif
            continue;
        }
        const int logical_x = static_cast<int>(
            cell % static_cast<std::size_t>(logical_width_));
        const int logical_y = static_cast<int>(
            cell / static_cast<std::size_t>(logical_width_));
        const std::uint32_t slot_id = slot_for(logical_x, logical_y, 0);
        if (slot_id == std::numeric_limits<std::uint32_t>::max()) {
            continue;
        }
        FusedSlot next = floor_cells_[cell];
        next.slot_id = slot_id;
        next.depth = 0.0f;
        next.confidence = ray_completion_confidences[cell];
        next.rgba = color_to_rgba(ray_completion_colors[cell]);
        next.floor = true;
        floor_cells_[cell] = next;
        floor_cell_valid_[cell] = 1U;
        result.slots.push_back(next);
        result.occupied_slots.push_back(slot_id);
#if defined(_WIN32)
        ++ray_r3_added;
        ++completion_direct_rgb_cells;
#endif
    }

#if defined(_WIN32)
    if (accepted_fuse_count_ == 0U) {
        std::size_t remaining_interior_holes = 0U;
        std::size_t remaining_exterior_empty = 0U;
        for (std::size_t cell = 0; cell < cell_count; ++cell) {
            if (floor_cell_valid_[cell] == 0U) {
                if (exterior_empty[cell] == 0U) {
                    ++remaining_interior_holes;
                } else {
                    ++remaining_exterior_empty;
                }
            }
        }
        const std::size_t ray_floor_after_r3 = static_cast<std::size_t>(std::count(
            floor_cell_valid_.begin(), floor_cell_valid_.end(), static_cast<std::uint8_t>(1U)));
        const float ray_floor_world_area_after =
            static_cast<float>(ray_floor_after_r3)
            * ray_logical_cell_world_size * ray_logical_cell_world_size;
        std::clog << "[WINDOWS_RAY_PLANE_RESULT] R1_candidates="
                  << ray_r1_candidate_count
                  << " R1_added=" << ray_r1_added
                  << " R2_candidates=" << ray_r2_candidate_count
                  << " R2_added=" << ray_r2_added
                  << " R3_candidates=" << ray_r3_candidate_count
                  << " R3_added=" << ray_r3_added
                  << " R3_missing_direct_rgb=" << ray_r3_missing_direct_rgb
                  << " floor_before_R3=" << ray_floor_before_r3
                  << " floor_after_R3=" << ray_floor_after_r3
                  << " floor_world_area_before=" << ray_floor_world_area_before
                  << " floor_world_area_after=" << ray_floor_world_area_after
                  << " remaining_interior_holes=" << remaining_interior_holes
                  << " remaining_exterior_empty=" << remaining_exterior_empty
                  << " geometry_without_direct_rgb="
                  << completion_geometry_without_direct_rgb
                  << " geometry_dropped_due_rgb="
                  << completion_geometry_dropped_due_rgb << std::endl;
        std::clog << "[WINDOWS_COMPLETION_TEXTURE] direct_rgb="
                  << completion_direct_rgb_cells
                  << " propagated_rgb=" << completion_propagated_rgb_cells
                  << " without_texture_source=" << completion_without_texture_source
                  << std::endl;

        // BGR channels encode observed floor (R), F1/F2/F3 completion (G),
        // and accepted R1/R2 ray underlay (B). This is a coverage diagnostic
        // only; it never participates in geometry acceptance.
        cv::Mat atlas_coverage(logical_height_, logical_width_, CV_8UC3,
            cv::Scalar(0, 0, 0));
        for (int y = 0; y < logical_height_; ++y) {
            for (int x = 0; x < logical_width_; ++x) {
                const std::size_t cell = static_cast<std::size_t>(y)
                    * static_cast<std::size_t>(logical_width_)
                    + static_cast<std::size_t>(x);
                cv::Vec3b& pixel = atlas_coverage.at<cv::Vec3b>(y, x);
                if (current_observed_floor_mask[cell] != 0U) {
                    pixel[2] = 255U;
                }
                if (floor_completion_tier[cell] != 0U) {
                    pixel[1] = 255U;
                }
                if (ray_completion_tier[cell] != 0U) {
                    pixel[0] = 255U;
                }
            }
        }
        cv::Mat atlas_coverage_visual;
        cv::resize(atlas_coverage, atlas_coverage_visual,
            cv::Size(canvas_width_, canvas_height_), 0.0, 0.0, cv::INTER_NEAREST);
        try {
            if (!cv::imwrite("tmp/atlas_canvas_coverage.png", atlas_coverage_visual)) {
                std::clog << "[WINDOWS_ATLAS_COVERAGE] write_failed=1" << std::endl;
            }
        } catch (const cv::Exception&) {
            std::clog << "[WINDOWS_ATLAS_COVERAGE] write_failed=1" << std::endl;
        }
    }
#endif

    std::vector<float> final_object_heights;
    std::size_t object_final = 0U;
    for (std::size_t cell = 0; cell < cell_count; ++cell) {
        const ObjectSurfaceCandidate& candidate = object_best[cell];
        if (!candidate.valid) {
            continue;
        }
        const std::uint32_t slot_id = slot_for(candidate.logical_x, candidate.logical_y, 1);
        if (slot_id == std::numeric_limits<std::uint32_t>::max()) {
            continue;
        }
        float confidence = 0.0f;
        const cv::Vec3f color = object_candidate_color(candidate, confidence);
        result.slots.push_back(FusedSlot{
            slot_id,
            candidate.height / display_scale_,
            confidence,
            color_to_rgba(color),
            false});
        result.occupied_slots.push_back(slot_id);
        final_object_heights.push_back(candidate.height);
        ++object_final;
    }
#if defined(_WIN32)
    if (accepted_fuse_count_ == 0U) {
        std::array<std::size_t, 5> object_tier_counts{};
        float object_world_hmax = 0.0f;
        float object_render_zmax = 0.0f;
        for (std::size_t cell = 0; cell < cell_count; ++cell) {
            if (!object_best[cell].valid) {
                continue;
            }
            object_world_hmax = std::max(object_world_hmax, object_best[cell].height);
            object_render_zmax = std::max(
                object_render_zmax, object_best[cell].height / display_scale_);
            const std::uint8_t tier = object_tier_provenance[cell];
            if (tier >= 1U && tier <= 4U) {
                ++object_tier_counts[tier];
            } else {
                ++object_tier_counts[0];
            }
        }
        std::clog << "[WINDOWS_OBJECT_TIER_DIAG] tier1="
                  << object_tier_counts[1]
                  << " tier2=" << object_tier_counts[2]
                  << " tier3=" << object_tier_counts[3]
                  << " tier4=" << object_tier_counts[4]
                  << " unclassified=" << object_tier_counts[0]
                  << " final=" << object_final
                  << " object_world_hmax=" << object_world_hmax
                  << " object_render_zmax=" << object_render_zmax << std::endl;
    }
#endif

#if defined(_WIN32)
    if (accepted_fuse_count_ == 0U) {
        std::array<std::size_t, 3> floor_only_kept_after{};
        std::array<std::size_t, 3> floor_only_lost_after{};
        std::array<std::size_t, 3> object_only_kept_after{};
        std::array<std::size_t, 3> object_only_lost_after{};
        std::size_t direct_floor_total = 0U;
        std::size_t direct_object_candidate_union = 0U;
        std::size_t final_empty_on_direct_floor_union = 0U;
        std::size_t object_candidate_not_selected_final = 0U;
        std::size_t inferred_floor_total = 0U;
        cv::Mat direct_union_vs_final(logical_height_, logical_width_, CV_8UC3,
            cv::Scalar(0, 0, 0));
        for (std::size_t cell = 0; cell < cell_count; ++cell) {
            const int x = static_cast<int>(cell % static_cast<std::size_t>(logical_width_));
            const int y = static_cast<int>(cell / static_cast<std::size_t>(logical_width_));
            std::uint8_t floor_pattern = 0U;
            std::uint8_t object_pattern = 0U;
            for (int view_index = 0; view_index < 3; ++view_index) {
                const std::uint8_t bit = static_cast<std::uint8_t>(1U) << view_index;
                if (direct_floor_presence_by_view[static_cast<std::size_t>(view_index)][cell]
                    != 0U) {
                    floor_pattern |= bit;
                }
                if (direct_object_presence_by_view[static_cast<std::size_t>(view_index)][cell]
                    != 0U) {
                    object_pattern |= bit;
                }
            }
            const bool direct_floor = floor_pattern != 0U;
            const bool direct_object = object_pattern != 0U;
            if (direct_floor) {
                ++direct_floor_total;
                if (support_count(floor_pattern) == 1) {
                    const int view_index = floor_pattern == 1U ? 0
                        : (floor_pattern == 2U ? 1 : 2);
                    if (floor_cell_valid_[cell] != 0U) {
                        ++floor_only_kept_after[static_cast<std::size_t>(view_index)];
                    } else {
                        ++floor_only_lost_after[static_cast<std::size_t>(view_index)];
                    }
                }
                if (floor_cell_valid_[cell] == 0U) {
                    ++final_empty_on_direct_floor_union;
                }
            }
            if (direct_object) {
                ++direct_object_candidate_union;
                if (support_count(object_pattern) == 1) {
                    const int view_index = object_pattern == 1U ? 0
                        : (object_pattern == 2U ? 1 : 2);
                    if (object_best[cell].valid) {
                        ++object_only_kept_after[static_cast<std::size_t>(view_index)];
                    } else {
                        ++object_only_lost_after[static_cast<std::size_t>(view_index)];
                    }
                }
                if (!object_best[cell].valid) {
                    ++object_candidate_not_selected_final;
                }
            }
            if (!direct_floor && !direct_object
                && floor_cell_valid_[cell] != 0U
                && current_observed_floor_mask[cell] == 0U) {
                ++inferred_floor_total;
            }

            cv::Vec3b& pixel = direct_union_vs_final.at<cv::Vec3b>(y, x);
            const bool direct_kept = (direct_floor && floor_cell_valid_[cell] != 0U)
                || (direct_object && object_best[cell].valid);
            if (direct_floor || direct_object) {
                if (direct_kept) {
                    pixel = cv::Vec3b(0U, 255U, 0U); // direct kept = green
                } else {
                    pixel = cv::Vec3b(0U, 0U, 255U); // direct lost = red
                }
            } else if (floor_cell_valid_[cell] != 0U
                && current_observed_floor_mask[cell] == 0U) {
                pixel = cv::Vec3b(255U, 0U, 0U); // inferred-only floor = blue
            }
        }
        std::clog << "[WINDOWS_DIRECT_FINAL] direct_floor=" << direct_floor_total
                  << " direct_object_candidate_union=" << direct_object_candidate_union
                  << " direct_floor_lost_after_F4_is_reported_above=1"
                  << " final_empty_on_direct_floor_union_after_inferred="
                        << final_empty_on_direct_floor_union
                  << " object_candidate_not_selected_final="
                        << object_candidate_not_selected_final
                  << " inferred_only_floor=" << inferred_floor_total
                  << " floor_only_view0_kept=" << floor_only_kept_after[0]
                  << " floor_only_view1_kept=" << floor_only_kept_after[1]
                  << " floor_only_view2_kept=" << floor_only_kept_after[2]
                  << " floor_only_view0_lost=" << floor_only_lost_after[0]
                  << " floor_only_view1_lost=" << floor_only_lost_after[1]
                  << " floor_only_view2_lost=" << floor_only_lost_after[2]
                  << " object_only_view0_kept=" << object_only_kept_after[0]
                  << " object_only_view1_kept=" << object_only_kept_after[1]
                  << " object_only_view2_kept=" << object_only_kept_after[2]
                  << " object_only_view0_lost=" << object_only_lost_after[0]
                  << " object_only_view1_lost=" << object_only_lost_after[1]
                  << " object_only_view2_lost=" << object_only_lost_after[2] << std::endl;
        try {
            if (!cv::imwrite("tmp/direct_union_vs_final.png",
                    resize_logical_mask(direct_union_vs_final))) {
                std::clog << "[WINDOWS_DIRECT_FINAL_MASK] write_failed=1" << std::endl;
            }
        } catch (const cv::Exception&) {
            std::clog << "[WINDOWS_DIRECT_FINAL_MASK] write_failed=1" << std::endl;
        }
    }
#endif

    // Brightness is a display-only correction. Geometry and all of the
    // floor/object decisions above have already completed without consulting
    // RGB. Keep the persistent floor cache in pre-gain colour space so a
    // cell that is temporarily occluded is not multiplied again next frame.
    std::vector<float> luminances;
    luminances.reserve(result.slots.size());
    for (const FusedSlot& slot : result.slots) {
        const auto channels = unpack_rgba(slot.rgba);
        if (channels[0] != 0U || channels[1] != 0U || channels[2] != 0U) {
            const float luminance = rgba_luminance(slot.rgba);
            if (luminance >= 0.02f && luminance <= 0.95f) {
                luminances.push_back(luminance);
            }
        }
    }
    if (!luminances.empty()) {
        const float median_luminance = median_value(luminances);
        if (std::isfinite(median_luminance) && median_luminance > kNumericEpsilon) {
            global_color_gain_ = std::clamp(
                0.22f / median_luminance, 1.0f, 1.6f);
        }
    }
    for (FusedSlot& slot : result.slots) {
        slot.rgba = apply_global_color_gain(slot.rgba, global_color_gain_);
    }

    ++accepted_fuse_count_;
    if (accepted_fuse_count_ % 30U == 0U) {
        const float object_h50 = percentile_value(final_object_heights, 0.50f);
        const float object_h90 = percentile_value(final_object_heights, 0.90f);
        const float object_h99 = percentile_value(final_object_heights, 0.99f);
        const std::size_t floor_cell_count = static_cast<std::size_t>(std::count(
            floor_cell_valid_.begin(), floor_cell_valid_.end(), static_cast<std::uint8_t>(1U)));
        std::clog << "[INFO] world_fusion_stats: floor_cells=" << floor_cell_count
                  << " object_pre_consistency=" << object_pre_consistency
                  << " object_support1=" << atlas_support1
                  << " atlas_support2=" << atlas_support2
                  << " atlas_support3=" << atlas_support3
                  << " reprojection_support2=" << reprojection_support2
                  << " reprojection_support3=" << reprojection_support3
                  << " object_final=" << object_final
                  << " object_h50=" << object_h50
                  << " object_h90=" << object_h90
                  << " object_h99=" << object_h99
                  << std::endl;
        std::clog << "[INFO] world_reprojection_stats: atlas_pair="
                  << reprojection_stats.atlas_pair
                  << " bidir_support=" << reprojection_stats.bidir_support
                  << " oneway_support=" << reprojection_stats.oneway_support
                  << " occluded=" << reprojection_stats.occluded
                  << " contradict=" << reprojection_stats.contradict
                  << " outside=" << reprojection_stats.outside
                  << " invalid=" << reprojection_stats.invalid
                  << " object_final=" << object_final
                  << std::endl;
    }

    result.accepted = true;
    return result;
}

}  // namespace omnivggt::observer
