#include "group_world_fusion.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <utility>
#include <vector>

namespace omnivggt::observer {

namespace {

bool finite_vec(const cv::Vec3f& value);

void windows_fusion_debug(const std::string& message) {
#ifdef _WIN32
    std::clog << "[observer-windows-fusion] " << message << std::endl;
#else
    (void)message;
#endif
}

void windows_fusion_map_stats(
    const std::string& label,
    const cv::Mat& points,
    const cv::Mat& confidence) {
#ifdef _WIN32
    std::size_t finite_points = 0U;
    std::size_t finite_confidence = 0U;
    std::size_t confidence_positive = 0U;
    std::size_t z_above_90 = 0U;
    cv::Vec3f point_min(
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity());
    cv::Vec3f point_max(
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity());
    float confidence_min = std::numeric_limits<float>::infinity();
    float confidence_max = -std::numeric_limits<float>::infinity();
    for (int y = 0; y < points.rows; ++y) {
        for (int x = 0; x < points.cols; ++x) {
            const cv::Vec3f point = points.at<cv::Vec3f>(y, x);
            const float point_confidence = confidence.at<float>(y, x);
            if (finite_vec(point)) {
                ++finite_points;
                for (int axis = 0; axis < 3; ++axis) {
                    point_min[axis] = std::min(point_min[axis], point[axis]);
                    point_max[axis] = std::max(point_max[axis], point[axis]);
                }
                if (point[2] > 0.90f) {
                    ++z_above_90;
                }
            }
            if (std::isfinite(point_confidence)) {
                ++finite_confidence;
                confidence_min = std::min(confidence_min, point_confidence);
                confidence_max = std::max(confidence_max, point_confidence);
                if (point_confidence > 0.0f) {
                    ++confidence_positive;
                }
            }
        }
    }
    std::ostringstream stats;
    stats << label
          << " points=" << finite_points
          << " conf_finite=" << finite_confidence
          << " conf_positive=" << confidence_positive
          << " xyz_min=[" << point_min[0] << ',' << point_min[1] << ',' << point_min[2]
          << "] xyz_max=[" << point_max[0] << ',' << point_max[1] << ',' << point_max[2]
          << "] conf=[" << confidence_min << ',' << confidence_max << ']'
          << " z_gt_0.90=" << z_above_90;
    windows_fusion_debug(stats.str());
#else
    (void)label;
    (void)points;
    (void)confidence;
#endif
}

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
};

struct HeightCluster {
    std::vector<ObjectSample> samples;
    float total_weight = 0.0f;
    float height = 0.0f;
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
#ifdef _WIN32
    std::vector<float> support_world_errors;
    std::vector<float> support_depth_errors;
    std::vector<float> failed_world_errors;
    std::vector<float> failed_depth_errors;
#endif
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

void record_reprojection_error(
    const ReprojectionResult& result,
    ReprojectionStats& stats) {
#ifdef _WIN32
    if (!std::isfinite(result.world_error) || !std::isfinite(result.depth_error)) {
        return;
    }
    const bool support = result.relation == ReprojectionRelation::Support;
    auto& world_errors = support
        ? stats.support_world_errors
        : stats.failed_world_errors;
    auto& depth_errors = support
        ? stats.support_depth_errors
        : stats.failed_depth_errors;
    world_errors.push_back(result.world_error);
    depth_errors.push_back(result.depth_error);
#else
    (void)result;
    (void)stats;
#endif
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

ViewObjectObservation make_object_observation(
    const std::vector<ObjectSample>& samples,
    const int view_id,
    const int logical_x,
    const int logical_y,
    const std::array<std::array<float, 3>, 3>& gains) {
    ViewObjectObservation observation;
    if (samples.empty()) {
        return observation;
    }
    observation.valid = true;
    observation.view_id = view_id;
    observation.logical_x = logical_x;
    observation.logical_y = logical_y;
    std::vector<float> u_values;
    std::vector<float> v_values;
    std::vector<float> confidence_values;
    u_values.reserve(samples.size());
    v_values.reserve(samples.size());
    confidence_values.reserve(samples.size());
    observation.weight = 0.0f;
    for (const ObjectSample& sample : samples) {
        u_values.push_back(sample.u);
        v_values.push_back(sample.v);
        confidence_values.push_back(sample.normalized_confidence);
        observation.weight += std::max(sample.weight, 1e-4f);
    }
    observation.u = median_value(std::move(u_values));
    observation.v = median_value(std::move(v_values));
    observation.height = weighted_median_height(samples);
    observation.confidence = median_value(std::move(confidence_values));
    float ignored_confidence = 0.0f;
    observation.color = weighted_color(samples, gains, ignored_confidence);
    observation.sample_count = static_cast<int>(samples.size());
    observation.valid = finite_vec(observation.color)
        && std::isfinite(observation.u)
        && std::isfinite(observation.v)
        && std::isfinite(observation.height)
        && std::isfinite(observation.confidence)
        && std::isfinite(observation.weight)
        && observation.weight > 0.0f;
    return observation;
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
#ifdef _WIN32
    for (int view_index = 0; view_index < 3; ++view_index) {
        windows_fusion_map_stats(
            "raw view=" + std::to_string(view_index),
            views[static_cast<std::size_t>(view_index)].world_points,
            views[static_cast<std::size_t>(view_index)].world_confidence);
    }
#endif

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
#ifdef _WIN32
    for (int view_index = 0; view_index < 3; ++view_index) {
        windows_fusion_map_stats(
            "transformed view=" + std::to_string(view_index),
            aligned_world_points[static_cast<std::size_t>(view_index)],
            views[static_cast<std::size_t>(view_index)].world_confidence);
    }
#endif

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
    std::vector<FloorSample> near_floor_best(cell_count * 3U);
    std::array<std::vector<std::vector<ObjectSample>>, 3> object_samples_by_view;
    for (auto& samples : object_samples_by_view) {
        samples.resize(cell_count);
    }
    std::size_t classified_finite_points = 0U;
    std::size_t atlas_outside_points = 0U;
    std::size_t strict_floor_points = 0U;
    std::size_t strict_floor_neighborhood_rejected = 0U;
    std::size_t near_floor_points = 0U;
    std::size_t near_floor_neighborhood_rejected = 0U;
    std::size_t positive_object_points = 0U;
    std::size_t object_continuity_rejected = 0U;
    std::size_t ignored_height_points = 0U;
#ifdef _WIN32
    std::vector<float> object_sample_heights;
    object_sample_heights.reserve(positive_object_points);
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
                ++classified_finite_points;
                const cv::Vec3f delta = point - plane_origin_;
                const float u = axis_x_.dot(delta);
                const float v = axis_y_.dot(delta);
                const float height = plane_normal_.dot(delta);
                int logical_x = 0;
                int logical_y = 0;
                if (!world_to_cell(u, v, logical_x, logical_y)) {
                    ++atlas_outside_points;
                    continue;
                }
                const std::size_t cell = static_cast<std::size_t>(logical_y)
                    * static_cast<std::size_t>(logical_width_)
                    + static_cast<std::size_t>(logical_x);
                const ConfidenceStats& confidence_range =
                    confidence_stats[static_cast<std::size_t>(view_index)];
                const float normalized = normalized_confidence(
                    confidence, confidence_range.q10, confidence_range.q90);
                const float weight = normalized * border_weight(
                    x, y, view.world_points.cols, view.world_points.rows);
                const float abs_height = std::abs(height);
                if (abs_height <= floor_band_) {
                    ++strict_floor_points;
                    if (!floor_neighborhood_consistent(
                        aligned_points, view.world_confidence, x, y,
                        plane_origin_, plane_normal_, floor_band_)) {
                        ++strict_floor_neighborhood_rejected;
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
                    ++near_floor_points;
                    if (!floor_neighborhood_consistent(
                        aligned_points, view.world_confidence, x, y,
                        plane_origin_, plane_normal_, floor_band_)) {
                        ++near_floor_neighborhood_rejected;
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
                    ++positive_object_points;
                    if (local_object_continuity(
                            aligned_points, view.world_confidence, x, y, point, scene_scale_)) {
#ifdef _WIN32
                        object_sample_heights.push_back(height);
#endif
                        object_samples_by_view[static_cast<std::size_t>(view_index)][cell].push_back(
                            ObjectSample{
                            u,
                            v,
                            height,
                            normalized,
                            std::max(weight, 1e-4f),
                            view.rgb.at<cv::Vec3f>(y, x),
                            view_index});
                    } else {
                        ++object_continuity_rejected;
                    }
                } else {
                    ++ignored_height_points;
                }
                // Points outside the strict/near-floor bands and positive
                // object band are intentionally ignored. Neither decision
                // uses RGB.
            }
        }
    }

    if (accepted_fuse_count_ == 0U) {
        std::size_t object_sample_points = 0U;
        for (const auto& per_view : object_samples_by_view) {
            for (const auto& per_cell : per_view) {
                object_sample_points += per_cell.size();
            }
        }
        std::ostringstream classification;
        classification << "first-group classification"
            << " plane_n=[" << plane_normal_[0] << ',' << plane_normal_[1] << ','
            << plane_normal_[2] << "] plane_origin=[" << plane_origin_[0] << ','
            << plane_origin_[1] << ',' << plane_origin_[2] << ']'
            << " scene_scale=" << scene_scale_
            << " floor_band=" << floor_band_
            << " display_scale=" << display_scale_
            << " max_object_height=" << max_object_height_
            << " atlas_u=[" << u_min_ << ',' << u_max_ << "] atlas_v=["
            << v_min_ << ',' << v_max_ << ']'
            << " finite=" << classified_finite_points
            << " outside_atlas=" << atlas_outside_points
            << " strict_floor=" << strict_floor_points
            << " strict_floor_neighborhood_rejected=" << strict_floor_neighborhood_rejected
            << " near_floor=" << near_floor_points
            << " near_floor_neighborhood_rejected=" << near_floor_neighborhood_rejected
            << " positive_object=" << positive_object_points
            << " object_continuity_rejected=" << object_continuity_rejected
            << " object_samples=" << object_sample_points
            << " ignored_height=" << ignored_height_points;
        windows_fusion_debug(classification.str());
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

    // First reduce each view to at most one robust object observation per
    // logical cell. The views are deliberately kept separate until the
    // cross-view consistency pass below; mixing them first was able to make a
    // single-view curtain look like a supported surface.
    std::array<std::vector<ViewObjectObservation>, 3> object_observations_by_view;
    std::array<std::vector<int>, 3> object_observation_index_by_view;
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
            for (const ObjectSample& sample : samples) {
                u_values.push_back(sample.u);
                v_values.push_back(sample.v);
                confidence_values.push_back(sample.normalized_confidence);
                observation.weight += std::max(sample.weight, 1e-4f);
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
            ++object_pre_consistency;
        }
    }

    const float object_cross_view_height_tolerance = std::max(
        3.0f * floor_band_, 0.02f * scene_scale_);
    const float reprojection_world_tolerance = std::max(
        3.0f * floor_band_, 0.012f * scene_scale_);
    const float reprojection_depth_tolerance = std::max(
        3.0f * floor_band_, 0.015f * scene_scale_);

    std::vector<ObjectSurfaceCandidate> object_best(cell_count);
    std::array<std::vector<std::uint8_t>, 3> object_matched_by_view;
    for (int view_index = 0; view_index < 3; ++view_index) {
        object_matched_by_view[static_cast<std::size_t>(view_index)].assign(
            cell_count, 0U);
    }
    std::vector<std::uint8_t> atlas_support2_cells(cell_count, 0U);
    std::vector<std::uint8_t> atlas_support3_cells(cell_count, 0U);
    ReprojectionStats reprojection_stats;
#ifdef _WIN32
    std::vector<float> pair_proposal_heights;
    std::vector<float> bidirectional_pair_heights;
    pair_proposal_heights.reserve(27795U);
    bidirectional_pair_heights.reserve(27795U);
#endif

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
                        record_reprojection_error(
                            proposal.first_to_second, reprojection_stats);
                        record_reprojection_error(
                            proposal.second_to_first, reprojection_stats);
                        const bool first_support = proposal.first_to_second.relation
                            == ReprojectionRelation::Support;
                        const bool second_support = proposal.second_to_first.relation
                            == ReprojectionRelation::Support;
                        proposal.bidirectional_support = first_support && second_support;
                        ++reprojection_stats.atlas_pair;
#ifdef _WIN32
                        pair_proposal_heights.push_back(
                            0.5f * (first.height + second->height));
                        if (proposal.bidirectional_support) {
                            bidirectional_pair_heights.push_back(
                                0.5f * (first.height + second->height));
                        }
#endif
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
        } else {
            // No third view was needed to validate this strict pair.  The
            // candidate is moved only after all optional third-view work.
            consider_object_candidate(std::move(pair));
        }
    }

    // A strict cross-view match is useful when it succeeds, but a partial
    // occlusion can leave a real object visible in only one of the three
    // images. Recover the former dominant-height-cluster behavior only for
    // cells where the strict pass produced no candidate. This keeps the
    // multi-view validation while preventing the floor layer from replacing
    // all one-view raised geometry.
    const float fallback_cluster_tolerance = std::max(
        2.0f * floor_band_, 0.01f * scene_scale_);
    std::vector<std::uint8_t> fallback_object_cells(cell_count, 0U);
    for (std::size_t cell = 0; cell < cell_count; ++cell) {
        if (object_best[cell].valid) {
            continue;
        }
        std::vector<ObjectSample> samples;
        for (int view_index = 0; view_index < 3; ++view_index) {
            const auto& per_view_samples = object_samples_by_view[
                static_cast<std::size_t>(view_index)][cell];
            samples.insert(samples.end(), per_view_samples.begin(), per_view_samples.end());
        }
        if (samples.empty()) {
            continue;
        }
        std::sort(samples.begin(), samples.end(), [](const ObjectSample& lhs,
            const ObjectSample& rhs) {
            return lhs.height < rhs.height;
        });
        std::vector<HeightCluster> clusters;
        for (const ObjectSample& sample : samples) {
            if (clusters.empty()
                || sample.height - clusters.back().height > fallback_cluster_tolerance) {
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

        const HeightCluster& selected = clusters[selected_index];
        ObjectSurfaceCandidate fallback;
        fallback.valid = true;
        fallback.logical_x = static_cast<int>(
            cell % static_cast<std::size_t>(logical_width_));
        fallback.logical_y = static_cast<int>(
            cell / static_cast<std::size_t>(logical_width_));
        for (int view_index = 0; view_index < 3; ++view_index) {
            std::vector<ObjectSample> view_samples;
            for (const ObjectSample& sample : selected.samples) {
                if (sample.view == view_index) {
                    view_samples.push_back(sample);
                }
            }
            if (view_samples.empty()) {
                continue;
            }
            ViewObjectObservation observation = make_object_observation(
                view_samples,
                view_index,
                fallback.logical_x,
                fallback.logical_y,
                color_gain_);
            if (!observation.valid) {
                continue;
            }
            fallback.view_mask |= static_cast<std::uint8_t>(1U) << view_index;
            fallback.observations[static_cast<std::size_t>(view_index)] =
                std::move(observation);
        }
        finalize_object_candidate(fallback);
        if (fallback.valid && support_count(fallback.view_mask) >= 1) {
            object_best[cell] = std::move(fallback);
            fallback_object_cells[cell] = 1U;
        }
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

    // Connected components run after cross-view matching and the conservative
    // single-view fallback. The existing minimum of eight logical cells is
    // retained to reject isolated noise.
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

    const float height_neighbor_threshold = std::max(
        4.0f * floor_band_, 0.03f * scene_scale_);
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
                    if (object_best[neighbor_cell].valid) {
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

    if (accepted_fuse_count_ == 0U) {
        std::size_t result_floor_slots = 0U;
        std::size_t result_object_slots = 0U;
        std::size_t result_above_001 = 0U;
        std::size_t result_above_005 = 0U;
        float result_depth_min = std::numeric_limits<float>::infinity();
        float result_depth_max = -std::numeric_limits<float>::infinity();
        for (const FusedSlot& slot : result.slots) {
            if (slot.floor) {
                ++result_floor_slots;
            } else {
                ++result_object_slots;
            }
            result_depth_min = std::min(result_depth_min, slot.depth);
            result_depth_max = std::max(result_depth_max, slot.depth);
            if (slot.depth > 0.01f) {
                ++result_above_001;
            }
            if (slot.depth > 0.05f) {
                ++result_above_005;
            }
        }
        std::ostringstream result_stats;
        result_stats << "first-group fusion result"
            << " slots=" << result.slots.size()
            << " floor_slots=" << result_floor_slots
            << " object_slots=" << result_object_slots
            << " occupied_slots=" << result.occupied_slots.size()
            << " depth=[" << result_depth_min << ',' << result_depth_max << ']'
            << " depth_gt_0.01=" << result_above_001
            << " depth_gt_0.05=" << result_above_005
            << " object_final=" << object_final
            << " object_height=[" << percentile_value(final_object_heights, 0.01f)
            << ',' << percentile_value(final_object_heights, 0.99f) << ']';
        windows_fusion_debug(result_stats.str());
    }

    // Keep a low-rate diagnostic in the normal server log. This class does
    // not own the commit worker, so an accepted fusion group is the closest
    // commit-visible event available here; never create a /tmp audit file.
    ++accepted_fuse_count_;
#ifdef _WIN32
    if (accepted_fuse_count_ == 1U) {
        std::ostringstream matching;
        const std::size_t fallback_object_count = static_cast<std::size_t>(std::count(
            fallback_object_cells.begin(), fallback_object_cells.end(),
            static_cast<std::uint8_t>(1U)));
        matching << "first-group matching"
            << " object_pre_consistency=" << object_pre_consistency
            << " pair_proposals=" << pair_proposals.size()
            << " object_candidates_before_components="
            << (reprojection_support2 + reprojection_support3 + fallback_object_count)
            << " fallback_object_cells=" << fallback_object_count
            << " support1_cells=" << atlas_support1
            << " atlas_support2_cells=" << atlas_support2
            << " atlas_support3_cells=" << atlas_support3
            << " reprojection_support2=" << reprojection_support2
            << " reprojection_support3=" << reprojection_support3
            << " object_final=" << object_final
            << " reprojection_bidir=" << reprojection_stats.bidir_support
            << " reprojection_oneway=" << reprojection_stats.oneway_support
            << " reprojection_occluded=" << reprojection_stats.occluded
            << " reprojection_contradict=" << reprojection_stats.contradict
            << " reprojection_outside=" << reprojection_stats.outside
            << " reprojection_invalid=" << reprojection_stats.invalid
            << " reproj_support_world_q50="
            << percentile_value(reprojection_stats.support_world_errors, 0.50f)
            << " reproj_support_world_q90="
            << percentile_value(reprojection_stats.support_world_errors, 0.90f)
            << " reproj_support_depth_q50="
            << percentile_value(reprojection_stats.support_depth_errors, 0.50f)
            << " reproj_support_depth_q90="
            << percentile_value(reprojection_stats.support_depth_errors, 0.90f)
            << " object_height_sample_q50="
            << percentile_value(object_sample_heights, 0.50f)
            << " object_height_sample_q90="
            << percentile_value(object_sample_heights, 0.90f)
            << " object_height_sample_q99="
            << percentile_value(object_sample_heights, 0.99f)
            << " pair_height_q50="
            << percentile_value(pair_proposal_heights, 0.50f)
            << " pair_height_q90="
            << percentile_value(pair_proposal_heights, 0.90f)
            << " pair_height_q99="
            << percentile_value(pair_proposal_heights, 0.99f)
            << " bidir_height_q50="
            << percentile_value(bidirectional_pair_heights, 0.50f)
            << " bidir_height_q90="
            << percentile_value(bidirectional_pair_heights, 0.90f)
            << " bidir_height_q99="
            << percentile_value(bidirectional_pair_heights, 0.99f)
            << " reproj_failed_world_q50="
            << percentile_value(reprojection_stats.failed_world_errors, 0.50f)
            << " reproj_failed_world_q90="
            << percentile_value(reprojection_stats.failed_world_errors, 0.90f)
            << " reproj_failed_world_q99="
            << percentile_value(reprojection_stats.failed_world_errors, 0.99f)
            << " reproj_failed_depth_q50="
            << percentile_value(reprojection_stats.failed_depth_errors, 0.50f)
            << " reproj_failed_depth_q90="
            << percentile_value(reprojection_stats.failed_depth_errors, 0.90f)
            << " reproj_failed_depth_q99="
            << percentile_value(reprojection_stats.failed_depth_errors, 0.99f)
            << " object_height_tolerance="
            << object_cross_view_height_tolerance
            << " fallback_cluster_tolerance=" << fallback_cluster_tolerance
            << " reprojection_tolerance=" << reprojection_world_tolerance;
        windows_fusion_debug(matching.str());
    }
#endif
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
