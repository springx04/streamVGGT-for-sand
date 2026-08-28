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

cv::Mat internal_hole_fill_mask(
    const cv::Mat& valid,
    const int close_size = 81,
    const int max_hole_area = 40000) {
    if (valid.empty() || cv::countNonZero(valid) == 0
        || close_size <= 1 || max_hole_area <= 0) {
        return cv::Mat::zeros(valid.size(), CV_8UC1);
    }
    const cv::Mat close_kernel = cv::Mat::ones(close_size, close_size, CV_8UC1);
    cv::Mat closed;
    cv::morphologyEx(valid, closed, cv::MORPH_CLOSE, close_kernel);
    cv::Mat inverse_valid;
    cv::bitwise_not(valid, inverse_valid);
    cv::Mat candidates;
    cv::bitwise_and(closed, inverse_valid, candidates);
    if (cv::countNonZero(candidates) == 0) {
        return cv::Mat::zeros(valid.size(), CV_8UC1);
    }

    // Only fill components completely enclosed by valid support.  A closing
    // operation also bridges a concave outer boundary; accepting those
    // pixels would turn the black aperture/background into a false surface.
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int component_count = cv::connectedComponentsWithStats(
        candidates, labels, stats, centroids, 8, CV_32S);
    cv::Mat result = cv::Mat::zeros(valid.size(), CV_8UC1);
    for (int component = 1; component < component_count; ++component) {
        const int x = stats.at<int>(component, cv::CC_STAT_LEFT);
        const int y = stats.at<int>(component, cv::CC_STAT_TOP);
        const int width = stats.at<int>(component, cv::CC_STAT_WIDTH);
        const int height = stats.at<int>(component, cv::CC_STAT_HEIGHT);
        const int area = stats.at<int>(component, cv::CC_STAT_AREA);
        const bool touches_border = x <= 0 || y <= 0
            || x + width >= valid.cols || y + height >= valid.rows;
        if (!touches_border && area > 0 && area <= max_hole_area) {
            result.setTo(255U, labels == component);
        }
    }
    return result;
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

std::vector<float> visual_depth_canvas(const CanvasState& state) {
    if (!state.shape_valid() || state.depth.size() != state.slot_count()
        || state.valid.size() != state.slot_count()) {
        return {};
    }
    // A negative depth scale marks the plane-aware world atlas. Its depth
    // values are already normalized floor residuals plus true anchor-owned
    // raised geometry; fitting/clipping them again makes the arm a texture.
    if (state.anchor_camera.depth_scale < 0.0f) {
        return state.depth;
    }

    std::vector<float> display_depth(state.slot_count(), 0.0f);
    std::vector<float> x_values;
    std::vector<float> y_values;
    std::vector<float> z_values;
    std::vector<std::size_t> slots;
    x_values.reserve(state.slot_count());
    y_values.reserve(state.slot_count());
    z_values.reserve(state.slot_count());
    slots.reserve(state.slot_count());

    const float scale = static_cast<float>(std::max(state.height, state.width));
    for (int y = 0; y < state.height; ++y) {
        for (int x = 0; x < state.width; ++x) {
            const std::size_t slot = static_cast<std::size_t>(y) * state.width + x;
            const float depth = state.depth[slot];
            if (state.valid[slot] == 0U || !std::isfinite(depth)) {
                continue;
            }
            x_values.push_back((static_cast<float>(x) - state.width * 0.5f) / scale);
            y_values.push_back(-(static_cast<float>(y) - state.height * 0.5f) / scale);
            z_values.push_back(depth);
            slots.push_back(slot);
        }
    }
    if (z_values.size() < 3U) {
        return display_depth;
    }

    // Fit the dominant canvas surface, reject large foreground residuals while
    // fitting, then bound the residual exposed to the GUI.  This is the same
    // geometry convention as the clean PLY exporter, but keeps the original
    // valid mask and colors untouched.
    const float base_depth = std::max(std::abs(median_value(z_values)), 1e-6f);
    std::vector<std::uint8_t> keep(z_values.size(), 1U);
    std::array<double, 6> coefficients{};
    fit_quadratic(x_values, y_values, z_values, keep, coefficients);
    for (int iteration = 0; iteration < 3; ++iteration) {
        fit_quadratic(x_values, y_values, z_values, keep, coefficients);
        std::vector<float> residuals;
        residuals.reserve(z_values.size());
        std::vector<float> kept_residuals;
        kept_residuals.reserve(z_values.size());
        for (std::size_t index = 0; index < z_values.size(); ++index) {
            const float residual = static_cast<float>(
                z_values[index] - quadratic_value(
                    coefficients, x_values[index], y_values[index]));
            residuals.push_back(residual);
            if (keep[index] != 0U) {
                kept_residuals.push_back(residual);
            }
        }
        if (kept_residuals.empty()) {
            break;
        }
        const float center = median_value(kept_residuals);
        std::vector<float> deviations;
        deviations.reserve(kept_residuals.size());
        for (const float value : kept_residuals) {
            deviations.push_back(std::abs(value - center));
        }
        const float limit = 4.0f * 1.4826f
            * std::max(median_value(deviations), 1e-6f);
        std::size_t kept = 0U;
        for (std::size_t index = 0; index < residuals.size(); ++index) {
            keep[index] = std::abs(residuals[index] - center) <= limit ? 1U : 0U;
            kept += keep[index] != 0U ? 1U : 0U;
        }
        if (kept < 128U) {
            std::fill(keep.begin(), keep.end(), static_cast<std::uint8_t>(1U));
            break;
        }
    }

    fit_quadratic(x_values, y_values, z_values, keep, coefficients);
    std::vector<float> residuals;
    residuals.reserve(z_values.size());
    for (std::size_t index = 0; index < z_values.size(); ++index) {
        residuals.push_back(static_cast<float>(
            z_values[index] - quadratic_value(
                coefficients, x_values[index], y_values[index])));
    }
    // Estimate display range from the fitted floor inliers only. Foreground
    // objects must not inflate the floor noise estimate, but they also must not
    // be flattened to the old three-percent cap. Preserve coherent object
    // height up to a conservative 12-20% of camera depth while still bounding
    // the catastrophic depth walls this display transform was introduced for.
    std::vector<float> floor_residuals;
    floor_residuals.reserve(residuals.size());
    for (std::size_t index = 0; index < residuals.size(); ++index) {
        if (keep[index] != 0U) {
            floor_residuals.push_back(residuals[index]);
        }
    }
    if (floor_residuals.empty()) {
        floor_residuals = residuals;
    }
    const float center = median_value(floor_residuals);
    std::vector<float> deviations;
    deviations.reserve(floor_residuals.size());
    for (const float value : floor_residuals) {
        deviations.push_back(std::abs(value - center));
    }
    const float robust_limit = 3.0f * 1.4826f
        * std::max(median_value(deviations), 1e-6f);
    const float floor_limit = std::clamp(
        robust_limit,
        0.02f * base_depth,
        0.05f * base_depth);
    const float foreground_limit = std::clamp(
        std::max(3.0f * robust_limit, 0.18f * base_depth),
        0.18f * base_depth,
        0.35f * base_depth);

    cv::Mat residual_canvas(
        state.height,
        state.width,
        CV_32FC1,
        cv::Scalar(std::numeric_limits<float>::quiet_NaN()));
    cv::Mat valid_mask(state.height, state.width, CV_8UC1, cv::Scalar(0));
    for (std::size_t index = 0; index < residuals.size(); ++index) {
        const std::size_t slot = slots[index];
        const int x = static_cast<int>(slot % static_cast<std::size_t>(state.width));
        const int y = static_cast<int>(slot / static_cast<std::size_t>(state.width));
        residual_canvas.at<float>(y, x) = residuals[index];
        valid_mask.at<std::uint8_t>(y, x) = 255U;
    }
    cv::Mat support_distance;
    cv::distanceTransform(valid_mask, support_distance, cv::DIST_L2, 3);

    // A real raised surface occupies a coherent interior neighbourhood.
    // Large residuals at the camera/support boundary and isolated spikes are
    // the failure mode that previously formed vertical sheets, so keep those
    // on the tight floor range while allowing internal coherent objects to
    // use the wider foreground range.
    const float coherence_tolerance = std::max(
        robust_limit,
        0.015f * base_depth);
    std::vector<float> clipped;
    clipped.reserve(residuals.size());
    for (std::size_t index = 0; index < residuals.size(); ++index) {
        const std::size_t slot = slots[index];
        const int x = static_cast<int>(slot % static_cast<std::size_t>(state.width));
        const int y = static_cast<int>(slot / static_cast<std::size_t>(state.width));
        const float value = residuals[index];
        int agreeing_neighbours = 0;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) {
                    continue;
                }
                const int nx = x + dx;
                const int ny = y + dy;
                if (nx < 0 || nx >= state.width || ny < 0 || ny >= state.height) {
                    continue;
                }
                const float neighbour = residual_canvas.at<float>(ny, nx);
                if (std::isfinite(neighbour)
                    && std::abs(neighbour - value) <= coherence_tolerance) {
                    ++agreeing_neighbours;
                }
            }
        }
        const bool coherent_foreground =
            std::abs(value - center) > floor_limit
            && support_distance.at<float>(y, x) >= 3.0f
            && agreeing_neighbours >= 2;
        if (keep[index] != 0U && !coherent_foreground) {
            // Flatten only genuine low-residual floor inliers. A coherent
            // raised object may be included by the broad robust fit and must
            // not be turned into a texture on the floor.
            clipped.push_back(center);
        } else {
            const float local_limit =
                coherent_foreground ? foreground_limit : floor_limit;
            clipped.push_back(std::clamp(
                value,
                center - local_limit,
                center + local_limit));
        }
    }
    const float clipped_center = median_value(clipped);
    for (std::size_t index = 0; index < clipped.size(); ++index) {
        // Camera depth decreases for surfaces raised toward the camera.
        // Publish geometric height with the opposite sign so the robot and
        // objects appear above the fitted floor rather than underneath it.
        display_depth[slots[index]] = (clipped_center - clipped[index]) / base_depth;
    }
    return display_depth;
}


CanvasState visual_canvas_state(const CanvasState& state) {
    if (!state.shape_valid() || state.depth.size() != state.slot_count()
        || state.confidence.size() != state.slot_count()
        || state.rgba.size() != state.slot_count()
        || state.last_update_frame.size() != state.slot_count()
        || state.valid.size() != state.slot_count()
        || state.support.size() != state.slot_count()) {
        return state;
    }

    CanvasState display = state;
    const std::vector<float> normalized_depth = visual_depth_canvas(state);
    if (normalized_depth.size() == state.slot_count()) {
        display.depth = normalized_depth;
    }
    if (state.anchor_camera.depth_scale < 0.0f) {
        // World/group slots already contain the authoritative fixed 2x2
        // logical-cell geometry. The viewer must not add support-gap slots or
        // copy a neighbouring surface into an unobserved world slot.
        return display;
    }

    cv::Mat valid(state.height, state.width, CV_8UC1, cv::Scalar(0));
    cv::Mat support(state.height, state.width, CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < state.height; ++y) {
        for (int x = 0; x < state.width; ++x) {
            const std::size_t slot = static_cast<std::size_t>(y) * state.width + x;
            valid.at<std::uint8_t>(y, x) = state.valid[slot] != 0U ? 255U : 0U;
            support.at<std::uint8_t>(y, x) = state.support[slot] != 0U ? 255U : 0U;
        }
    }
    if (cv::countNonZero(valid) == 0 || cv::countNonZero(support) == 0) {
        return display;
    }

    // Only repair small gaps that are explicitly inside an observed camera
    // support rectangle.  Outer background/aperture pixels remain invalid.
    const std::vector<std::int32_t> nearest = nearest_indices(valid);
    cv::Mat distance;
    cv::distanceTransform(valid, distance, cv::DIST_L2, 3);
    constexpr float kMaxDisplayGapPixels = 48.0f;
    for (int y = 0; y < state.height; ++y) {
        for (int x = 0; x < state.width; ++x) {
            const std::size_t slot = static_cast<std::size_t>(y) * state.width + x;
            if (state.valid[slot] != 0U
                || state.support[slot] == 0U
                || distance.at<float>(y, x) > kMaxDisplayGapPixels) {
                continue;
            }
            const int source = nearest[slot];
            if (source < 0 || static_cast<std::size_t>(source) >= state.slot_count()) {
                continue;
            }
            const std::size_t source_slot = static_cast<std::size_t>(source);
            if (normalized_depth.size() != state.slot_count()
                || state.valid[source_slot] == 0U
                || !std::isfinite(normalized_depth[source_slot])) {
                continue;
            }
            display.depth[slot] = normalized_depth[source_slot];
            display.confidence[slot] = state.confidence[source_slot];
            display.rgba[slot] = state.rgba[source_slot];
            display.last_update_frame[slot] = state.last_update_frame[source_slot];
            display.valid[slot] = 1U;
        }
    }
    return display;
}

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
    if (state.anchor_camera.depth_scale < 0.0f) {
        // World/group mode is already expressed in the same slot geometry as
        // the real GUI. Do not fill holes, smooth, fit, clip, or regenerate
        // X/Y from any auxiliary CanvasState fields in this branch.
        const float scale = static_cast<float>(std::max(height, width));
        std::vector<ExportPoint> points;
        points.reserve(static_cast<std::size_t>(std::count(
            state.valid.begin(), state.valid.end(), static_cast<std::uint8_t>(1U))));
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const std::size_t slot = static_cast<std::size_t>(y) * width + x;
                if (state.valid[slot] == 0U
                    || !std::isfinite(state.depth[slot])) {
                    continue;
                }
                const auto color = unpack_rgba(state.rgba[slot]);
                points.push_back(ExportPoint{
                    (static_cast<float>(x) - width * 0.5f) / scale,
                    -(static_cast<float>(y) - height * 0.5f) / scale,
                    state.depth[slot],
                    color[0],
                    color[1],
                    color[2],
                    changed_frame != 0U
                        && state.last_update_frame[slot]
                            == static_cast<std::uint32_t>(changed_frame)});
            }
        }
        return points;
    }

    if (cv::countNonZero(valid) == 0) {
        return {};
    }

    // The rotating aperture can leave a sizeable internal support gap when a
    // three-image group is committed as one canvas update.  Match the Python
    // geometry-first exporter: close up to 81 pixels, fill only enclosed
    // holes (not the outer aperture), and cap the copied area.  This is an
    // export/viewer regularization; the authoritative streaming state is not
    // modified and no extra model forward is introduced.
    const cv::Mat fill = internal_hole_fill_mask(valid, 81, 40000);
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
