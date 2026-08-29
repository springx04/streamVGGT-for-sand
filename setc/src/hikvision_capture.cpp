#include "hikvision_capture.hpp"

#include <MvCameraControl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <opencv2/core/persistence.hpp>
#include <opencv2/imgcodecs.hpp>

namespace omnivggt::hikvision {

namespace {

void check_sdk(int code, const char* operation) {
    if (code == MV_OK) {
        return;
    }
    std::ostringstream message;
    message << operation << " failed with MVS error 0x"
            << std::hex << std::uppercase << static_cast<unsigned int>(code);
    throw std::runtime_error(message.str());
}

std::string json_escape(const std::string& value) {
    std::ostringstream escaped;
    for (const unsigned char character : value) {
        switch (character) {
        case '"':
            escaped << "\\\"";
            break;
        case '\\':
            escaped << "\\\\";
            break;
        case '\b':
            escaped << "\\b";
            break;
        case '\f':
            escaped << "\\f";
            break;
        case '\n':
            escaped << "\\n";
            break;
        case '\r':
            escaped << "\\r";
            break;
        case '\t':
            escaped << "\\t";
            break;
        default:
            if (character < 0x20U) {
                escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<unsigned int>(character) << std::dec;
            } else {
                escaped << static_cast<char>(character);
            }
            break;
        }
    }
    return escaped.str();
}

std::string csv_escape(const std::string& value) {
    std::string escaped = value;
    std::size_t position = 0U;
    while ((position = escaped.find('"', position)) != std::string::npos) {
        escaped.insert(position, 1U, '"');
        position += 2U;
    }
    return "\"" + escaped + "\"";
}

std::string frame_stem(const std::uint64_t frame_index) {
    std::ostringstream name;
    name << "frame_" << std::setw(8) << std::setfill('0') << frame_index;
    return name.str();
}

std::uint64_t host_timestamp_ns() {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

cv::Mat read_matrix_node(
    const cv::FileStorage& storage,
    const std::initializer_list<const char*>& names,
    const char* description) {
    for (const char* name : names) {
        const cv::FileNode node = storage[name];
        if (node.empty()) {
            continue;
        }
        cv::Mat value;
        node >> value;
        if (!value.empty()) {
            return value;
        }
    }
    throw std::runtime_error(std::string("calibration file is missing ") + description);
}

int read_optional_int(const cv::FileStorage& storage, const char* name) {
    const cv::FileNode node = storage[name];
    if (node.empty()) {
        return 0;
    }
    int value = 0;
    node >> value;
    return value;
}

void read_image_size(const cv::FileStorage& storage, int& width, int& height) {
    width = read_optional_int(storage, "image_width");
    height = read_optional_int(storage, "image_height");
    if (width > 0 && height > 0) {
        return;
    }

    const cv::FileNode size_node = storage["image_size"];
    if (!size_node.empty() && size_node.isSeq() && size_node.size() >= 2U) {
        width = static_cast<int>(size_node[0]);
        height = static_cast<int>(size_node[1]);
    }
}

std::vector<double> mat_to_vector(const cv::Mat& input) {
    cv::Mat flat = input.reshape(1, 1);
    cv::Mat flat64;
    flat.convertTo(flat64, CV_64F);
    std::vector<double> values;
    values.reserve(flat64.total());
    for (int index = 0; index < flat64.cols; ++index) {
        values.push_back(flat64.at<double>(0, index));
    }
    return values;
}

std::string read_string_node(void* handle, const char* name) {
    MVCC_STRINGVALUE value = {};
    if (MV_CC_GetStringValue(handle, name, &value) != MV_OK) {
        return {};
    }
    return value.chCurValue;
}

double read_float_node(void* handle, const std::string& name) {
    MVCC_FLOATVALUE value = {};
    const int code = MV_CC_GetFloatValue(handle, name.c_str(), &value);
    if (code != MV_OK) {
        std::ostringstream message;
        message << "MV_CC_GetFloatValue(" << name << ") failed with MVS error 0x"
                << std::hex << std::uppercase << static_cast<unsigned int>(code);
        throw std::runtime_error(message.str());
    }
    return static_cast<double>(value.fCurValue);
}

struct ImageBufferGuard {
    void* handle = nullptr;
    MV_FRAME_OUT* frame = nullptr;

    ~ImageBufferGuard() {
        if (handle != nullptr && frame != nullptr && frame->pBufAddr != nullptr) {
            (void)MV_CC_FreeImageBuffer(handle, frame);
        }
    }
};

}  // namespace

bool IntrinsicNodeNames::has_matrix_nodes() const noexcept {
    return !fx.empty() && !fy.empty() && !cx.empty() && !cy.empty();
}

bool IntrinsicNodeNames::has_any_distortion_node() const noexcept {
    return !k1.empty() || !k2.empty() || !p1.empty() || !p2.empty() || !k3.empty();
}

bool IntrinsicNodeNames::has_complete_distortion_nodes() const noexcept {
    return !k1.empty() && !k2.empty() && !p1.empty() && !p2.empty() && !k3.empty();
}

bool CameraIntrinsics::valid() const noexcept {
    if (image_width <= 0 || image_height <= 0
        || !std::isfinite(fx) || !std::isfinite(fy)
        || !std::isfinite(cx) || !std::isfinite(cy)
        || fx <= 0.0 || fy <= 0.0) {
        return false;
    }
    return std::all_of(distortion.begin(), distortion.end(), [](const double value) {
        return std::isfinite(value);
    });
}

cv::Mat CameraIntrinsics::camera_matrix() const {
    return (cv::Mat_<double>(3, 3)
        << fx, 0.0, cx,
           0.0, fy, cy,
           0.0, 0.0, 1.0);
}

cv::Mat CameraIntrinsics::distortion_coefficients() const {
    if (distortion.empty()) {
        return {};
    }
    cv::Mat result(1, static_cast<int>(distortion.size()), CV_64F);
    for (std::size_t index = 0U; index < distortion.size(); ++index) {
        result.at<double>(0, static_cast<int>(index)) = distortion[index];
    }
    return result;
}

HikvisionCamera::HikvisionCamera(HikvisionCameraOptions options)
    : options_(std::move(options)) {
    if (options_.timeout_ms <= 0) {
        throw std::invalid_argument("Hikvision MVS timeout must be positive");
    }
    if (!options_.calibration_file.empty()) {
        calibration_intrinsics_ = load_calibration_file(options_.calibration_file);
    }
}

HikvisionCamera::~HikvisionCamera() {
    close();
}

void HikvisionCamera::open() {
    if (device_open_) {
        return;
    }

    MV_CC_DEVICE_INFO_LIST devices = {};
    check_sdk(
        MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &devices),
        "MV_CC_EnumDevices");
    if (devices.nDeviceNum == 0U) {
        throw std::runtime_error("MVS did not enumerate any GigE or USB camera");
    }
    if (options_.device_index >= devices.nDeviceNum) {
        std::ostringstream message;
        message << "--device-index " << options_.device_index
                << " is out of range; MVS enumerated " << devices.nDeviceNum << " device(s)";
        throw std::out_of_range(message.str());
    }

    try {
        check_sdk(
            MV_CC_CreateHandle(&handle_, devices.pDeviceInfo[options_.device_index]),
            "MV_CC_CreateHandle");
        check_sdk(MV_CC_OpenDevice(handle_, MV_ACCESS_Exclusive, 0), "MV_CC_OpenDevice");
        device_open_ = true;
    } catch (...) {
        if (handle_ != nullptr) {
            (void)MV_CC_DestroyHandle(handle_);
            handle_ = nullptr;
        }
        throw;
    }

    camera_id_ = options_.camera_name;
    if (camera_id_.empty()) {
        camera_id_ = read_string_node(handle_, "DeviceSerialNumber");
    }
    if (camera_id_.empty()) {
        camera_id_ = "mvs_device_" + std::to_string(options_.device_index);
    }
}

void HikvisionCamera::start() {
    if (!device_open_) {
        throw std::logic_error("HikvisionCamera::start requires open()");
    }
    if (grabbing_) {
        return;
    }

    check_sdk(
        MV_CC_SetEnumValue(handle_, "TriggerMode", options_.software_trigger ? 1U : 0U),
        "MV_CC_SetEnumValue(TriggerMode)");
    if (options_.software_trigger) {
        check_sdk(
            MV_CC_SetEnumValueByString(handle_, "TriggerSource", "Software"),
            "MV_CC_SetEnumValueByString(TriggerSource)");
    }
    check_sdk(MV_CC_StartGrabbing(handle_), "MV_CC_StartGrabbing");
    grabbing_ = true;
}

void HikvisionCamera::stop() {
    if (!grabbing_) {
        return;
    }
    check_sdk(MV_CC_StopGrabbing(handle_), "MV_CC_StopGrabbing");
    grabbing_ = false;
}

void HikvisionCamera::close() noexcept {
    if (handle_ == nullptr) {
        device_open_ = false;
        grabbing_ = false;
        return;
    }
    if (grabbing_) {
        (void)MV_CC_StopGrabbing(handle_);
        grabbing_ = false;
    }
    if (device_open_) {
        (void)MV_CC_CloseDevice(handle_);
        device_open_ = false;
    }
    (void)MV_CC_DestroyHandle(handle_);
    handle_ = nullptr;
}

bool HikvisionCamera::is_open() const noexcept {
    return device_open_;
}

bool HikvisionCamera::is_grabbing() const noexcept {
    return grabbing_;
}

const std::string& HikvisionCamera::camera_id() const noexcept {
    return camera_id_;
}

CameraIntrinsics HikvisionCamera::load_calibration_file(const std::filesystem::path& path) {
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error("calibration file does not exist: " + path.string());
    }
    cv::FileStorage storage(path.string(), cv::FileStorage::READ);
    if (!storage.isOpened()) {
        throw std::runtime_error("failed to open calibration file: " + path.string());
    }

    const cv::Mat matrix = read_matrix_node(storage, {"camera_matrix", "K"}, "camera_matrix/K");
    if (matrix.total() != 9U) {
        throw std::runtime_error("calibration camera_matrix must contain 9 values");
    }
    cv::Mat matrix64;
    matrix.reshape(1, 3).convertTo(matrix64, CV_64F);
    CameraIntrinsics intrinsics;
    intrinsics.fx = matrix64.at<double>(0, 0);
    intrinsics.fy = matrix64.at<double>(1, 1);
    intrinsics.cx = matrix64.at<double>(0, 2);
    intrinsics.cy = matrix64.at<double>(1, 2);

    for (const char* name : {"distortion_coefficients", "distCoeffs", "D"}) {
        const cv::FileNode node = storage[name];
        if (!node.empty()) {
            cv::Mat distortion;
            node >> distortion;
            if (!distortion.empty()) {
                intrinsics.distortion = mat_to_vector(distortion);
                break;
            }
        }
    }
    read_image_size(storage, intrinsics.image_width, intrinsics.image_height);
    intrinsics.source = "calibration_file:" + path.generic_string();
    if (!intrinsics.valid() && intrinsics.image_width > 0 && intrinsics.image_height > 0) {
        throw std::runtime_error("calibration file contains an invalid camera matrix: " + path.string());
    }
    return intrinsics;
}

void HikvisionCamera::validate_intrinsics_for_image(
    CameraIntrinsics& intrinsics,
    const int image_width,
    const int image_height) const {
    if (image_width <= 0 || image_height <= 0) {
        throw std::invalid_argument("captured image dimensions must be positive");
    }
    if (intrinsics.image_width == 0) {
        intrinsics.image_width = image_width;
    }
    if (intrinsics.image_height == 0) {
        intrinsics.image_height = image_height;
    }
    if (intrinsics.image_width != image_width || intrinsics.image_height != image_height) {
        std::ostringstream message;
        message << "intrinsic image size " << intrinsics.image_width << "x" << intrinsics.image_height
                << " does not match captured image " << image_width << "x" << image_height
                << "; keep MVS ROI/decimation fixed or recalibrate for the output resolution";
        throw std::runtime_error(message.str());
    }
    if (!intrinsics.valid()) {
        throw std::runtime_error("camera intrinsics are invalid");
    }
}

CameraIntrinsics HikvisionCamera::read_intrinsics_from_nodes(
    const int image_width,
    const int image_height) const {
    if (!options_.intrinsic_nodes.has_matrix_nodes()) {
        throw std::runtime_error(
            "this MVS camera did not receive complete intrinsic node names; "
            "ordinary 2-D cameras normally need --calibration-file, while a device-specific "
            "calibrated camera can use --fx-node/--fy-node/--cx-node/--cy-node");
    }

    CameraIntrinsics intrinsics;
    intrinsics.image_width = image_width;
    intrinsics.image_height = image_height;
    intrinsics.fx = read_float_node(handle_, options_.intrinsic_nodes.fx);
    intrinsics.fy = read_float_node(handle_, options_.intrinsic_nodes.fy);
    intrinsics.cx = read_float_node(handle_, options_.intrinsic_nodes.cx);
    intrinsics.cy = read_float_node(handle_, options_.intrinsic_nodes.cy);
    if (options_.intrinsic_nodes.has_any_distortion_node()) {
        if (!options_.intrinsic_nodes.has_complete_distortion_nodes()) {
            throw std::runtime_error(
                "distortion node names are incomplete; provide k1,k2,p1,p2,k3 or omit all of them");
        }
        intrinsics.distortion = {
            read_float_node(handle_, options_.intrinsic_nodes.k1),
            read_float_node(handle_, options_.intrinsic_nodes.k2),
            read_float_node(handle_, options_.intrinsic_nodes.p1),
            read_float_node(handle_, options_.intrinsic_nodes.p2),
            read_float_node(handle_, options_.intrinsic_nodes.k3)};
    }
    intrinsics.source = "MVS GenICam nodes: fx=" + options_.intrinsic_nodes.fx
        + ",fy=" + options_.intrinsic_nodes.fy
        + ",cx=" + options_.intrinsic_nodes.cx
        + ",cy=" + options_.intrinsic_nodes.cy;
    validate_intrinsics_for_image(intrinsics, image_width, image_height);
    return intrinsics;
}

CameraIntrinsics HikvisionCamera::read_intrinsics(
    const int image_width,
    const int image_height) const {
    if (!device_open_ || handle_ == nullptr) {
        throw std::logic_error("HikvisionCamera::read_intrinsics requires open()");
    }
    if (calibration_intrinsics_.has_value()) {
        CameraIntrinsics intrinsics = *calibration_intrinsics_;
        validate_intrinsics_for_image(intrinsics, image_width, image_height);
        return intrinsics;
    }
    return read_intrinsics_from_nodes(image_width, image_height);
}

std::optional<CapturedFrame> HikvisionCamera::grab() {
    if (!grabbing_ || handle_ == nullptr) {
        throw std::logic_error("HikvisionCamera::grab requires start()");
    }
    if (options_.software_trigger) {
        check_sdk(
            MV_CC_SetCommandValue(handle_, "TriggerSoftware"),
            "MV_CC_SetCommandValue(TriggerSoftware)");
    }

    MV_FRAME_OUT output = {};
    const int code = MV_CC_GetImageBuffer(
        handle_, &output, static_cast<unsigned int>(options_.timeout_ms));
    if (code == MV_E_NODATA) {
        return std::nullopt;
    }
    check_sdk(code, "MV_CC_GetImageBuffer");
    ImageBufferGuard guard{handle_, &output};
    if (output.pBufAddr == nullptr) {
        throw std::runtime_error("MV_CC_GetImageBuffer returned a null image buffer");
    }

    const int width = static_cast<int>(output.stFrameInfo.nWidth);
    const int height = static_cast<int>(output.stFrameInfo.nHeight);
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("MVS returned an invalid image size");
    }
    const std::size_t pixel_count = static_cast<std::size_t>(width)
        * static_cast<std::size_t>(height);
    if (pixel_count > std::numeric_limits<std::size_t>::max() / 3U) {
        throw std::runtime_error("captured image is too large");
    }
    const std::size_t bgr_bytes = pixel_count * 3U;
    if (bgr_bytes > std::numeric_limits<unsigned int>::max()) {
        throw std::runtime_error("captured image buffer exceeds the MVS conversion size limit");
    }
    std::vector<unsigned char> bgr_buffer(bgr_bytes);
    MV_CC_PIXEL_CONVERT_PARAM convert = {};
    convert.nWidth = output.stFrameInfo.nWidth;
    convert.nHeight = output.stFrameInfo.nHeight;
    convert.pSrcData = output.pBufAddr;
    convert.nSrcDataLen = output.stFrameInfo.nFrameLen;
    convert.enSrcPixelType = output.stFrameInfo.enPixelType;
    convert.enDstPixelType = PixelType_Gvsp_BGR8_Packed;
    convert.pDstBuffer = bgr_buffer.data();
    convert.nDstBufferSize = static_cast<unsigned int>(bgr_buffer.size());
    check_sdk(MV_CC_ConvertPixelType(handle_, &convert), "MV_CC_ConvertPixelType(BGR8)");

    cv::Mat bgr_view(height, width, CV_8UC3, bgr_buffer.data());
    CapturedFrame frame;
    frame.capture_index = capture_index_++;
    frame.device_frame_number = static_cast<std::uint64_t>(output.stFrameInfo.nFrameNum);
    frame.host_timestamp_ns = host_timestamp_ns();
    frame.camera_id = camera_id_;
    frame.bgr = bgr_view.clone();
    frame.intrinsics = read_intrinsics(width, height);
    return frame;
}

void save_captured_frame(
    const CapturedFrame& frame,
    const CaptureOutputOptions& options) {
    if (options.output_dir.empty()) {
        throw std::invalid_argument("capture output directory must not be empty");
    }
    if (frame.bgr.empty() || frame.bgr.type() != CV_8UC3) {
        throw std::invalid_argument("captured frame must contain a non-empty CV_8UC3 BGR image");
    }
    if (!frame.intrinsics.valid()
        || frame.intrinsics.image_width != frame.bgr.cols
        || frame.intrinsics.image_height != frame.bgr.rows) {
        throw std::invalid_argument("captured frame intrinsics do not match its image");
    }
    if (options.png_compression < 0 || options.png_compression > 9) {
        throw std::invalid_argument("PNG compression must be in [0, 9]");
    }

    const std::string stem = frame_stem(frame.capture_index);
    const std::filesystem::path image_dir = options.output_dir / "images";
    const std::filesystem::path camera_dir = options.output_dir / "cameras";
    std::filesystem::create_directories(image_dir);
    std::filesystem::create_directories(camera_dir);
    const std::filesystem::path image_path = image_dir / (stem + ".png");
    const std::filesystem::path camera_path = camera_dir / (stem + ".json");
    if (std::filesystem::exists(image_path) || std::filesystem::exists(camera_path)) {
        throw std::runtime_error(
            "capture output already contains " + stem
            + "; use a new output directory instead of overwriting an existing frame");
    }

    if (!cv::imwrite(
            image_path.string(),
            frame.bgr,
            {cv::IMWRITE_PNG_COMPRESSION, options.png_compression})) {
        throw std::runtime_error("failed to write captured image: " + image_path.string());
    }

    const cv::Mat matrix = frame.intrinsics.camera_matrix();
    std::ofstream camera_file(camera_path, std::ios::out | std::ios::trunc);
    if (!camera_file) {
        throw std::runtime_error("failed to write camera metadata: " + camera_path.string());
    }
    camera_file << std::setprecision(12)
        << "{\n"
        << "  \"frame_id\": " << frame.capture_index << ",\n"
        << "  \"device_frame_number\": " << frame.device_frame_number << ",\n"
        << "  \"host_timestamp_ns\": " << frame.host_timestamp_ns << ",\n"
        << "  \"camera_id\": \"" << json_escape(frame.camera_id) << "\",\n"
        << "  \"image_path\": \"images/" << stem << ".png\",\n"
        << "  \"image_size\": [" << frame.bgr.cols << ", " << frame.bgr.rows << "],\n"
        << "  \"intrinsic_source\": \"" << json_escape(frame.intrinsics.source) << "\",\n"
        << "  \"intrinsic\": [\n"
        << "    [" << matrix.at<double>(0, 0) << ", " << matrix.at<double>(0, 1) << ", "
        << matrix.at<double>(0, 2) << "],\n"
        << "    [" << matrix.at<double>(1, 0) << ", " << matrix.at<double>(1, 1) << ", "
        << matrix.at<double>(1, 2) << "],\n"
        << "    [" << matrix.at<double>(2, 0) << ", " << matrix.at<double>(2, 1) << ", "
        << matrix.at<double>(2, 2) << "]\n"
        << "  ],\n"
        << "  \"distortion\": [";
    for (std::size_t index = 0U; index < frame.intrinsics.distortion.size(); ++index) {
        if (index != 0U) {
            camera_file << ", ";
        }
        camera_file << frame.intrinsics.distortion[index];
    }
    camera_file << "]\n}\n";
    if (!camera_file) {
        throw std::runtime_error("failed while writing camera metadata: " + camera_path.string());
    }

    const std::filesystem::path manifest_path = options.output_dir / "frames.csv";
    const bool write_header = !std::filesystem::exists(manifest_path)
        || std::filesystem::file_size(manifest_path) == 0U;
    std::ofstream manifest(manifest_path, std::ios::out | std::ios::app);
    if (!manifest) {
        throw std::runtime_error("failed to open capture manifest: " + manifest_path.string());
    }
    if (write_header) {
        manifest << "frame_id,device_frame_number,host_timestamp_ns,camera_id,image_path,"
            "camera_json,width,height,fx,fy,cx,cy,intrinsic_source\n";
    }
    manifest << frame.capture_index << ','
        << frame.device_frame_number << ','
        << frame.host_timestamp_ns << ','
        << csv_escape(frame.camera_id) << ','
        << csv_escape((std::filesystem::path("images") / (stem + ".png")).generic_string()) << ','
        << csv_escape((std::filesystem::path("cameras") / (stem + ".json")).generic_string()) << ','
        << frame.bgr.cols << ',' << frame.bgr.rows << ','
        << frame.intrinsics.fx << ',' << frame.intrinsics.fy << ','
        << frame.intrinsics.cx << ',' << frame.intrinsics.cy << ','
        << csv_escape(frame.intrinsics.source) << '\n';
    if (!manifest) {
        throw std::runtime_error("failed while writing capture manifest: " + manifest_path.string());
    }
}

void capture_to_directory(
    HikvisionCamera& camera,
    const CaptureOutputOptions& output,
    const std::uint64_t max_frames,
    const CaptureCallback& callback) {
    std::uint64_t saved = 0U;
    while (max_frames == 0U || saved < max_frames) {
        const std::optional<CapturedFrame> frame = camera.grab();
        if (!frame.has_value()) {
            continue;
        }
        save_captured_frame(*frame, output);
        ++saved;
        if (callback && !callback(*frame)) {
            break;
        }
    }
}

}  // namespace omnivggt::hikvision
