cmake_minimum_required(VERSION 3.18)

project(omnivggt_edge_cpp_linux LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

option(
    OMNIVGGT_ENABLE_LIVE_OBSERVER
    "Build the independent C++ streaming server, viewer, and history tools"
    OFF)

set(LIBTORCH_ROOT "$ENV{LIBTORCH}" CACHE PATH "Path to an extracted Linux LibTorch directory")
if(NOT LIBTORCH_ROOT)
    message(FATAL_ERROR
        "LIBTORCH_ROOT is not set. Set LIBTORCH or pass -DLIBTORCH_ROOT=/path/to/libtorch.")
endif()
get_filename_component(LIBTORCH_ROOT "${LIBTORCH_ROOT}" ABSOLUTE)

set(LIBTORCH_CONFIG "${LIBTORCH_ROOT}/share/cmake/Torch/TorchConfig.cmake")
if(NOT EXISTS "${LIBTORCH_CONFIG}")
    message(FATAL_ERROR
        "Linux LibTorch CMake package was not found at ${LIBTORCH_CONFIG}. "
        "Point LIBTORCH_ROOT at the extracted LibTorch directory.")
endif()

list(PREPEND CMAKE_PREFIX_PATH "${LIBTORCH_ROOT}")
find_package(Torch REQUIRED CONFIG)

# LibTorch publishes the ABI/compiler flags needed by its C++ API through
# TORCH_CXX_FLAGS.  Keep those flags on every target that includes torch.
if(TORCH_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
endif()

set(OMNIVGGT_OPENCV_COMPONENTS core imgcodecs imgproc features2d calib3d photo)
if(OMNIVGGT_ENABLE_LIVE_OBSERVER)
    list(APPEND OMNIVGGT_OPENCV_COMPONENTS highgui)
endif()
find_package(OpenCV REQUIRED COMPONENTS ${OMNIVGGT_OPENCV_COMPONENTS})
find_package(Threads REQUIRED)

set(OMNIVGGT_RUNTIME_RPATH "${LIBTORCH_ROOT}/lib")
if(OpenCV_LIBRARY_DIRS)
    list(APPEND OMNIVGGT_RUNTIME_RPATH ${OpenCV_LIBRARY_DIRS})
endif()

function(configure_omnivggt_target target_name)
    target_link_libraries(${target_name} PRIVATE
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
        Threads::Threads
        ${CMAKE_DL_LIBS}
    )

    if(TORCH_INCLUDE_DIRS)
        target_include_directories(${target_name} PRIVATE ${TORCH_INCLUDE_DIRS})
    endif()
    target_include_directories(${target_name} PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_features(${target_name} PRIVATE cxx_std_17)
    set_target_properties(${target_name} PROPERTIES
        BUILD_RPATH "${OMNIVGGT_RUNTIME_RPATH}"
        INSTALL_RPATH "${OMNIVGGT_RUNTIME_RPATH}"
    )
endfunction()

add_executable(omnivggt_edge
    src/omnivggt_edge.cpp
)
configure_omnivggt_target(omnivggt_edge)

add_executable(omnivggt_stream
    src/omnivggt_stream.cpp
)
configure_omnivggt_target(omnivggt_stream)

# The live observer is deliberately opt-in so the two original command-line
# programs keep their small dependency graph unless the viewer is requested.
if(OMNIVGGT_ENABLE_LIVE_OBSERVER)
    add_library(omnivggt_observer_history
        src/observer/stream_types.cpp
        src/observer/version_store.cpp
        src/observer/history_compactor.cpp
        src/observer/protocol.cpp
        src/observer/tcp_transport.cpp
    )
    target_include_directories(omnivggt_observer_history
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/src/observer
            ${OpenCV_INCLUDE_DIRS}
    )
    target_compile_features(omnivggt_observer_history PUBLIC cxx_std_17)
    # protocol.hpp now exposes SubmitFrameMessage{cv::Mat}, so OpenCV core is a
    # PUBLIC dependency of the history library (consumed by the viewer, replay
    # log and validate_history tools in addition to the inference/server targets).
    target_link_libraries(omnivggt_observer_history
        PUBLIC
            Threads::Threads
            ${OpenCV_LIBS}
    )
    set_target_properties(omnivggt_observer_history PROPERTIES
        BUILD_RPATH "${OMNIVGGT_RUNTIME_RPATH}"
        INSTALL_RPATH "${OMNIVGGT_RUNTIME_RPATH}"
    )

    add_library(omnivggt_observer_inference
        src/observer/frame_source.cpp
        src/observer/live_frame_source.cpp
        src/observer/inference_pipeline.cpp
        src/observer/group_world_fusion.cpp
    )
    target_include_directories(omnivggt_observer_inference
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/src/observer
            ${TORCH_INCLUDE_DIRS}
            ${OpenCV_INCLUDE_DIRS}
    )
    target_link_libraries(omnivggt_observer_inference
        PUBLIC
            omnivggt_observer_history
            ${TORCH_LIBRARIES}
            ${OpenCV_LIBS}
            ${CMAKE_DL_LIBS}
    )
    target_compile_options(omnivggt_observer_inference PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_features(omnivggt_observer_inference PUBLIC cxx_std_17)
    set_target_properties(omnivggt_observer_inference PROPERTIES
        BUILD_RPATH "${OMNIVGGT_RUNTIME_RPATH}"
        INSTALL_RPATH "${OMNIVGGT_RUNTIME_RPATH}"
    )

    add_executable(omnivggt_stream_server
        src/observer/stream_server_main.cpp
        src/observer/pointcloud_export.cpp
    )
    target_link_libraries(omnivggt_stream_server
        PRIVATE
            omnivggt_observer_inference
            ${TORCH_LIBRARIES}
            ${OpenCV_LIBS}
            ${CMAKE_DL_LIBS}
    )

    add_executable(omnivggt_live_viewer
        src/observer/live_viewer_main.cpp
        src/observer/pointcloud_export.cpp
    )
    target_include_directories(omnivggt_live_viewer
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src/observer
            ${OpenCV_INCLUDE_DIRS}
    )
    target_link_libraries(omnivggt_live_viewer
        PRIVATE
            omnivggt_observer_history
            ${OpenCV_LIBS}
    )

    add_executable(omnivggt_replay_log
        src/observer/replay_log_main.cpp
        src/observer/pointcloud_export.cpp
    )
    target_include_directories(omnivggt_replay_log PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/observer
        ${OpenCV_INCLUDE_DIRS}
    )
    target_link_libraries(omnivggt_replay_log PRIVATE
        omnivggt_observer_history
        ${OpenCV_LIBS}
    )

    add_executable(omnivggt_validate_history
        src/observer/validate_history_main.cpp
    )
    target_include_directories(omnivggt_validate_history PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/observer
    )
    target_link_libraries(omnivggt_validate_history PRIVATE omnivggt_observer_history)

    add_executable(omnivggt_observer_core_smoke
        src/observer/observer_core_smoke.cpp
    )
    target_include_directories(omnivggt_observer_core_smoke PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/observer
    )
    target_link_libraries(omnivggt_observer_core_smoke PRIVATE omnivggt_observer_history)

    set(OMNIVGGT_OBSERVER_TARGETS
        omnivggt_observer_history
        omnivggt_observer_inference
        omnivggt_stream_server
        omnivggt_live_viewer
        omnivggt_replay_log
        omnivggt_validate_history
        omnivggt_observer_core_smoke)
    foreach(observer_target IN LISTS OMNIVGGT_OBSERVER_TARGETS)
        target_compile_options(${observer_target} PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_features(${observer_target} PRIVATE cxx_std_17)
        set_target_properties(${observer_target} PROPERTIES
            BUILD_RPATH "${OMNIVGGT_RUNTIME_RPATH}"
            INSTALL_RPATH "${OMNIVGGT_RUNTIME_RPATH}"
        )
    endforeach()

    if(WIN32)
        target_link_libraries(omnivggt_observer_history PUBLIC ws2_32)
    endif()

    include(CTest)
    if(BUILD_TESTING)
        add_test(NAME omnivggt_observer_core_smoke
            COMMAND omnivggt_observer_core_smoke)
    endif()
endif()
