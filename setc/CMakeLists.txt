cmake_minimum_required(VERSION 3.18)

project(omnivggt_edge_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(
    OMNIVGGT_ENABLE_LIVE_OBSERVER
    "Build the independent C++ streaming server, viewer, and history tools"
    OFF)

set(OMNIVGGT_OPENCV_COMPONENTS core imgcodecs imgproc features2d calib3d)
if(OMNIVGGT_ENABLE_LIVE_OBSERVER)
    list(APPEND OMNIVGGT_OPENCV_COMPONENTS highgui)
endif()
find_package(OpenCV REQUIRED COMPONENTS ${OMNIVGGT_OPENCV_COMPONENTS})

set(LIBTORCH_ROOT "$ENV{LIBTORCH}" CACHE PATH "Path to an extracted LibTorch directory")
if(NOT LIBTORCH_ROOT)
    set(LIBTORCH_ROOT "C:/Dev/libtorch/2.7.0-cu128")
endif()

set(LIBTORCH_INCLUDE_DIRS
    "${LIBTORCH_ROOT}/include"
    "${LIBTORCH_ROOT}/include/torch/csrc/api/include"
)

set(LIBTORCH_LIB_DIR "${LIBTORCH_ROOT}/lib")
set(LIBTORCH_LIBS
    "${LIBTORCH_LIB_DIR}/torch.lib"
    "${LIBTORCH_LIB_DIR}/torch_cpu.lib"
    "${LIBTORCH_LIB_DIR}/torch_cuda.lib"
    "${LIBTORCH_LIB_DIR}/c10.lib"
    "${LIBTORCH_LIB_DIR}/c10_cuda.lib"
    "${LIBTORCH_LIB_DIR}/caffe2_nvrtc.lib"
)

foreach(required_libtorch_file IN LISTS LIBTORCH_LIBS)
    if(NOT EXISTS "${required_libtorch_file}")
        message(FATAL_ERROR "Missing LibTorch library: ${required_libtorch_file}")
    endif()
endforeach()

function(configure_omnivggt_target target_name)
    target_link_libraries(${target_name} PRIVATE
        ${LIBTORCH_LIBS}
        ${OpenCV_LIBS}
    )

    target_include_directories(${target_name} PRIVATE
        ${LIBTORCH_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
    )

    if(MSVC)
        target_compile_options(${target_name} PRIVATE /W4 /EHsc /bigobj /utf-8)
    else()
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    set_property(TARGET ${target_name} PROPERTY CXX_STANDARD 17)
endfunction()

add_executable(omnivggt_edge
    src/omnivggt_edge.cpp
)
configure_omnivggt_target(omnivggt_edge)

add_executable(omnivggt_stream
    src/omnivggt_stream.cpp
)
configure_omnivggt_target(omnivggt_stream)

# The live observer is deliberately opt-in.  Existing targets and the original
# C++ runner retain their current dependency graph and behavior when this option
# is OFF.  The observer uses a small versioned TCP protocol instead of making
# gRPC/GLFW/Zstd mandatory, so it remains drop-in on the already validated
# OpenCV + LibTorch Windows environment.
if(OMNIVGGT_ENABLE_LIVE_OBSERVER)
    find_package(Threads REQUIRED)

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
    )
    target_compile_features(omnivggt_observer_history PUBLIC cxx_std_17)
    target_link_libraries(omnivggt_observer_history PUBLIC Threads::Threads)

    add_library(omnivggt_observer_inference
        src/observer/frame_source.cpp
        src/observer/live_frame_source.cpp
        src/observer/inference_pipeline.cpp
    )
    target_include_directories(omnivggt_observer_inference
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/src/observer
            ${LIBTORCH_INCLUDE_DIRS}
            ${OpenCV_INCLUDE_DIRS}
    )
    target_link_libraries(omnivggt_observer_inference
        PUBLIC
            omnivggt_observer_history
            ${LIBTORCH_LIBS}
            ${OpenCV_LIBS}
    )

    add_executable(omnivggt_stream_server
        src/observer/stream_server_main.cpp
    )
    target_link_libraries(omnivggt_stream_server
        PRIVATE
            omnivggt_observer_inference
            ${LIBTORCH_LIBS}
            ${OpenCV_LIBS}
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
    target_link_libraries(omnivggt_replay_log PRIVATE omnivggt_observer_history ${OpenCV_LIBS})

    add_executable(omnivggt_validate_history
        src/observer/validate_history_main.cpp
    )
    target_include_directories(omnivggt_validate_history PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/observer)
    target_link_libraries(omnivggt_validate_history PRIVATE omnivggt_observer_history)

    add_executable(omnivggt_observer_core_smoke
        src/observer/observer_core_smoke.cpp
    )
    target_include_directories(omnivggt_observer_core_smoke PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/observer)
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
        if(MSVC)
            target_compile_options(${observer_target} PRIVATE /W4 /EHsc /bigobj /utf-8)
        else()
            target_compile_options(${observer_target} PRIVATE -Wall -Wextra -Wpedantic)
        endif()
        set_property(TARGET ${observer_target} PROPERTY CXX_STANDARD 17)
    endforeach()

    if(WIN32)
        target_link_libraries(omnivggt_observer_history PUBLIC ws2_32)
    endif()
endif()
