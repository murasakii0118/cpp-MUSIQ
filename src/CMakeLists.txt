cmake_minimum_required(VERSION 3.15)
project(MUSIQ_Inference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MINGW)
    set(CMAKE_C_COMPILER gcc)
    set(CMAKE_CXX_COMPILER g++)
endif()

set(OPENCV_DIR "${CMAKE_SOURCE_DIR}/../opencv4120-extra/x64/mingw")
set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/../onnx-mingw64")

set(OpenCV_DIR "${OPENCV_DIR}/lib")
find_package(OpenCV REQUIRED)

set(OPENCV_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../opencv4120-extra/include")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_DIR}/lib")

include_directories(
    ${OPENCV_INCLUDE_DIR}
    ${ONNXRUNTIME_INCLUDE_DIR}
)

add_executable(musiq_inference
    main.cpp
    musiq_preprocess.cpp
    musiq_inference.cpp
    musiq_postprocess.cpp
)

target_link_libraries(musiq_inference
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIB_DIR}/onnxruntime.lib
)

if(MINGW)
    target_link_libraries(musiq_inference
        ${ONNXRUNTIME_LIB_DIR}/onnxruntime_common.lib
        ${ONNXRUNTIME_LIB_DIR}/onnxruntime_flatbuffers.lib
        ${ONNXRUNTIME_LIB_DIR}/onnxruntime_framework.lib
        ${ONNXRUNTIME_LIB_DIR}/onnxruntime_graph.lib
        ${ONNXRUNTIME_LIB_DIR}/onnxruntime_mlas.lib
        ${ONNXRUNTIME_LIB_DIR}/onnxruntime_optimizer.lib
        ${ONNXRUNTIME_LIB_DIR}/onnxruntime_providers.lib
        ${ONNXRUNTIME_LIB_DIR}/onnxruntime_session.lib
        ${ONNXRUNTIME_LIB_DIR}/onnxruntime_util.lib
    )
endif()

set_target_properties(musiq_inference PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime lib: ${ONNXRUNTIME_LIB_DIR}")
