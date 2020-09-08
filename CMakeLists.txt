project(project02)
cmake_minimum_required(VERSION 3.15.1 FATAL_ERROR)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")
set(CMAKE_BUILD_TYPE "Debug")

# 使CUDA NVCC 编译器支持C++11特性
LIST(APPEND CUDA_NVCC_FLAGS -std=c++11;-O2)
LIST(APPEND CUDA_NVCC_FLAGS -Xcompiler;-fPIC)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_50,code=sm_50)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_61,code=sm_61)

#set(OpenCV_DIR "opencv-3.2.0/build")
set(OpenCV_INCLUDE_DIRS "opencv348/include")
set(OPENCV_LIB_DIR opencv348/lib)
set(OPENCV_LIBS opencv_calib3d;opencv_core;opencv_cudaarithm;opencv_cudabgsegm;opencv_cudacodec;opencv_cudafeatures2d;opencv_cudafilters;opencv_cudaimgproc;opencv_cudaobjdetect;opencv_cudaoptflow;opencv_cudastereo;opencv_cudawarping;opencv_cudev;opencv_features2d;opencv_flann;opencv_highgui;opencv_imgcodecs;opencv_imgproc;opencv_ml;opencv_objdetect;opencv_photo;opencv_shape;opencv_stitching;opencv_superres;opencv_video;opencv_videoio;opencv_videostab)
include_directories(${OPENCV_INLCUDE_DIR})
link_directories(${OPENCV_LIB_DIR})

#cuda
find_package(CUDA REQUIRED)
message(STATUS "CUDA library status:")
message(STATUS "    version: ${CUDA_VERSION}")
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    libraries: ${CUDA_CUDA_LIBRARY}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
#INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(/usr/local/cuda-10.1/targets/x86_64-linux/include)
link_directories(/usr/local/cuda-10.1/targets/x86_64-linux/lib)

#tensorrt
set(tensorRT_INCLUDE_DIR TensorRT-5.1.5.0/include)
set(tensorRT_LIB_DIR TensorRT-5.1.5.0/lib)
set(tensorRT_LIBS nvinfer nvinfer_plugin nvparsers nvonnxparser nvonnxparser_runtime)
include_directories(${tensorRT_INCLUDE_DIR})
include_directories(TensorRT-5.1.5.0/samples/common)
link_directories(${tensorRT_LIB_DIR})
set(logger_file "TensorRT-5.1.5.0/samples/common/logger.cpp")


#aux_source_directory(. SRC_LIST)
add_executable(${PROJECT_NAME} crowdcount.cpp ${logger_file})
target_link_libraries(${PROJECT_NAME} ${tensorRT_LIBS} ${OPENCV_LIBS} ${CUDA_nvcuvid_LIBRARY} ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY} cudart)

