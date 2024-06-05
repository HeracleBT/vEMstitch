cmake_minimum_required(VERSION 3.2)
project(vEMstitching_cpp)

# 设置 OpenCV 的路径
set(OpenCV_DIR "/usr/local/opencv4")

# 寻找 OpenCV 库
find_package(OpenCV REQUIRED)

# 输出 OpenCV 信息
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# 设置包含目录
include_directories(${PROJECT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})

# 设置库目录
link_directories(${OpenCV_LIB_DIR})

# 指定使用 g++-5 编译器
set(CMAKE_CXX_COMPILER g++)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
	message("OPENMP FOUND")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    # set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# 查找源文件
aux_source_directory(./src SrcFiles)

# 设置可执行文件输出路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# 添加可执行文件
add_executable(vEMstitching ${SrcFiles})

# 链接 OpenCV 库
target_link_libraries(vEMstitching PRIVATE ${OpenCV_LIBS})
