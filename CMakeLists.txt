# cmake needs this line
cmake_minimum_required(VERSION 3.1)

# Define project name
project(sobel_simd_opencv_project)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
set(OpenCV_DIR "~/Downloads/build")
find_package(OpenCV REQUIRED PATHS "~/Downloads/build")

# Enable SIMD instructions for Intel Intrinsics
# https://software.intel.com/sites/landingpage/IntrinsicsGuide/
# if(NOT WIN32)
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCV_DISABLE_OPTIMIZATION=ON -DCV_ENABLE_INTRINSICS=OFF -DENABLE_NEON=OFF")
# endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCV_DISABLE_OPTIMIZATION=ON -DCV_ENABLE_INTRINSICS=OFF -DENABLE_NEON=OFF")

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Declare the executable target built from your sources
add_executable(sobel_simd_opencv main.cpp)

# Link your application with OpenCV libraries
target_link_libraries(sobel_simd_opencv PRIVATE ${OpenCV_LIBS})

# Copy input file for image processing to build folder
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/monarch.jpg
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR})