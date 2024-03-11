cmake_minimum_required(VERSION 3.16)

project(DisplayImage VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(OpenCV_DIR "/home/ismail/Downloads/build/opencv4")

find_package( OpenCV REQUIRED )

# include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( DisplayImage main.cpp )
target_link_libraries( DisplayImage ${OpenCV_LIBS} )

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
