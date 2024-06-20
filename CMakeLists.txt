# Specify CMake Minimum Version
cmake_minimum_required(VERSION 3.0)

# Project Declaration
project(Spin_image)


# Set C++ Standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Specify OpenMesh installation directory
set(OPENMESH_DIR /usr/local)

# Find OpenMesh package
find_package(OpenMesh REQUIRED PATHS ${OPENMESH_DIR})
find_package(OpenCV REQUIRED)
find_package(TBB REQUIRED)

# Include Directories
include_directories(${OPENMESH_INCLUDE_DIRS})


# Add Executable
add_executable(spin main.cpp)

# Link OpenMesh
target_link_libraries(spin /usr/local/lib/libOpenMeshCore.so ${OpenCV_LIBS}  TBB::tbb)
target_include_directories(spin PRIVATE ${OpenCV_INCLUDE_DIRS})



