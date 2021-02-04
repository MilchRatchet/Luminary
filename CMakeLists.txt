# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

project ("Luminary")

enable_language(CUDA)

set(CMAKE_CUDA_ARCHITECTURES 61)

# Include sub-projects.
add_subdirectory ("Luminary")
