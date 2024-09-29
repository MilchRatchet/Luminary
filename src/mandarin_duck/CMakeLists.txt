#=============================================================
# Find packages
#=============================================================

find_package(SDL3 REQUIRED CONFIG)
find_package(SDL3_ttf REQUIRED CONFIG)

#=============================================================
# Add executable
#=============================================================

add_executable(LuminaryMD)

#=============================================================
# Sources
#=============================================================

file(GLOB mandarin_duck_source_files "${CMAKE_CURRENT_SOURCE_DIR}/*.c")

target_sources(LuminaryMD PRIVATE ${mandarin_duck_source_files})

#=============================================================
# Include
#=============================================================

target_include_directories(LuminaryMD PRIVATE SDL3::Headers)
target_include_directories(LuminaryMD PRIVATE SDL3_ttf::SDL3_ttf)
target_include_directories(LuminaryMD PRIVATE Luminary::Headers)

#=============================================================
# Linking
#=============================================================

target_link_libraries(LuminaryMD PRIVATE SDL3::SDL3)
target_link_libraries(LuminaryMD PRIVATE SDL3_ttf::SDL3_ttf)
target_link_libraries(LuminaryMD PRIVATE Luminary::Library)
