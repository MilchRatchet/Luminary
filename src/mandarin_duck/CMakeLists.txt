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

file(GLOB mandarin_duck_source_files "${CMAKE_CURRENT_SOURCE_DIR}/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/elements/*.c" "${CMAKE_CURRENT_SOURCE_DIR}/windows/*.c")

target_sources(LuminaryMD PRIVATE ${mandarin_duck_source_files})

#=============================================================
# Include
#=============================================================

target_include_directories(LuminaryMD PRIVATE SDL3::Headers)
target_include_directories(LuminaryMD PRIVATE SDL3_ttf::SDL3_ttf)
target_include_directories(LuminaryMD PRIVATE Luminary::Headers)
target_include_directories(LuminaryMD PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

#=============================================================
# Linking
#=============================================================

target_link_libraries(LuminaryMD PRIVATE SDL3::SDL3)
target_link_libraries(LuminaryMD PRIVATE SDL3_ttf::SDL3_ttf)
target_link_libraries(LuminaryMD PRIVATE Luminary::Library)

#=============================================================
# Print summary
#=============================================================

message("========================== Mandarin Duck ==========================")
message("SDL3 Directory                 ||  ${SDL3_DIR}")
message("SDL3_ttf Directory             ||  ${SDL3_ttf_DIR}")
