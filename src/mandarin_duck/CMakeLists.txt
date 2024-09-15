#=============================================================
# Find packages
#=============================================================

find_package(SDL2 REQUIRED CONFIG)
find_package(SDL2_ttf REQUIRED CONFIG)

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

target_include_directories(LuminaryMD PRIVATE SDL2::Headers)
target_include_directories(LuminaryMD PRIVATE SDL2_ttf::SDL2_ttf)
target_include_directories(LuminaryMD PRIVATE Luminary::Headers)

#=============================================================
# Linking
#=============================================================

target_link_libraries(LuminaryMD PRIVATE SDL2::SDL2)
target_link_libraries(LuminaryMD PRIVATE SDL2_ttf::SDL2_ttf)
target_link_libraries(LuminaryMD PRIVATE Luminary::Shared)

#=============================================================
# File copying
#=============================================================

if(WIN32)
    add_custom_command(TARGET LuminaryMD POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:SDL2::SDL2>"
        "$<TARGET_FILE:SDL2_ttf::SDL2_ttf>"
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()
