# CMakeList.txt : CMake project for Luminary, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)


# Add source to this project's executable.
add_executable (Luminary "Luminary.c" "Luminary.h" "lib/png.c" "lib/png.h" "lib/zlib/deflate.c" "lib/zlib/deflate.h" "lib/zlib/zutil.h" "lib/zlib/zutil.c" "lib/zlib/zlib.h" "lib/zlib/zconf.h" "lib/zlib/gzguts.h" "lib/zlib/crc32.c" "lib/zlib/crc32.h" "lib/zlib/adler32.c" "lib/zlib/trees.c" "lib/zlib/trees.h" "lib/zlib/compress.c" "lib/image.h" "lib/primitives.h" "lib/cudatest.cu")

# TODO: Add tests and install targets if needed.
