#ifndef LUMINARY_INTERNAL_PATH_H
#define LUMINARY_INTERNAL_PATH_H

#include "utils.h"

struct LuminaryPath {
  uint16_t working_dir_len;
  uint16_t file_path_len;
  uint16_t output_memory_available;
  char* memory;
  char* working_dir;
  char* file_path;
  char* output;
} typedef LuminaryPath;

/*
 * Creates an identical copy of a path instance. This allows Luminary to use a API side given path asynchronously as the API side could
 * possibly destroy the path in the meantime. The output path is created as if luminary_path_create was called and must be destroyed using
 * luminary_path_destroy.
 * @param path Path instance.
 * @param src_path Path instance to copy from.
 */
LuminaryResult path_copy(Path** path, const Path* src_path);

/*
 * Creates a new path instance by extending the working directory of the src path by the directory given in extension and overrides the file
 * name to the one given in extension. If extension is an absolute path, this function is identical to luminary_path_create and
 * luminary_path_set_from_string. The output path is created as if luminary_path_create was called and must be destroyed using
 * luminary_path_destroy.
 * @param path Path instance.
 * @param src_path Path instance to copy from.
 * @param entension String that defines the
 */
LuminaryResult path_extend(Path** path, const Path* src_path, const char* extension);

#endif /* LUMINARY_INTERNAL_PATH_H */
