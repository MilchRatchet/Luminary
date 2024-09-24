#ifndef LUMINARY_INTERNAL_PATH_H
#define LUMINARY_INTERNAL_PATH_H

#include "utils.h"

struct LuminaryPath {
  char* working_dir;
  char* file;
  char* output;
} typedef LuminaryPath;

/*
 * Returns an ASCII encoded string corresponding to the given path. If override_path is not null,
 * then if override is an absolute override_path override will be returned, else a string containing the absolute
 * path given by interpreting override_path relative to the given path. The string must not be freed and is only valid until
 * the next path_* function call on this path object.
 * @param path Path instance.
 * @param override_path String containing a path. Optional.
 * @param string The destination the address of the string will be written to.
 *               If the computed path is empty, an empty string will be returned.
 */
LuminaryResult path_apply(Path* path, const char* override_path, const char** string);

#endif /* LUMINARY_INTERNAL_PATH_H */
