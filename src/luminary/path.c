#include <string.h>

#include "internal_error.h"
#include "internal_path.h"

#define PATH_BUFFER_SIZE (1u << 14)

LuminaryResult luminary_path_create(Path** _path) {
  __CHECK_NULL_ARGUMENT(_path);

  Path* path;
  __FAILURE_HANDLE(host_malloc(&path, sizeof(Path)));

  __FAILURE_HANDLE(host_malloc(&path->working_dir, PATH_BUFFER_SIZE));
  __FAILURE_HANDLE(host_malloc(&path->file, PATH_BUFFER_SIZE));
  __FAILURE_HANDLE(host_malloc(&path->output, PATH_BUFFER_SIZE));

  memset(path->working_dir, 0, PATH_BUFFER_SIZE);
  memset(path->file, 0, PATH_BUFFER_SIZE);
  memset(path->output, 0, PATH_BUFFER_SIZE);

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_path_destroy(Path** path) {
  __CHECK_NULL_ARGUMENT(path);
  __CHECK_NULL_ARGUMENT(*path);

  __FAILURE_HANDLE(host_free(&(*path)->working_dir));
  __FAILURE_HANDLE(host_free(&(*path)->file));
  __FAILURE_HANDLE(host_free(&(*path)->output));

  __FAILURE_HANDLE(host_free(path));

  return LUMINARY_SUCCESS;
}
