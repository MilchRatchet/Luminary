#include <string.h>

#include "internal_error.h"
#include "internal_path.h"

#define PATH_BUFFER_SIZE (1u << 14)

static bool _path_is_absolute(const char* string) {
  if (string == (const char*) 0)
    return false;

  // Windows C:, D:, ...
  const char* windows_disk_designator = strchr(string, ':');

  // Unix /usr/ ...
  return (windows_disk_designator || (string[0] == '/'));
}

LuminaryResult luminary_path_create(Path** _path) {
  __CHECK_NULL_ARGUMENT(_path);

  Path* path;
  __FAILURE_HANDLE(host_malloc(&path, sizeof(Path)));

  __FAILURE_HANDLE(host_malloc(&path->memory, PATH_BUFFER_SIZE));

  memset(path->memory, 0, PATH_BUFFER_SIZE);

  path->working_dir             = (char*) 0;
  path->file_path               = (char*) 0;
  path->output                  = (char*) 0;
  path->working_dir_len         = 0;
  path->file_path_len           = 0;
  path->output_memory_available = 0;

  *_path = path;

  return LUMINARY_SUCCESS;
}

LuminaryResult path_copy(Path** path, const Path* src_path) {
  __CHECK_NULL_ARGUMENT(path);
  __CHECK_NULL_ARGUMENT(src_path);

  __FAILURE_HANDLE(luminary_path_create(path));

  memcpy((*path)->memory, src_path->memory, PATH_BUFFER_SIZE);

  (*path)->working_dir_len         = src_path->working_dir_len;
  (*path)->file_path_len           = src_path->file_path_len;
  (*path)->output_memory_available = src_path->output_memory_available;

  (*path)->working_dir = (*path)->memory;
  (*path)->file_path   = (*path)->working_dir + (*path)->working_dir_len + 1;
  (*path)->output      = (*path)->file_path + (*path)->file_path_len + 1;

  return LUMINARY_SUCCESS;
}

LuminaryResult path_extend(Path** path, const Path* src_path, const char* extension) {
  __CHECK_NULL_ARGUMENT(path);
  __CHECK_NULL_ARGUMENT(src_path);
  __CHECK_NULL_ARGUMENT(extension);

  const bool extension_is_absolute_path = _path_is_absolute(extension);

  if (extension_is_absolute_path) {
    __FAILURE_HANDLE(luminary_path_create(path));
    __FAILURE_HANDLE(luminary_path_set_from_string(*path, extension));
  }
  else {
    __FAILURE_HANDLE(path_copy(path, src_path));

    const char* last_forward_slash  = strrchr(extension, '/');
    const char* last_backward_slash = strrchr(extension, '\\');

    const char* file_path = (last_forward_slash) ? last_forward_slash + 1 : extension;
    file_path             = (last_backward_slash) ? last_backward_slash + 1 : file_path;

    const size_t file_path_len = strlen(file_path);

    const char* working_dir = (file_path != extension) ? extension : (const char*) 0;

    if (working_dir) {
      const size_t working_dir_len = (working_dir) ? (size_t) (file_path - extension - 1) : 0;

      if ((*path)->output_memory_available < working_dir_len + 1) {
        __RETURN_ERROR(
          LUMINARY_ERROR_OUT_OF_MEMORY, "Path exceeds path buffer size after extension. Working dir size increased by %llu bytes.",
          working_dir_len + 1);
      }

#if defined(WIN32)
      (*path)->memory[(*path)->working_dir_len] = '\\';
#else  /* WIN32 */
      (*path)->memory[(*path)->working_dir_len] = '/';
#endif /* !WIN32 */

      memcpy((*path)->memory + (*path)->working_dir_len + 1, working_dir, working_dir_len);
      (*path)->working_dir_len += working_dir_len + 1;
      (*path)->memory[(*path)->working_dir_len] = '\0';
    }

    (*path)->file_path = (*path)->working_dir + (*path)->working_dir_len + 1;
    memcpy((*path)->file_path, file_path, file_path_len + 1);

    (*path)->file_path_len = file_path_len;

    (*path)->output = (*path)->file_path + file_path_len + 1;

    (*path)->output_memory_available = PATH_BUFFER_SIZE - (*path)->working_dir_len - (*path)->file_path_len - 2;
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_path_set_from_string(Path* path, const char* string) {
  __CHECK_NULL_ARGUMENT(path);
  __CHECK_NULL_ARGUMENT(string);

  const char* last_forward_slash  = strrchr(string, '/');
  const char* last_backward_slash = strrchr(string, '\\');

  const char* file_path = (last_forward_slash) ? last_forward_slash + 1 : string;
  file_path             = (last_backward_slash) ? last_backward_slash + 1 : file_path;

  const size_t file_path_len = strlen(file_path);

  const char* working_dir      = (file_path != string) ? string : (const char*) 0;
  const size_t working_dir_len = (working_dir) ? (size_t) (file_path - string - 1) : 0;

  if (working_dir_len + file_path_len + 3 > PATH_BUFFER_SIZE) {
    __RETURN_ERROR(
      LUMINARY_ERROR_OUT_OF_MEMORY, "Path exceeds path buffer size. What kind of paths are these? Lengths are %llu and %llu.",
      working_dir_len, file_path_len);
  }

  path->working_dir = path->memory;
  memcpy(path->working_dir, working_dir, working_dir_len);
  path->working_dir[working_dir_len] = '\0';

  path->working_dir_len = working_dir_len;

  path->file_path = path->working_dir + working_dir_len + 1;
  memcpy(path->file_path, file_path, file_path_len + 1);

  path->file_path_len = file_path_len;

  path->output = path->file_path + file_path_len + 1;

  path->output_memory_available = PATH_BUFFER_SIZE - working_dir_len - file_path_len - 2;

  return LUMINARY_SUCCESS;
}

static uint32_t _path_apply_working_dir(Path* path) {
  uint32_t offset = 0;

  if (path->working_dir_len == 0)
    return offset;

  if (_path_is_absolute(path->working_dir)) {
    memcpy(path->output + offset, path->working_dir, path->working_dir_len);
    offset += path->working_dir_len;
  }
  else {
    // Sanitize relative paths to start with './' or '.\'
    path->output[offset++] = '.';

#if defined(WIN32)
    path->output[offset++] = '\\';
#else  /* WIN32 */
    path->output[offset++] = '/';
#endif /* !WIN32 */

    uint32_t read_offset = 0;

    // Skip past the automatically inserted characters
    if (path->working_dir_len > read_offset && path->working_dir[read_offset] == '.')
      read_offset++;

#if defined(WIN32)
    if ((path->working_dir_len > read_offset) && ((path->working_dir[read_offset] == '\\') || (path->working_dir[read_offset] == '/')))
      read_offset++;
#else  /* WIN32 */
    if ((path->working_dir_len > read_offset) && (path->working_dir[read_offset] == '/'))
      read_offset++;
#endif /* !WIN32 */

    __DEBUG_ASSERT(path->working_dir_len >= read_offset);

    uint32_t copy_length = path->working_dir_len - read_offset;

    memcpy(path->output + offset, path->working_dir + read_offset, copy_length);
    offset += copy_length;
  }

#if defined(WIN32)
  path->output[offset++] = '\\';
#else  /* WIN32 */
  path->output[offset++] = '/';
#endif /* !WIN32 */

  return offset;
}

static LuminaryResult _path_apply_no_override(Path* path) {
  __CHECK_NULL_ARGUMENT(path);

  if (!path->working_dir || !path->file_path || !path->output) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Path was never initialized and no override was given either.");
  }

  if (path->working_dir_len + path->file_path_len + 2 > path->output_memory_available) {
    __RETURN_ERROR(
      LUMINARY_ERROR_OUT_OF_MEMORY, "Ran out of memory when combining the path. Lengths are %llu and %llu.", path->working_dir_len,
      path->file_path_len);
  }

  uint32_t file_path_offset = _path_apply_working_dir(path);

  memcpy(path->output + file_path_offset, path->file_path, path->file_path_len + 1);

  return LUMINARY_SUCCESS;
}

static LuminaryResult _path_apply_override(Path* path, const char* override) {
  __CHECK_NULL_ARGUMENT(path);
  __CHECK_NULL_ARGUMENT(override);

  const size_t override_len = strlen(override);

  if (override_len > path->output_memory_available) {
    __RETURN_ERROR(LUMINARY_ERROR_OUT_OF_MEMORY, "Override path exceeded path buffer size. Override length is %llu.", override_len);
  }

  // Path was never initialized, just output the override.
  if (!path->working_dir || !path->file_path || !path->output) {
    path->output_memory_available = PATH_BUFFER_SIZE;

    path->output = path->memory;
    memcpy(path->output, override, override_len + 1);

    return LUMINARY_SUCCESS;
  }

  const bool override_is_absolute_path = _path_is_absolute(override);

  if (override_is_absolute_path) {
    memcpy(path->output, override, override_len + 1);

    return LUMINARY_SUCCESS;
  }

  if (path->working_dir_len + override_len + 2 > path->output_memory_available) {
    __RETURN_ERROR(
      LUMINARY_ERROR_OUT_OF_MEMORY, "Ran out of memory when combining the path with the override. Lengths are %llu and %llu.",
      path->working_dir_len, override_len);
  }

  uint32_t file_path_offset = _path_apply_working_dir(path);

  memcpy(path->output + file_path_offset, override, override_len + 1);

  return LUMINARY_SUCCESS;
}

LuminaryResult path_apply(Path* path, const char* override, const char** string) {
  __CHECK_NULL_ARGUMENT(path);
  __CHECK_NULL_ARGUMENT(string);

  if (override) {
    __FAILURE_HANDLE(_path_apply_override(path, override));
  }
  else {
    __FAILURE_HANDLE(_path_apply_no_override(path));
  }

  *string = path->output;

  return LUMINARY_SUCCESS;
}

LuminaryResult luminary_path_destroy(Path** path) {
  __CHECK_NULL_ARGUMENT(path);
  __CHECK_NULL_ARGUMENT(*path);

  __FAILURE_HANDLE(host_free(&(*path)->memory));

  __FAILURE_HANDLE(host_free(path));

  return LUMINARY_SUCCESS;
}
