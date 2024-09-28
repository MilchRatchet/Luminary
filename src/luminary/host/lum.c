#include "lum.h"

#include <stdio.h>
#include <stdlib.h>

#include "camera.h"
#include "internal_error.h"
#include "internal_path.h"
#include "utils.h"

#define LINE_SIZE 4096
#define CURRENT_VERSION 5
#define MIN_SUPPORTED_VERSION 4

LuminaryResult lum_content_create(LumFileContent** _content) {
  __CHECK_NULL_ARGUMENT(_content);

  LumFileContent* content;
  __FAILURE_HANDLE(host_malloc(&content, sizeof(LumFileContent)));

  __FAILURE_HANDLE(array_create(&content->obj_file_path_strings, sizeof(char*), 16));

  __FAILURE_HANDLE(camera_get_default(&content->camera));

  *_content = content;

  return LUMINARY_SUCCESS;
}

/*
 * Determines whether the file is a supported *.lum file and on success puts the file accessor past the header.
 * @param file File handle.
 * @result 0 if file is supported, non zero value else.
 */
static LuminaryResult _lum_validate_file(FILE* file, uint32_t* version) {
  __CHECK_NULL_ARGUMENT(file);
  __CHECK_NULL_ARGUMENT(version);

  char* line;
  __FAILURE_HANDLE(host_malloc(&line, LINE_SIZE));

  fgets(line, LINE_SIZE, file);

  {
    uint32_t result = 0;

    result += line[0] ^ 'L';
    result += line[1] ^ 'u';
    result += line[2] ^ 'm';
    result += line[3] ^ 'i';
    result += line[4] ^ 'n';
    result += line[5] ^ 'a';
    result += line[6] ^ 'r';
    result += line[7] ^ 'y';

    if (result) {
      __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "File is not a Luminary file.")
    }
  }

  fgets(line, LINE_SIZE, file);

  if (line[0] == 'v' || line[0] == 'V') {
    sscanf(line, "%*c %u\n", version);
  }
  else {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Luminary file has no version information.");
  }

  __FAILURE_HANDLE(host_free(&line));

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_read_file(Path* path, LumFileContent* content) {
  __CHECK_NULL_ARGUMENT(path);
  __CHECK_NULL_ARGUMENT(content);

  const char* lum_file_path_string;
  __FAILURE_HANDLE(path_apply(path, (const char*) 0, &lum_file_path_string));

  FILE* file = fopen(lum_file_path_string, "rb");

  if (!file) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "File %s could not be opened.", lum_file_path_string);
  }

  uint32_t version;
  __FAILURE_HANDLE(_lum_validate_file(file, &version));

  switch (version) {
    case 1:
    case 2:
    case 3:
      __RETURN_ERROR(
        LUMINARY_ERROR_API_EXCEPTION, "Luminary file is version %u but minimum supported version is %u.", version, MIN_SUPPORTED_VERSION);
    case 4:
      __FAILURE_HANDLE(lum_parse_file_v4(file, content));
      break;
    case 5:
      __FAILURE_HANDLE(lum_parse_file_v5(file, content));
      break;
    default:
      __RETURN_ERROR(
        LUMINARY_ERROR_API_EXCEPTION, "Luminary file is version %u is unknown. Current supported range [%u, %u].", version,
        MIN_SUPPORTED_VERSION, CURRENT_VERSION);
  }

  fclose(file);

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_write_content(Path* path, LumFileContent* content) {
  __CHECK_NULL_ARGUMENT(path);
  __CHECK_NULL_ARGUMENT(content);

  __RETURN_ERROR(LUMINARY_ERROR_NOT_IMPLEMENTED, "Writing lum files is not yet implemented.");
}

LuminaryResult lum_content_destroy(LumFileContent** content) {
  __CHECK_NULL_ARGUMENT(content);
  __CHECK_NULL_ARGUMENT(*content);

  uint32_t num_obj_file_path_strings;
  __FAILURE_HANDLE(array_get_num_elements((*content)->obj_file_path_strings, &num_obj_file_path_strings));

  for (uint32_t i = 0; i < num_obj_file_path_strings; i++) {
    __FAILURE_HANDLE(host_free(&(*content)->obj_file_path_strings[i]));
  }

  __FAILURE_HANDLE(array_destroy(&(*content)->obj_file_path_strings));

  __FAILURE_HANDLE(host_free(content));

  return LUMINARY_SUCCESS;
}
