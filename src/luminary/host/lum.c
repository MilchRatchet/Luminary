#include "lum.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "internal_error.h"
#include "internal_path.h"
#include "lum/lum_binary.h"
#include "lum/lum_file_content.h"
#include "lum/lum_virtual_machine.h"
#include "utils.h"

#define LINE_SIZE 4096
#define CURRENT_VERSION 5
#define MIN_SUPPORTED_VERSION 4

LuminaryResult lum_file_parse_v5(FILE* file, LumBinary* binary);
LuminaryResult lum_file_parse_v4(FILE* file, LumFileContent* content);

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

  uint32_t magic_error = 0;

  magic_error += line[0] ^ 'L';
  magic_error += line[1] ^ 'u';
  magic_error += line[2] ^ 'm';
  magic_error += line[3] ^ 'i';
  magic_error += line[4] ^ 'n';
  magic_error += line[5] ^ 'a';
  magic_error += line[6] ^ 'r';
  magic_error += line[7] ^ 'y';

  if (magic_error != 0) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "File is not a Luminary file.")
  }

  fgets(line, LINE_SIZE, file);

  if (line[0] == 'v' || line[0] == 'V') {
    sscanf(line, "%*s %u\n", version);
  }
  else {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Luminary file has no version information.");
  }

  __FAILURE_HANDLE(host_free(&line));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_file_reset(LumFile* file) {
  __CHECK_NULL_ARGUMENT(file);

  switch (file->parsed_major_version) {
    case 4: {
      LumFileContent* content = (LumFileContent*) file->parsed_data;
      __FAILURE_HANDLE(lum_file_content_destroy(&content));
    } break;
    case 5: {
      LumBinary* binary = (LumBinary*) file->parsed_data;
      __FAILURE_HANDLE(lum_binary_destroy(&binary));
    } break;
    default:
      break;
  }

  file->parsed_major_version = LUM_FILE_MAJOR_VERSION_INVALID;
  file->parsed_data          = (void*) 0;

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_file_create(LumFile** file) {
  __CHECK_NULL_ARGUMENT(file);

  __FAILURE_HANDLE(host_malloc(file, sizeof(LumFile)));
  memset(*file, 0, sizeof(LumFile));

  __FAILURE_HANDLE(_lum_file_reset(*file));

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_file_parse(LumFile* file, Path* path) {
  __CHECK_NULL_ARGUMENT(file);
  __CHECK_NULL_ARGUMENT(path);

  __FAILURE_HANDLE(_lum_file_reset(file));

  const char* lum_file_path_string;
  __FAILURE_HANDLE(path_apply(path, (const char*) 0, &lum_file_path_string));

  FILE* file_handle = fopen(lum_file_path_string, "rb");

  if (file_handle == (FILE*) 0) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "File %s could not be opened.", lum_file_path_string);
  }

  __FAILURE_HANDLE(path_copy(&file->path, path));

  __FAILURE_HANDLE(_lum_validate_file(file_handle, &file->parsed_major_version));

  switch (file->parsed_major_version) {
    case 4: {
      LumFileContent* content;
      __FAILURE_HANDLE(lum_file_content_create(&content));
      __FAILURE_HANDLE(lum_file_parse_v4(file_handle, content));

      file->parsed_data = (void*) content;
    } break;
    case 5: {
      LumBinary* binary;
      __FAILURE_HANDLE(lum_binary_create(&binary));
      __FAILURE_HANDLE(lum_file_parse_v5(file_handle, binary));

      file->parsed_data = (void*) binary;
    } break;
    default:
      break;
  }

  fclose(file_handle);

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_file_apply(LumFile* file, LuminaryHost* host) {
  __CHECK_NULL_ARGUMENT(file);
  __CHECK_NULL_ARGUMENT(host);

  switch (file->parsed_major_version) {
    case LUM_FILE_MAJOR_VERSION_INVALID:
      warn_message("No Luminary file was parsed.");
      break;
    case 1:
    case 2:
    case 3:
      __RETURN_ERROR(
        LUMINARY_ERROR_API_EXCEPTION, "Luminary file is version %u but minimum supported version is %u.", file->parsed_major_version,
        MIN_SUPPORTED_VERSION);
    case 4: {
      LumFileContent* content = (LumFileContent*) file->parsed_data;
      __FAILURE_HANDLE(lum_file_content_apply(content, host, file->path));
    } break;
    case 5: {
      LumBinary* binary = (LumBinary*) file->parsed_data;

      LumVirtualMachineExecutionInfo info;
      info.debug_mode = false;

      LumVirtualMachine* vm;
      __FAILURE_HANDLE(lum_virtual_machine_create(&vm));
      __FAILURE_HANDLE(lum_virtual_machine_execute(vm, host, binary, &info));
      __FAILURE_HANDLE(lum_virtual_machine_destroy(&vm));
    } break;
    default:
      __RETURN_ERROR(
        LUMINARY_ERROR_API_EXCEPTION, "Luminary file is version %u is unknown. Current supported range [%u, %u].",
        file->parsed_major_version, MIN_SUPPORTED_VERSION, CURRENT_VERSION);
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_file_destroy(LumFile** file) {
  __CHECK_NULL_ARGUMENT(file);
  __CHECK_NULL_ARGUMENT(*file);

  __FAILURE_HANDLE(_lum_file_reset(*file));

  if ((*file)->path) {
    __FAILURE_HANDLE(luminary_path_destroy(&(*file)->path));
  }

  __FAILURE_HANDLE(host_free(file));

  return LUMINARY_SUCCESS;
}
