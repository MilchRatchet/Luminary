#include <string.h>

#include "internal_error.h"
#include "lum.h"
#include "lum/lum_binary.h"
#include "lum/lum_parser.h"
#include "lum/lum_virtual_machine.h"

LuminaryResult lum_file_parse_v5(FILE* file, LumBinary* binary) {
  __CHECK_NULL_ARGUMENT(file);
  __CHECK_NULL_ARGUMENT(binary);

  size_t read_chars = 0;
  size_t file_size  = 4096;
  char* code;
  __FAILURE_HANDLE(host_malloc(&code, file_size));

  // 4GB limit
  while (file_size < UINT32_MAX) {
    read_chars += fread(code + read_chars, 1, file_size - read_chars, file);

    if (read_chars != file_size) {
      break;
    }

    file_size *= 2;
    __FAILURE_HANDLE(host_realloc(&code, file_size));
  }

  memset(code + read_chars, 0, file_size - read_chars);

  LumParser* parser;
  __FAILURE_HANDLE(lum_parser_create(&parser));

  __FAILURE_HANDLE(lum_parser_execute(parser, code, binary));

  __FAILURE_HANDLE(lum_binary_print(binary));

  __FAILURE_HANDLE(lum_parser_destroy(&parser));

  __FAILURE_HANDLE(host_free(&code));

  return LUMINARY_SUCCESS;
}
