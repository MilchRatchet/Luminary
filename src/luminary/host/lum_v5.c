#include <string.h>

#include "internal_error.h"
#include "lum.h"
#include "lum/lum_binary.h"
#include "lum/lum_parser.h"
#include "lum/lum_virtual_machine.h"

LuminaryResult lum_file_parse_v5(FILE* file, LumBinary* binary) {
  __CHECK_NULL_ARGUMENT(file);
  __CHECK_NULL_ARGUMENT(binary);

  // Keep the payload below UINT32_MAX so the tokenizer can represent the final NUL offset.
  const size_t max_code_size = (size_t) 0x7FFFFFFF * 2;
  size_t read_chars          = 0;
  size_t code_capacity       = 4096;
  char* code;
  __FAILURE_HANDLE(host_malloc(&code, code_capacity + 1));

  while (true) {
    if (read_chars == code_capacity) {
      if (code_capacity == max_code_size) {
        if (fgetc(file) != EOF) {
          __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Luminary V5 source exceeds the maximum supported size.");
        }

        if (ferror(file)) {
          __RETURN_ERROR(LUMINARY_ERROR_C_STD, "Failed to read Luminary V5 source.");
        }

        break;
      }

      const size_t new_capacity = (code_capacity > max_code_size / 2) ? max_code_size : code_capacity * 2;

      __FAILURE_HANDLE(host_realloc(&code, new_capacity + 1));
      code_capacity = new_capacity;
    }

    const size_t bytes_read = fread(code + read_chars, 1, code_capacity - read_chars, file);
    read_chars += bytes_read;

    if (ferror(file)) {
      __RETURN_ERROR(LUMINARY_ERROR_C_STD, "Failed to read Luminary V5 source.");
    }

    if (feof(file)) {
      break;
    }

    if (bytes_read == 0) {
      __RETURN_ERROR(LUMINARY_ERROR_C_STD, "Luminary V5 source read made no progress.");
    }
  }

  if (memchr(code, '\0', read_chars) != (void*) 0) {
    __RETURN_ERROR(LUMINARY_ERROR_API_EXCEPTION, "Luminary V5 source contains an embedded NUL byte.");
  }

  code[read_chars] = '\0';

  LumParser* parser;
  __FAILURE_HANDLE(lum_parser_create(&parser));

  __FAILURE_HANDLE(lum_parser_execute(parser, code, binary));

  __FAILURE_HANDLE(lum_binary_print(binary));

  __FAILURE_HANDLE(lum_parser_destroy(&parser));

  __FAILURE_HANDLE(host_free(&code));

  return LUMINARY_SUCCESS;
}
