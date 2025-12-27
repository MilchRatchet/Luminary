#include "lum_parser.h"

#include <string.h>

#include "internal_error.h"
#include "lum_compiler.h"
#include "lum_tokenizer.h"

LuminaryResult lum_parser_create(LumParser** parser) {
  __CHECK_NULL_ARGUMENT(parser);

  __FAILURE_HANDLE(host_malloc(parser, sizeof(LumParser)));
  memset(*parser, 0, sizeof(LumParser));

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_parser_execute(LumParser* parser, const char* code, LumBinary* binary) {
  __CHECK_NULL_ARGUMENT(parser);
  __CHECK_NULL_ARGUMENT(code);

  LumCompiler* compiler;
  __FAILURE_HANDLE(lum_compiler_create(&compiler));

  LumCompilerCompileInfo info;
  info.binary             = binary;
  info.code               = code;
  info.print_parsed_token = true;

  __FAILURE_HANDLE(lum_compiler_compile(compiler, &info));

  __FAILURE_HANDLE(lum_compiler_destroy(&compiler));

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_parser_destroy(LumParser** parser) {
  __CHECK_NULL_ARGUMENT(parser);
  __CHECK_NULL_ARGUMENT(*parser);

  __FAILURE_HANDLE(host_free(parser));

  return LUMINARY_SUCCESS;
}
