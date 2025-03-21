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

  LumTokenizer* tokenizer;
  __FAILURE_HANDLE(lum_tokenizer_create(&tokenizer));

  __FAILURE_HANDLE(lum_tokenizer_execute(tokenizer, code));

  __FAILURE_HANDLE(lum_tokenizer_print(tokenizer));

  LumCompiler* compiler;
  __FAILURE_HANDLE(lum_compiler_create(&compiler));

  __FAILURE_HANDLE(lum_compiler_compile(compiler, tokenizer->tokens, binary));

  __FAILURE_HANDLE(lum_compiler_destroy(&compiler));

  __FAILURE_HANDLE(lum_tokenizer_destroy(&tokenizer));

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_parser_destroy(LumParser** parser) {
  __CHECK_NULL_ARGUMENT(parser);
  __CHECK_NULL_ARGUMENT(*parser);

  __FAILURE_HANDLE(host_free(parser));

  return LUMINARY_SUCCESS;
}
