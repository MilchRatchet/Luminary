#include "lum_parser.h"

#include <string.h>

#include "internal_error.h"
#include "lum_tokenizer.h"

LuminaryResult lum_parser_create(LumParser** parser) {
  __CHECK_NULL_ARGUMENT(parser);

  __FAILURE_HANDLE(host_malloc(parser, sizeof(LumParser)));
  memset(*parser, 0, sizeof(LumParser));

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_parser_execute(LumParser* parser, const char* code) {
  __CHECK_NULL_ARGUMENT(parser);
  __CHECK_NULL_ARGUMENT(code);

  LumTokenizer* tokenizer;
  __FAILURE_HANDLE(lum_tokenizer_create(&tokenizer));

  __FAILURE_HANDLE(lum_tokenizer_execute(tokenizer, code));

  __FAILURE_HANDLE(lum_tokenizer_print(tokenizer));

  __FAILURE_HANDLE(lum_tokenizer_destroy(&tokenizer));

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_parser_destroy(LumParser** parser) {
  __CHECK_NULL_ARGUMENT(parser);
  __CHECK_NULL_ARGUMENT(*parser);

  __FAILURE_HANDLE(host_free(parser));

  return LUMINARY_SUCCESS;
}
