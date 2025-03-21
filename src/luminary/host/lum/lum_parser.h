#ifndef LUMINARY_LUM_PARSER_H
#define LUMINARY_LUM_PARSER_H

#include "lum_binary.h"
#include "utils.h"

struct LumParser {
  uint32_t dummy;
} typedef LumParser;

LuminaryResult lum_parser_create(LumParser** parser);
LuminaryResult lum_parser_execute(LumParser* parser, const char* code, LumBinary* binary);
LuminaryResult lum_parser_destroy(LumParser** parser);

#endif /* LUMINARY_LUM_PARSER_H */
