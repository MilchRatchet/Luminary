#ifndef LUMINARY_LUM_COMPILER_H
#define LUMINARY_LUM_COMPILER_H

#include "lum_binary.h"
#include "lum_tokenizer.h"
#include "utils.h"

struct LumCompiler {
  bool build_asynchronous_part;
  uint32_t log_level;  // 0 = Only errors, 1 = Errors and Warnings, 2 = Everything
} typedef LumCompiler;

LuminaryResult lum_compiler_create(LumCompiler** compiler);
LuminaryResult lum_compiler_compile(LumCompiler* compiler, ARRAY const LumToken* tokens, LumBinary* binary);
LuminaryResult lum_compiler_destroy(LumCompiler** compiler);

#endif /* LUMINARY_LUM_COMPILER_H */
