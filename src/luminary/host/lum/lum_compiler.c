#include "lum_compiler.h"

#include <string.h>

#include "internal_error.h"

LuminaryResult lum_compiler_create(LumCompiler** compiler) {
  __CHECK_NULL_ARGUMENT(compiler);

  __FAILURE_HANDLE(host_malloc(compiler, sizeof(LumCompiler)));
  memset(*compiler, 0, sizeof(LumCompiler));

  return LUMINARY_SUCCESS;
}

#define LUM_VARIABLE_REGISTER_NOT_ASSIGNED 0xFFFFFFFF

struct LumVariable {
  size_t offset;
  uint32_t current_register_map;
} typedef LumVariable;

LuminaryResult lum_compiler_compile(LumCompiler* compiler, ARRAY const LumToken* tokens, LumBinary* binary) {
  __CHECK_NULL_ARGUMENT(compiler);
  __CHECK_NULL_ARGUMENT(tokens);
  __CHECK_NULL_ARGUMENT(binary);

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_compiler_destroy(LumCompiler** compiler) {
  __CHECK_NULL_ARGUMENT(compiler);
  __CHECK_NULL_ARGUMENT(*compiler);

  __FAILURE_HANDLE(host_free(compiler));

  return LUMINARY_SUCCESS;
}
