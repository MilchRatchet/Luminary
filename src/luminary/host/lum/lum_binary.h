#ifndef LUMINARY_LUM_BINARY_H
#define LUMINARY_LUM_BINARY_H

#include "lum_instruction.h"
#include "utils.h"

struct LumBinary {
  size_t stack_size;
  ARRAY LumInstruction* instructions;
  void* constant_memory;
  size_t constant_memory_size;
} typedef LumBinary;

LuminaryResult lum_binary_create(LumBinary** binary);
LuminaryResult lum_binary_print(LumBinary* binary);
LuminaryResult lum_binary_destroy(LumBinary** binary);

#endif /* LUMINARY_LUM_BINARY_H */
