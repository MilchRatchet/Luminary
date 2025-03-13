#ifndef LUMINARY_LUM_BINARY_H
#define LUMINARY_LUM_BINARY_H

#include "lum_instruction.h"
#include "utils.h"

enum LumBinaryEntryPoint {
  LUM_BINARY_ENTRY_POINT_SYNCHRONOUS,
  LUM_BINARY_ENTRY_POINT_ASYNCHRONOUS,
  LUM_BINARY_ENTRY_POINT_COUNT
} typedef LumBinaryEntryPoint;

#define LUM_BINARY_ENTRY_POINT_INVALID 0xFFFFFFFF

struct LumBinary {
  size_t stack_frame_size;
  uint32_t entry_points[LUM_BINARY_ENTRY_POINT_COUNT];
  ARRAY LumInstruction* instructions;
} typedef LumBinary;

LuminaryResult lum_binary_create(LumBinary** binary);
LuminaryResult lum_binary_compute_stack_frame_size(LumBinary* binary);
LuminaryResult lum_binary_print(LumBinary* binary);
LuminaryResult lum_binary_destroy(LumBinary** binary);

#endif /* LUMINARY_LUM_BINARY_H */
