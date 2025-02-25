#include "lum_binary.h"

#include <string.h>

#include "internal_error.h"

LuminaryResult lum_binary_create(LumBinary** binary) {
  __CHECK_NULL_ARGUMENT(binary);

  __FAILURE_HANDLE(host_malloc(binary, sizeof(LumBinary)));
  memset(*binary, 0, sizeof(LumBinary));

  __FAILURE_HANDLE(array_create(&(*binary)->instructions, sizeof(LumInstruction), 64));

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_binary_compute_stack_frame_size(LumBinary* binary);
LuminaryResult lum_binary_print(LumBinary* binary);

LuminaryResult lum_binary_destroy(LumBinary** binary) {
  __CHECK_NULL_ARGUMENT(binary);
  __CHECK_NULL_ARGUMENT(*binary);

  if ((*binary)->instructions) {
    __FAILURE_HANDLE(array_destroy(&(*binary)->instructions));
  }

  __FAILURE_HANDLE(host_free(binary));

  return LUMINARY_SUCCESS;
}
