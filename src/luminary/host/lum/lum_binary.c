#include "lum_binary.h"

#include <stdio.h>
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

LuminaryResult lum_binary_print(LumBinary* binary) {
  __CHECK_NULL_ARGUMENT(binary);

  FILE* file = fopen("DebugLUMV5BinaryAssembly.s", "wb");

  if (file == (FILE*) 0) {
    __RETURN_ERROR(LUMINARY_ERROR_C_STD, "Failed to open file \"DebugLUMV5BinaryAssembly.s\"");
  }

  fprintf(file, "======= .data =======\n");

  // TODO

  fprintf(file, "======= .text =======\n");

  uint32_t num_instructions;
  __FAILURE_HANDLE(array_get_num_elements(binary->instructions, &num_instructions));

  uint32_t offset = 0;

  for (uint32_t instruction_id = 0; instruction_id < num_instructions; instruction_id++) {
    const LumInstruction* instruction = binary->instructions + instruction_id;

    fprintf(file, "%08X ", offset);

    uint8_t bytes[9];
    uint8_t size;
    __FAILURE_HANDLE(lum_instruction_get_bytes(instruction, bytes, &size));

    uint8_t byte_id = 0;
    for (; byte_id < size; byte_id++) {
      fprintf(file, "%02X", bytes[byte_id]);
    }

    for (; byte_id < 8; byte_id++) {
      fprintf(file, "  ");
    }

    char mnemonic[256];
    __FAILURE_HANDLE(lum_instruction_get_mnemonic(instruction, mnemonic));

    fprintf(file, " %-20s ", mnemonic);

    char arg_string[256];
    __FAILURE_HANDLE(lum_instruction_get_args(instruction, arg_string));
    fprintf(file, "%s\n", arg_string);

    offset += size;
  }

  fclose(file);

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_binary_destroy(LumBinary** binary) {
  __CHECK_NULL_ARGUMENT(binary);
  __CHECK_NULL_ARGUMENT(*binary);

  if ((*binary)->instructions) {
    __FAILURE_HANDLE(array_destroy(&(*binary)->instructions));
  }

  __FAILURE_HANDLE(host_free(binary));

  return LUMINARY_SUCCESS;
}
