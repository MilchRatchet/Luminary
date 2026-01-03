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

LuminaryResult lum_binary_print(LumBinary* binary) {
  __CHECK_NULL_ARGUMENT(binary);

  FILE* file = fopen("DebugLUMV5BinaryAssembly.s", "wb");

  if (file == (FILE*) 0) {
    __RETURN_ERROR(LUMINARY_ERROR_C_STD, "Failed to open file \"DebugLUMV5BinaryAssembly.s\"");
  }

  fprintf(file, "======= .data =======\n");

  const uint32_t constant_bytes_per_line = 16;
  const uint8_t* constant_memory_src     = (const uint8_t*) binary->constant_memory;

  for (uint32_t byte_offset = 0; byte_offset < binary->constant_memory_size; byte_offset += constant_bytes_per_line) {
    fprintf(file, "%08X ", byte_offset);

    uint8_t byte_id = 0;
    for (; byte_id < constant_bytes_per_line && byte_offset + byte_id < binary->constant_memory_size; byte_id++) {
      fprintf(file, "%02X", constant_memory_src[byte_offset + byte_id]);
    }

    for (; byte_id < constant_bytes_per_line; byte_id++) {
      fprintf(file, "  ");
    }

    fprintf(file, "\n");
  }

  fprintf(file, "======= .text =======\n");

  uint32_t num_instructions;
  __FAILURE_HANDLE(array_get_num_elements(binary->instructions, &num_instructions));

  uint32_t offset = 0;

  for (uint32_t instruction_id = 0; instruction_id < num_instructions; instruction_id++) {
    const LumInstruction* instruction = binary->instructions + instruction_id;

    fprintf(file, "%08X ", offset);

    uint64_t bytes;
    __FAILURE_HANDLE(lum_instruction_get_bytes(instruction, &bytes));

    uint8_t byte_id = 0;
    for (; byte_id < sizeof(uint64_t); byte_id++) {
      fprintf(file, "%02llX", (bytes >> (8 * byte_id)) & 0xFF);
    }

    char mnemonic[256];
    __FAILURE_HANDLE(lum_instruction_get_mnemonic(instruction, mnemonic));

    fprintf(file, " %-20s ", mnemonic);

    char arg_string[256];
    __FAILURE_HANDLE(lum_instruction_get_args(instruction, arg_string));
    fprintf(file, "%s\n", arg_string);

    offset += sizeof(LumInstruction);
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

  if ((*binary)->constant_memory) {
    __FAILURE_HANDLE(host_free(&(*binary)->constant_memory));
  }

  __FAILURE_HANDLE(host_free(binary));

  return LUMINARY_SUCCESS;
}
