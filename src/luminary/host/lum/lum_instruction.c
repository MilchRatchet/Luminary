#include "lum_instruction.h"

#include <stdio.h>
#include <string.h>

#include "internal_error.h"

#define LUM_INSTRUCTION_TYPE_SHIFT 56

////////////////////////////////////////////////////////////////////
// NOP
////////////////////////////////////////////////////////////////////

LuminaryResult lum_instruction_encode_nop(LumInstruction* instruction) {
  __CHECK_NULL_ARGUMENT(instruction);

  memset(instruction, 0, sizeof(LumInstruction));

  uint64_t type = LUM_INSTRUCTION_TYPE_NOP;

  *instruction |= type << LUM_INSTRUCTION_TYPE_SHIFT;

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// REGMAP
////////////////////////////////////////////////////////////////////

#define LUM_INSTRUCTION_REGMAP_DATA_SECTION_SHIFT 40
#define LUM_INSTRUCTION_REGMAP_DATA_SECTION_MASK 0x1
#define LUM_INSTRUCTION_REGMAP_REGISTER_SHIFT 32
#define LUM_INSTRUCTION_REGMAP_REGISTER_MASK 0xFF
#define LUM_INSTRUCTION_REGMAP_OFFSET_MASK 0xFFFFFFFF

LuminaryResult lum_instruction_encode_regmap(LumInstruction* instruction, uint8_t reg, bool is_data_section, uint32_t offset) {
  __CHECK_NULL_ARGUMENT(instruction);

  memset(instruction, 0, sizeof(LumInstruction));

  uint64_t type = LUM_INSTRUCTION_TYPE_REGMAP;

  *instruction |= type << LUM_INSTRUCTION_TYPE_SHIFT;
  *instruction |= (is_data_section ? 1ull : 0ull) << LUM_INSTRUCTION_REGMAP_DATA_SECTION_SHIFT;
  *instruction |= ((uint64_t) reg) << LUM_INSTRUCTION_REGMAP_REGISTER_SHIFT;
  *instruction |= offset;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_instruction_regmap_get_mnemonic_suffix(const LumInstruction* instruction, char* mnemonic) {
  __CHECK_NULL_ARGUMENT(instruction);
  __CHECK_NULL_ARGUMENT(mnemonic);

  const bool is_data_section =
    (((*instruction) >> LUM_INSTRUCTION_REGMAP_DATA_SECTION_SHIFT) & LUM_INSTRUCTION_REGMAP_DATA_SECTION_MASK) != 0;

  if (is_data_section) {
    sprintf(mnemonic, ".data");
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_instruction_regmap_get_args(const LumInstruction* instruction, char* args) {
  __CHECK_NULL_ARGUMENT(instruction);
  __CHECK_NULL_ARGUMENT(args);

  const uint8_t reg = ((*instruction) >> LUM_INSTRUCTION_REGMAP_REGISTER_SHIFT) & LUM_INSTRUCTION_REGMAP_REGISTER_MASK;

  const uint32_t offset = *instruction & LUM_INSTRUCTION_REGMAP_OFFSET_MASK;

  sprintf(args, "%%r%u, 0x%08X", reg, offset);

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// MOV
////////////////////////////////////////////////////////////////////

LuminaryResult lum_instruction_encode_mov(LumInstruction* instruction) {
  __CHECK_NULL_ARGUMENT(instruction);

  memset(instruction, 0, sizeof(LumInstruction));

  uint64_t type = LUM_INSTRUCTION_TYPE_MOV;

  *instruction |= type << LUM_INSTRUCTION_TYPE_SHIFT;

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// CALL
////////////////////////////////////////////////////////////////////

LuminaryResult lum_instruction_encode_call(LumInstruction* instruction) {
  __CHECK_NULL_ARGUMENT(instruction);

  memset(instruction, 0, sizeof(LumInstruction));

  uint64_t type = LUM_INSTRUCTION_TYPE_CALL;

  *instruction |= type << LUM_INSTRUCTION_TYPE_SHIFT;

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// RET
////////////////////////////////////////////////////////////////////

LuminaryResult lum_instruction_encode_ret(LumInstruction* instruction) {
  __CHECK_NULL_ARGUMENT(instruction);

  memset(instruction, 0, sizeof(LumInstruction));

  uint64_t type = LUM_INSTRUCTION_TYPE_RET;

  *instruction |= type << LUM_INSTRUCTION_TYPE_SHIFT;

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// CVT
////////////////////////////////////////////////////////////////////

LuminaryResult lum_instruction_encode_cvt(LumInstruction* instruction) {
  __CHECK_NULL_ARGUMENT(instruction);

  memset(instruction, 0, sizeof(LumInstruction));

  uint64_t type = LUM_INSTRUCTION_TYPE_CVT;

  *instruction |= type << LUM_INSTRUCTION_TYPE_SHIFT;

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// COMMON
////////////////////////////////////////////////////////////////////

static const char* lum_instruction_type_mnemonics[LUM_INSTRUCTION_TYPE_COUNT] = {
  [LUM_INSTRUCTION_TYPE_NOP] = "nop",   [LUM_INSTRUCTION_TYPE_REGMAP] = "regmap", [LUM_INSTRUCTION_TYPE_MOV] = "mov",
  [LUM_INSTRUCTION_TYPE_CALL] = "call", [LUM_INSTRUCTION_TYPE_RET] = "ret",       [LUM_INSTRUCTION_TYPE_CVT] = "cvt"};

LuminaryResult lum_instruction_get_type(const LumInstruction* instruction, LumInstructionType* type) {
  __CHECK_NULL_ARGUMENT(instruction);
  __CHECK_NULL_ARGUMENT(type);

  *type = (LumInstructionType) (*instruction >> LUM_INSTRUCTION_TYPE_SHIFT);

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_instruction_get_mnemonic(const LumInstruction* instruction, char* mnemonic) {
  __CHECK_NULL_ARGUMENT(instruction);
  __CHECK_NULL_ARGUMENT(mnemonic);

  LumInstructionType type = (LumInstructionType) (*instruction >> LUM_INSTRUCTION_TYPE_SHIFT);

  const char* string         = lum_instruction_type_mnemonics[type];
  const size_t string_length = strlen(string);

  memcpy(mnemonic, string, string_length);

  mnemonic[string_length] = '\0';

  switch (type) {
    case LUM_INSTRUCTION_TYPE_REGMAP:
      __FAILURE_HANDLE(_lum_instruction_regmap_get_mnemonic_suffix(instruction, mnemonic + string_length));
      break;
    default:
      break;
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_instruction_get_bytes(const LumInstruction* instruction, uint8_t* data, uint8_t* size) {
  __CHECK_NULL_ARGUMENT(instruction);
  __CHECK_NULL_ARGUMENT(data);
  __CHECK_NULL_ARGUMENT(size);

  memcpy(data, instruction, sizeof(LumInstruction));

  *size = (uint8_t) sizeof(LumInstruction);

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_instruction_get_args(const LumInstruction* instruction, char* args) {
  __CHECK_NULL_ARGUMENT(instruction);
  __CHECK_NULL_ARGUMENT(args);

  LumInstructionType type = (LumInstructionType) (*instruction >> LUM_INSTRUCTION_TYPE_SHIFT);

  switch (type) {
    case LUM_INSTRUCTION_TYPE_REGMAP:
      __FAILURE_HANDLE(_lum_instruction_regmap_get_args(instruction, args));
      break;
    default:
      args[0] = '\0';
      break;
  }

  return LUMINARY_SUCCESS;
}