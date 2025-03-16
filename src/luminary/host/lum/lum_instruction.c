#include "lum_instruction.h"

#include <stdio.h>
#include <string.h>

#include "internal_error.h"
#include "lum_builtins.h"
#include "lum_function_tables.h"

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

#define LUM_INSTRUCTION_MOV_ARG_MASK 0xFF
#define LUM_INSTRUCTION_MOV_TYPE_SHIFT 16
#define LUM_INSTRUCTION_MOV_DST_REG_SHIFT 8
#define LUM_INSTRUCTION_MOV_SRC_REG_SHIFT 0

LuminaryResult lum_instruction_encode_mov(LumInstruction* instruction, LumBuiltinType type, uint8_t dst_reg, uint8_t src_reg) {
  __CHECK_NULL_ARGUMENT(instruction);

  memset(instruction, 0, sizeof(LumInstruction));

  *instruction |= ((uint64_t) LUM_INSTRUCTION_TYPE_MOV) << LUM_INSTRUCTION_TYPE_SHIFT;
  *instruction |= ((uint64_t) type) << LUM_INSTRUCTION_MOV_TYPE_SHIFT;
  *instruction |= ((uint64_t) dst_reg) << LUM_INSTRUCTION_MOV_DST_REG_SHIFT;
  *instruction |= ((uint64_t) src_reg) << LUM_INSTRUCTION_MOV_SRC_REG_SHIFT;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_instruction_mov_get_mnemonic_suffix(const LumInstruction* instruction, char* mnemonic) {
  __CHECK_NULL_ARGUMENT(instruction);
  __CHECK_NULL_ARGUMENT(mnemonic);

  const LumBuiltinType type = ((*instruction) >> LUM_INSTRUCTION_MOV_TYPE_SHIFT) & LUM_INSTRUCTION_MOV_ARG_MASK;

  const size_t type_size = lum_builtin_types_sizes[type];

  if (type_size > 0) {
    const char* type_mnemonic = lum_builtin_types_mnemonic[type];
    sprintf(mnemonic, ".%s", type_mnemonic);
  }

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_instruction_mov_get_args(const LumInstruction* instruction, char* args) {
  __CHECK_NULL_ARGUMENT(instruction);
  __CHECK_NULL_ARGUMENT(args);

  const uint8_t dst_reg = ((*instruction) >> LUM_INSTRUCTION_MOV_DST_REG_SHIFT) & LUM_INSTRUCTION_MOV_ARG_MASK;
  const uint8_t src_reg = ((*instruction) >> LUM_INSTRUCTION_MOV_SRC_REG_SHIFT) & LUM_INSTRUCTION_MOV_ARG_MASK;

  sprintf(args, "%%r%u, %%r%u", dst_reg, src_reg);

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// CALL
////////////////////////////////////////////////////////////////////

#define LUM_INSTRUCTION_CALL_ARG_MASK 0xFF
#define LUM_INSTRUCTION_CALL_TYPE_SHIFT 48
#define LUM_INSTRUCTION_CALL_FUNC_ID_SHIFT 40
#define LUM_INSTRUCTION_CALL_DST_REG_SHIFT 32
#define LUM_INSTRUCTION_CALL_SRC_REG_3_SHIFT 24
#define LUM_INSTRUCTION_CALL_SRC_REG_2_SHIFT 16
#define LUM_INSTRUCTION_CALL_SRC_REG_1_SHIFT 8
#define LUM_INSTRUCTION_CALL_SRC_REG_0_SHIFT 0

LuminaryResult lum_instruction_encode_call(
  LumInstruction* instruction, LumBuiltinType type, uint8_t func_id, uint8_t dst_reg, uint8_t src_regs[4]) {
  __CHECK_NULL_ARGUMENT(instruction);

  memset(instruction, 0, sizeof(LumInstruction));

  *instruction |= ((uint64_t) LUM_INSTRUCTION_TYPE_CALL) << LUM_INSTRUCTION_TYPE_SHIFT;
  *instruction |= ((uint64_t) type) << LUM_INSTRUCTION_CALL_TYPE_SHIFT;
  *instruction |= ((uint64_t) func_id) << LUM_INSTRUCTION_CALL_FUNC_ID_SHIFT;
  *instruction |= ((uint64_t) dst_reg) << LUM_INSTRUCTION_CALL_DST_REG_SHIFT;
  *instruction |= ((uint64_t) src_regs[3]) << LUM_INSTRUCTION_CALL_SRC_REG_3_SHIFT;
  *instruction |= ((uint64_t) src_regs[2]) << LUM_INSTRUCTION_CALL_SRC_REG_2_SHIFT;
  *instruction |= ((uint64_t) src_regs[1]) << LUM_INSTRUCTION_CALL_SRC_REG_1_SHIFT;
  *instruction |= ((uint64_t) src_regs[0]) << LUM_INSTRUCTION_CALL_SRC_REG_0_SHIFT;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_instruction_call_get_args(const LumInstruction* instruction, char* args) {
  __CHECK_NULL_ARGUMENT(instruction);
  __CHECK_NULL_ARGUMENT(args);

  const LumBuiltinType type = ((*instruction) >> LUM_INSTRUCTION_CALL_TYPE_SHIFT) & LUM_INSTRUCTION_CALL_ARG_MASK;
  const uint8_t func_id     = ((*instruction) >> LUM_INSTRUCTION_CALL_FUNC_ID_SHIFT) & LUM_INSTRUCTION_CALL_ARG_MASK;

  const char* class_name          = lum_builtin_types_strings[type];
  const LumFunctionEntry function = lum_function_tables[type][func_id];

  int offset = 0;

  offset += sprintf(args + offset, "%s::%s", class_name, function.name);

  if (function.signature.dst != LUM_BUILTIN_TYPE_VOID) {
    const uint8_t dst_reg = ((*instruction) >> LUM_INSTRUCTION_CALL_DST_REG_SHIFT) & LUM_INSTRUCTION_CALL_ARG_MASK;
    offset += sprintf(args + offset, ", %%r%u", dst_reg);
  }

  if (function.signature.src_0 != LUM_BUILTIN_TYPE_VOID) {
    const uint8_t src_reg0 = ((*instruction) >> LUM_INSTRUCTION_CALL_SRC_REG_0_SHIFT) & LUM_INSTRUCTION_CALL_ARG_MASK;
    offset += sprintf(args + offset, ", %%r%u", src_reg0);
  }

  if (function.signature.src_1 != LUM_BUILTIN_TYPE_VOID) {
    const uint8_t src_reg1 = ((*instruction) >> LUM_INSTRUCTION_CALL_SRC_REG_1_SHIFT) & LUM_INSTRUCTION_CALL_ARG_MASK;
    offset += sprintf(args + offset, ", %%r%u", src_reg1);
  }

  if (function.signature.src_2 != LUM_BUILTIN_TYPE_VOID) {
    const uint8_t src_reg2 = ((*instruction) >> LUM_INSTRUCTION_CALL_SRC_REG_2_SHIFT) & LUM_INSTRUCTION_CALL_ARG_MASK;
    offset += sprintf(args + offset, ", %%r%u", src_reg2);
  }

  if (function.signature.src_3 != LUM_BUILTIN_TYPE_VOID) {
    const uint8_t src_reg3 = ((*instruction) >> LUM_INSTRUCTION_CALL_SRC_REG_3_SHIFT) & LUM_INSTRUCTION_CALL_ARG_MASK;
    offset += sprintf(args + offset, ", %%r%u", src_reg3);
  }

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
    case LUM_INSTRUCTION_TYPE_MOV:
      __FAILURE_HANDLE(_lum_instruction_mov_get_mnemonic_suffix(instruction, mnemonic + string_length));
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
    case LUM_INSTRUCTION_TYPE_MOV:
      __FAILURE_HANDLE(_lum_instruction_mov_get_args(instruction, args));
      break;
    case LUM_INSTRUCTION_TYPE_CALL:
      __FAILURE_HANDLE(_lum_instruction_call_get_args(instruction, args));
      break;
    default:
      args[0] = '\0';
      break;
  }

  return LUMINARY_SUCCESS;
}