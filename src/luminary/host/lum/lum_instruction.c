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

  instruction->meta |= type << LUM_INSTRUCTION_TYPE_SHIFT;

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// MOV
////////////////////////////////////////////////////////////////////

#define LUM_INSTRUCTION_MOV_ARG_MASK 0xFF
#define LUM_INSTRUCTION_MOV_TYPE_SHIFT 16

LuminaryResult lum_instruction_encode_mov(
  LumInstruction* instruction, LumBuiltinType type, LumMemoryAllocation dst, LumMemoryAllocation src) {
  __CHECK_NULL_ARGUMENT(instruction);

  memset(instruction, 0, sizeof(LumInstruction));

  instruction->meta |= ((uint64_t) LUM_INSTRUCTION_TYPE_MOV) << LUM_INSTRUCTION_TYPE_SHIFT;
  instruction->meta |= ((uint64_t) type) << LUM_INSTRUCTION_MOV_TYPE_SHIFT;

  instruction->dst = dst;
  instruction->src = src;

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_instruction_mov_get_mnemonic_suffix(const LumInstruction* instruction, char* mnemonic) {
  __CHECK_NULL_ARGUMENT(instruction);
  __CHECK_NULL_ARGUMENT(mnemonic);

  const LumBuiltinType type = (instruction->meta >> LUM_INSTRUCTION_MOV_TYPE_SHIFT) & LUM_INSTRUCTION_MOV_ARG_MASK;

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

  int offset = 0;
  offset += sprintf(args + offset, "%%w[%u,%u]", instruction->dst.offset, instruction->dst.offset + instruction->dst.size);
  offset += sprintf(args + offset, ", ");
  offset += sprintf(args + offset, "%%r[%u,%u]", instruction->src.offset, instruction->src.offset + instruction->src.size);

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// LDG
////////////////////////////////////////////////////////////////////

#define LUM_INSTRUCTION_LDG_ARG_MASK 0xFF
#define LUM_INSTRUCTION_LDG_TYPE_SHIFT 48

LuminaryResult lum_instruction_encode_ldg(
  LumInstruction* instruction, LumBuiltinType type, LumMemoryAllocation dst, LumMemoryAllocation src) {
  __CHECK_NULL_ARGUMENT(instruction);

  memset(instruction, 0, sizeof(LumInstruction));

  instruction->meta |= ((uint64_t) LUM_INSTRUCTION_TYPE_LDG) << LUM_INSTRUCTION_TYPE_SHIFT;
  instruction->meta |= ((uint64_t) type) << LUM_INSTRUCTION_LDG_TYPE_SHIFT;

  instruction->dst = dst;
  instruction->src = src;

  __DEBUG_ASSERT(dst.size == lum_builtin_types_sizes[type]);

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_instruction_ldg_get_args(const LumInstruction* instruction, char* args) {
  __CHECK_NULL_ARGUMENT(instruction);
  __CHECK_NULL_ARGUMENT(args);

  const LumBuiltinType type = (instruction->meta >> LUM_INSTRUCTION_LDG_TYPE_SHIFT) & LUM_INSTRUCTION_LDG_ARG_MASK;

  const char* class_name = lum_builtin_types_strings[type];

  int offset = 0;

  offset += sprintf(args + offset, "%%w[%u,%u]", instruction->dst.offset, instruction->dst.offset + instruction->dst.size);
  offset += sprintf(args + offset, ", ");
  offset += sprintf(args + offset, "%%r[%u,%u]", instruction->src.offset, instruction->src.offset + instruction->src.size);

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_instruction_ldg_get_mnemonic_suffix(const LumInstruction* instruction, char* mnemonic) {
  __CHECK_NULL_ARGUMENT(instruction);
  __CHECK_NULL_ARGUMENT(mnemonic);

  const LumBuiltinType type = (instruction->meta >> LUM_INSTRUCTION_LDG_TYPE_SHIFT) & LUM_INSTRUCTION_LDG_ARG_MASK;

  __DEBUG_ASSERT(type != LUM_BUILTIN_TYPE_VOID);

  const char* type_mnemonic = lum_builtin_types_mnemonic[type];
  sprintf(mnemonic, ".%s", type_mnemonic);

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// STG
////////////////////////////////////////////////////////////////////

#define LUM_INSTRUCTION_STG_ARG_MASK 0xFF
#define LUM_INSTRUCTION_STG_TYPE_SHIFT 48

LuminaryResult lum_instruction_encode_stg(LumInstruction* instruction, LumBuiltinType type, LumMemoryAllocation src) {
  __CHECK_NULL_ARGUMENT(instruction);

  memset(instruction, 0, sizeof(LumInstruction));

  instruction->meta |= ((uint64_t) LUM_INSTRUCTION_TYPE_STG) << LUM_INSTRUCTION_TYPE_SHIFT;
  instruction->meta |= ((uint64_t) type) << LUM_INSTRUCTION_STG_TYPE_SHIFT;

  instruction->src = src;

  __DEBUG_ASSERT(src.size == lum_builtin_types_sizes[type]);

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_instruction_stg_get_args(const LumInstruction* instruction, char* args) {
  __CHECK_NULL_ARGUMENT(instruction);
  __CHECK_NULL_ARGUMENT(args);

  const LumBuiltinType type = (instruction->meta >> LUM_INSTRUCTION_STG_TYPE_SHIFT) & LUM_INSTRUCTION_STG_ARG_MASK;

  const char* class_name = lum_builtin_types_strings[type];

  int offset = 0;

  offset += sprintf(args + offset, "%%r[%u,%u]", instruction->src.offset, instruction->src.offset + instruction->src.size);

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_instruction_stg_get_mnemonic_suffix(const LumInstruction* instruction, char* mnemonic) {
  __CHECK_NULL_ARGUMENT(instruction);
  __CHECK_NULL_ARGUMENT(mnemonic);

  const LumBuiltinType type = (instruction->meta >> LUM_INSTRUCTION_STG_TYPE_SHIFT) & LUM_INSTRUCTION_STG_ARG_MASK;

  __DEBUG_ASSERT(type != LUM_BUILTIN_TYPE_VOID);

  const char* type_mnemonic = lum_builtin_types_mnemonic[type];
  sprintf(mnemonic, ".%s", type_mnemonic);

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// RET
////////////////////////////////////////////////////////////////////

LuminaryResult lum_instruction_encode_ret(LumInstruction* instruction) {
  __CHECK_NULL_ARGUMENT(instruction);

  memset(instruction, 0, sizeof(LumInstruction));

  uint64_t type = LUM_INSTRUCTION_TYPE_RET;

  instruction->meta |= type << LUM_INSTRUCTION_TYPE_SHIFT;

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// CVT
////////////////////////////////////////////////////////////////////

LuminaryResult lum_instruction_encode_cvt(LumInstruction* instruction) {
  __CHECK_NULL_ARGUMENT(instruction);

  memset(instruction, 0, sizeof(LumInstruction));

  uint64_t type = LUM_INSTRUCTION_TYPE_CVT;

  instruction->meta |= type << LUM_INSTRUCTION_TYPE_SHIFT;

  return LUMINARY_SUCCESS;
}

////////////////////////////////////////////////////////////////////
// COMMON
////////////////////////////////////////////////////////////////////

static const char* lum_instruction_type_mnemonics[LUM_INSTRUCTION_TYPE_COUNT] = {
  [LUM_INSTRUCTION_TYPE_NOP] = "nop", [LUM_INSTRUCTION_TYPE_MOV] = "mov", [LUM_INSTRUCTION_TYPE_LDG] = "ldg",
  [LUM_INSTRUCTION_TYPE_STG] = "stg", [LUM_INSTRUCTION_TYPE_RET] = "ret", [LUM_INSTRUCTION_TYPE_CVT] = "cvt"};

LuminaryResult lum_instruction_get_type(const LumInstruction* instruction, LumInstructionType* type) {
  __CHECK_NULL_ARGUMENT(instruction);
  __CHECK_NULL_ARGUMENT(type);

  *type = (LumInstructionType) (instruction->meta >> LUM_INSTRUCTION_TYPE_SHIFT);

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_instruction_get_mnemonic(const LumInstruction* instruction, char* mnemonic) {
  __CHECK_NULL_ARGUMENT(instruction);
  __CHECK_NULL_ARGUMENT(mnemonic);

  LumInstructionType type = (LumInstructionType) (instruction->meta >> LUM_INSTRUCTION_TYPE_SHIFT);

  const char* string         = lum_instruction_type_mnemonics[type];
  const size_t string_length = strlen(string);

  memcpy(mnemonic, string, string_length);

  mnemonic[string_length] = '\0';

  switch (type) {
    case LUM_INSTRUCTION_TYPE_MOV:
      __FAILURE_HANDLE(_lum_instruction_mov_get_mnemonic_suffix(instruction, mnemonic + string_length));
      break;
    case LUM_INSTRUCTION_TYPE_LDG:
      __FAILURE_HANDLE(_lum_instruction_ldg_get_mnemonic_suffix(instruction, mnemonic + string_length));
      break;
    case LUM_INSTRUCTION_TYPE_STG:
      __FAILURE_HANDLE(_lum_instruction_stg_get_mnemonic_suffix(instruction, mnemonic + string_length));
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

  LumInstructionType type = (LumInstructionType) (instruction->meta >> LUM_INSTRUCTION_TYPE_SHIFT);

  switch (type) {
    case LUM_INSTRUCTION_TYPE_MOV:
      __FAILURE_HANDLE(_lum_instruction_mov_get_args(instruction, args));
      break;
    case LUM_INSTRUCTION_TYPE_LDG:
      __FAILURE_HANDLE(_lum_instruction_ldg_get_args(instruction, args));
      break;
    case LUM_INSTRUCTION_TYPE_STG:
      __FAILURE_HANDLE(_lum_instruction_stg_get_args(instruction, args));
      break;
    default:
      args[0] = '\0';
      break;
  }

  return LUMINARY_SUCCESS;
}
