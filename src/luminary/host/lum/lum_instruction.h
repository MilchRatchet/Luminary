#ifndef LUMINARY_LUM_INSTRUCTION_H
#define LUMINARY_LUM_INSTRUCTION_H

#include "lum_builtins.h"
#include "utils.h"

enum LumInstructionType {
  LUM_INSTRUCTION_TYPE_NOP,
  LUM_INSTRUCTION_TYPE_MOV,
  LUM_INSTRUCTION_TYPE_LDG,
  LUM_INSTRUCTION_TYPE_STG,
  LUM_INSTRUCTION_TYPE_COUNT
} typedef LumInstructionType;

#define LUM_MEMORY_CONSTANT_MEMORY_SPACE_BIT (0x80000000)
#define LUM_MEMORY_OFFSET_MASK (0x7FFFFFFF)

struct LumMemoryAllocation {
  uint32_t offset;
  uint32_t size;
} typedef LumMemoryAllocation;

struct LumInstruction {
  uint64_t meta;
  LumMemoryAllocation src;
  LumMemoryAllocation dst;
} typedef LumInstruction;

LuminaryResult lum_instruction_get_type(const LumInstruction* instruction, LumInstructionType* type);
LuminaryResult lum_instruction_get_mnemonic(const LumInstruction* instruction, char* mnemonic);
LuminaryResult lum_instruction_get_bytes(const LumInstruction* instruction, uint64_t* data);
LuminaryResult lum_instruction_get_args(const LumInstruction* instruction, char* args);

LuminaryResult lum_instruction_encode_nop(LumInstruction* instruction);
LuminaryResult lum_instruction_encode_mov(
  LumInstruction* instruction, LumBuiltinType type, LumMemoryAllocation dst, LumMemoryAllocation src);
LuminaryResult lum_instruction_encode_ldg(
  LumInstruction* instruction, LumBuiltinType type, LumMemoryAllocation dst, LumMemoryAllocation src);
LuminaryResult lum_instruction_encode_stg(LumInstruction* instruction, LumBuiltinType type, LumMemoryAllocation src);

LuminaryResult lum_instruction_decode_mov(const LumInstruction* instruction, LumBuiltinType* type);
LuminaryResult lum_instruction_decode_ldg(const LumInstruction* instruction, LumBuiltinType* type);
LuminaryResult lum_instruction_decode_stg(const LumInstruction* instruction, LumBuiltinType* type);

#endif /* LUMINARY_LUM_INSTRUCTION_H */
