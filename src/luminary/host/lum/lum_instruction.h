#ifndef LUMINARY_LUM_INSTRUCTION_H
#define LUMINARY_LUM_INSTRUCTION_H

#include "lum_builtins.h"
#include "utils.h"

enum LumInstructionType {
  LUM_INSTRUCTION_TYPE_NOP,
  LUM_INSTRUCTION_TYPE_REGMAP,
  LUM_INSTRUCTION_TYPE_MOV,
  LUM_INSTRUCTION_TYPE_CALL,
  LUM_INSTRUCTION_TYPE_RET,
  LUM_INSTRUCTION_TYPE_CVT,
  LUM_INSTRUCTION_TYPE_COUNT
} typedef LumInstructionType;

typedef uint64_t LumInstruction;

LuminaryResult lum_instruction_get_type(const LumInstruction* instruction, LumInstructionType* type);
LuminaryResult lum_instruction_get_mnemonic(const LumInstruction* instruction, char* mnemonic);
LuminaryResult lum_instruction_get_bytes(const LumInstruction* instruction, uint8_t* data, uint8_t* size);
LuminaryResult lum_instruction_get_args(const LumInstruction* instruction, char* args);

LuminaryResult lum_instruction_encode_nop(LumInstruction* instruction);
LuminaryResult lum_instruction_encode_regmap(LumInstruction* instruction, uint8_t reg, bool is_data_section, uint32_t offset);
LuminaryResult lum_instruction_encode_mov(LumInstruction* instruction, LumBuiltinType type, uint8_t dst_reg, uint8_t src_reg);
LuminaryResult lum_instruction_encode_call(
  LumInstruction* instruction, LumBuiltinType type, uint8_t func_id, uint8_t dst_reg, uint8_t src_regs[4]);
LuminaryResult lum_instruction_encode_ret(LumInstruction* instruction);
LuminaryResult lum_instruction_encode_cvt(LumInstruction* instruction);

#endif /* LUMINARY_LUM_INSTRUCTION_H */