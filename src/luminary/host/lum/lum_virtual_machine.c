#include "lum_virtual_machine.h"

#include <string.h>

#include "internal_error.h"
#include "lum_function_tables.h"

LuminaryResult lum_virtual_machine_create(LumVirtualMachine** vm) {
  __CHECK_NULL_ARGUMENT(vm);

  __FAILURE_HANDLE(host_malloc(vm, sizeof(LumVirtualMachine)));
  memset(*vm, 0, sizeof(LumVirtualMachine));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_virtual_machine_execute_nop(LumVirtualMachine* vm, LuminaryHost* host, const LumInstruction* instruction) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(instruction);

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_virtual_machine_execute_mov(LumVirtualMachine* vm, LuminaryHost* host, const LumInstruction* instruction) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(instruction);

  LumBuiltinType type;
  __FAILURE_HANDLE(lum_instruction_decode_mov(instruction, &type));

  const size_t size = lum_builtin_types_sizes[type];

  void* dst;
  __FAILURE_HANDLE(lum_function_resolve_stack_address(vm, &instruction->dst, &dst));

  const void* src;
  __FAILURE_HANDLE(lum_function_resolve_generic_address(vm, &instruction->src, &src));

  memcpy(dst, src, size);

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_virtual_machine_execute_ldg(LumVirtualMachine* vm, LuminaryHost* host, const LumInstruction* instruction) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(instruction);

  LumBuiltinType type;
  __FAILURE_HANDLE(lum_instruction_decode_ldg(instruction, &type));

  LumFunctionLoadInfo info;
  info.dst = instruction->dst;
  info.src = instruction->src;

  __FAILURE_HANDLE(lum_function_tables_ldg[type](host, vm, &info));

  return LUMINARY_SUCCESS;
}

static LuminaryResult _lum_virtual_machine_execute_stg(LumVirtualMachine* vm, LuminaryHost* host, const LumInstruction* instruction) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(instruction);

  LumBuiltinType type;
  __FAILURE_HANDLE(lum_instruction_decode_stg(instruction, &type));

  LumFunctionStoreInfo info;
  info.src = instruction->src;

  __FAILURE_HANDLE(lum_function_tables_stg[type](host, vm, &info));

  return LUMINARY_SUCCESS;
}

typedef LuminaryResult (*LumVirtualMachineInstructionFunc)(LumVirtualMachine* vm, LuminaryHost* host, const LumInstruction* instruction);

static LumVirtualMachineInstructionFunc _lum_virtual_machine_instruction_funcs[LUM_INSTRUCTION_TYPE_COUNT] = {
  [LUM_INSTRUCTION_TYPE_NOP] = (const LumVirtualMachineInstructionFunc) _lum_virtual_machine_execute_nop,
  [LUM_INSTRUCTION_TYPE_MOV] = (const LumVirtualMachineInstructionFunc) _lum_virtual_machine_execute_mov,
  [LUM_INSTRUCTION_TYPE_LDG] = (const LumVirtualMachineInstructionFunc) _lum_virtual_machine_execute_ldg,
  [LUM_INSTRUCTION_TYPE_STG] = (const LumVirtualMachineInstructionFunc) _lum_virtual_machine_execute_stg};

LuminaryResult lum_virtual_machine_execute(
  LumVirtualMachine* vm, LuminaryHost* host, const LumBinary* binary, const LumVirtualMachineExecutionInfo* info) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(binary);

  if (binary->stack_size > vm->stack_size) {
    if (vm->stack_memory) {
      __FAILURE_HANDLE(host_free(&vm->stack_memory));
    }

    __FAILURE_HANDLE(host_malloc(&vm->stack_memory, binary->stack_size));

    vm->stack_size = binary->stack_size;
  }

  if (info->debug_mode) {
    memset(vm->stack_memory, 0xCD, vm->stack_size);
  }

  vm->constant_memory      = binary->constant_memory;
  vm->constant_memory_size = binary->constant_memory_size;

  uint32_t num_instructions;
  __FAILURE_HANDLE(array_get_num_elements(binary->instructions, &num_instructions));

  for (uint32_t instruction_id = 0; instruction_id < num_instructions; instruction_id++) {
    const LumInstruction* instruction = binary->instructions + instruction_id;

    LumInstructionType type;
    __FAILURE_HANDLE(lum_instruction_get_type(instruction, &type));

    __FAILURE_HANDLE(_lum_virtual_machine_instruction_funcs[type](vm, host, instruction));
  }

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_virtual_machine_destroy(LumVirtualMachine** vm) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(*vm);

  __FAILURE_HANDLE(host_free(vm));

  return LUMINARY_SUCCESS;
}
