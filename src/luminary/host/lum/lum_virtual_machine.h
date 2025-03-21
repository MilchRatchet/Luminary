#ifndef LUMINARY_LUM_VIRTUAL_MACHINE_H
#define LUMINARY_LUM_VIRTUAL_MACHINE_H

#include "lum_binary.h"
#include "utils.h"

#define LUM_VIRTUAL_MACHINE_NUM_REGISTERS 0x100

struct LumVirtualMachine {
  size_t stack_size;
  void* stack_memory;
  size_t register_map[LUM_VIRTUAL_MACHINE_NUM_REGISTERS];
} typedef LumVirtualMachine;

struct LumVirtualMachineExecutionArgs {
  LumBinaryEntryPoint entry_point;
} typedef LumVirtualMachineExecutionArgs;

LuminaryResult lum_virtual_machine_create(LumVirtualMachine** vm);
LuminaryResult lum_virtual_machine_execute(
  LumVirtualMachine* vm, LuminaryHost* host, const LumBinary* binary, LumVirtualMachineExecutionArgs args);
LuminaryResult lum_virtual_machine_destroy(LumVirtualMachine** vm);

#endif /* LUMINARY_LUM_VIRTUAL_MACHINE_H */
