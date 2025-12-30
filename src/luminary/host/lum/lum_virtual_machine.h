#ifndef LUMINARY_LUM_VIRTUAL_MACHINE_H
#define LUMINARY_LUM_VIRTUAL_MACHINE_H

#include "lum_binary.h"
#include "utils.h"

struct LumVirtualMachine {
  size_t stack_size;
  void* stack_memory;
  size_t constant_memory_size;
  const void* constant_memory;
} typedef LumVirtualMachine;

struct LumVirtualMachineExecutionInfo {
  LumBinaryEntryPoint entry_point;
  bool debug_mode;
} typedef LumVirtualMachineExecutionInfo;

LuminaryResult lum_virtual_machine_create(LumVirtualMachine** vm);
LuminaryResult lum_virtual_machine_execute(
  LumVirtualMachine* vm, LuminaryHost* host, const LumBinary* binary, const LumVirtualMachineExecutionInfo* info);
LuminaryResult lum_virtual_machine_destroy(LumVirtualMachine** vm);

#endif /* LUMINARY_LUM_VIRTUAL_MACHINE_H */
