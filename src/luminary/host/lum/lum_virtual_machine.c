#include "lum_virtual_machine.h"

#include <string.h>

#include "internal_error.h"

LuminaryResult lum_virtual_machine_create(LumVirtualMachine** vm) {
  __CHECK_NULL_ARGUMENT(vm);

  __FAILURE_HANDLE(host_malloc(vm, sizeof(LumVirtualMachine)));
  memset(*vm, 0, sizeof(LumVirtualMachine));

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_virtual_machine_execute(
  LumVirtualMachine* vm, LuminaryHost* host, const LumBinary* binary, LumVirtualMachineExecutionArgs args) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(host);
  __CHECK_NULL_ARGUMENT(binary);

  return LUMINARY_SUCCESS;
}

LuminaryResult lum_virtual_machine_destroy(LumVirtualMachine** vm) {
  __CHECK_NULL_ARGUMENT(vm);
  __CHECK_NULL_ARGUMENT(*vm);

  __FAILURE_HANDLE(host_free(vm));

  return LUMINARY_SUCCESS;
}
