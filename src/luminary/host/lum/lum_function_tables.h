#ifndef LUMINARY_LUM_FUNCTION_TABLES_H
#define LUMINARY_LUM_FUNCTION_TABLES_H

#include "lum_builtins.h"
#include "lum_instruction.h"
#include "lum_virtual_machine.h"
#include "utils.h"

struct LumFunctionLoadInfo {
  LumMemoryAllocation dst;
  LumMemoryAllocation src;
} typedef LumFunctionLoadInfo;

typedef LuminaryResult (*LumFunctionLoad)(LuminaryHost* host, LumVirtualMachine* vm, const LumFunctionLoadInfo* info);

struct LumFunctionStoreInfo {
  LumMemoryAllocation src;
} typedef LumFunctionStoreInfo;

typedef LuminaryResult (*LumFunctionStore)(LuminaryHost* host, LumVirtualMachine* vm, const LumFunctionStoreInfo* info);

extern const LumFunctionLoad lum_function_tables_ldg[LUM_BUILTIN_TYPE_COUNT];
extern const LumFunctionStore lum_function_tables_stg[LUM_BUILTIN_TYPE_COUNT];

LuminaryResult lum_function_resolve_stack_address(LumVirtualMachine* vm, const LumMemoryAllocation* mem, void** ptr);
LuminaryResult lum_function_resolve_generic_address(LumVirtualMachine* vm, const LumMemoryAllocation* mem, const void** ptr);

#endif /* LUMINARY_LUM_FUNCTION_TABLES_H */
