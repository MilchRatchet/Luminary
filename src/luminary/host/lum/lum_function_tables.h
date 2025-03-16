#ifndef LUMINARY_LUM_FUNCTION_TABLES_H
#define LUMINARY_LUM_FUNCTION_TABLES_H

#include "lum_builtins.h"
#include "utils.h"

struct LumFunctionOperand {
  uint8_t register_id;
} typedef LumFunctionOperand;

struct LumFunctionSignature {
  LumBuiltinType dst;
  LumBuiltinType src_0;
  LumBuiltinType src_1;
  LumBuiltinType src_2;
  LumBuiltinType src_3;
} typedef LumFunctionSignature;

// TODO:
typedef void (*LumFunction)(
  LumFunctionOperand* dst, const LumFunctionOperand* src_a, const LumFunctionOperand* src_b, const LumFunctionOperand* src_c);

struct LumFunctionEntry {
  const char* name;
  const LumFunction* func;
  bool is_static;
  LumFunctionSignature signature;
} typedef LumFunctionEntry;

extern const LumFunctionEntry* lum_function_tables[LUM_BUILTIN_TYPE_COUNT];
extern const uint32_t lum_function_tables_count[LUM_BUILTIN_TYPE_COUNT];

#endif /* LUMINARY_LUM_FUNCTION_TABLES_H */
