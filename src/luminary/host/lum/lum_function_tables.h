#ifndef LUMINARY_LUM_FUNCTION_TABLES_H
#define LUMINARY_LUM_FUNCTION_TABLES_H

#include "lum_builtins.h"
#include "utils.h"

// TODO:
typedef void (*LumFunction)(void);

struct LumFunctionEntry {
  const char* name;
  bool is_static;
  LumFunction func;
} typedef LumFunctionEntry;

extern LumFunctionEntry* lum_function_tables[LUM_BUILTIN_TYPE_COUNT];
extern uint32_t lum_function_tables_count[LUM_BUILTIN_TYPE_COUNT];

#endif /* LUMINARY_LUM_FUNCTION_TABLES_H */
