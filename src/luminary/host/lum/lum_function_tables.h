#ifndef LUMINARY_LUM_FUNCTION_TABLES_H
#define LUMINARY_LUM_FUNCTION_TABLES_H

#include "lum_builtins.h"
#include "lum_instruction.h"
#include "utils.h"

typedef void (*LumFunction)(LumMemoryAllocation dst, LumMemoryAllocation src);

extern const LumFunction* lum_function_tables_ldg[LUM_BUILTIN_TYPE_COUNT];
extern const LumFunction* lum_function_tables_stg[LUM_BUILTIN_TYPE_COUNT];

#endif /* LUMINARY_LUM_FUNCTION_TABLES_H */
