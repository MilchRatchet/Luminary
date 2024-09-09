#ifndef LUMINARY_ERROR_H
#define LUMINARY_ERROR_H

#include <stdint.h>

#include "utils.h"

typedef uint64_t LuminaryResult;

LUMINARY_API char* luminary_result_to_string(LuminaryResult result);

#endif /* LUMINARY_ERROR_H */
