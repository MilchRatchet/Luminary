#ifndef LUMINARY_API_ERROR_H
#define LUMINARY_API_ERROR_H

#include <luminary/api_utils.h>
#include <stdint.h>

typedef uint64_t LuminaryResult;

LUMINARY_API char* luminary_result_to_string(LuminaryResult result);

#endif /* LUMINARY_API_ERROR_H */
