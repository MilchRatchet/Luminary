#include "utils.h"

const char* luminary_result_to_string(LuminaryResult result) {
  switch (result & ~LUMINARY_ERROR_PROPAGATED) {
    case LUMINARY_SUCCESS:
      return "Success";
    case LUMINARY_ERROR_ARGUMENT_NULL:
      return "Encountered NULL argument";
    case LUMINARY_ERROR_NOT_IMPLEMENTED:
      return "Encountered a section that is not implemented";
    case LUMINARY_ERROR_INVALID_API_ARGUMENT:
      return "Encountered an invalid argument";
    case LUMINARY_ERROR_MEMORY_LEAK:
      return "Identified a memory leak";
    case LUMINARY_ERROR_OUT_OF_MEMORY:
      return "Ran out of memory";
    case LUMINARY_ERROR_C_STD:
      return "Encountered an error in a call to a C stdlib function";
    case LUMINARY_ERROR_API_EXCEPTION:
      return "Encountered an internal error";
    case LUMINARY_ERROR_CUDA:
      return "Encountered an error reported by CUDA";
    case LUMINARY_ERROR_OPTIX:
      return "Encountered an error reported by OptiX";
    case LUMINARY_ERROR_PREVIOUS_ERROR:
      return "Encountered an unstable state due to a previous error";
    default:
      return "Unknown";
  }
}
