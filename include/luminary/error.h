/*
  Copyright (C) 2021-2025 Max Jenke

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU Affero General Public License as published
  by the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU Affero General Public License for more details.

  You should have received a copy of the GNU Affero General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#ifndef LUMINARY_API_ERROR_H
#define LUMINARY_API_ERROR_H

#include <luminary/api_utils.h>
#include <stdint.h>

typedef uint64_t LuminaryResult;

#define LUMINARY_SUCCESS (0ull)

struct LuminaryStackTrace {
  const char* function_name;
  const char* file_name;
  uint64_t line;
  struct LuminaryStackTrace* caller;
} typedef LuminaryStackTrace;

enum LuminaryErrorKind {
  LUMINARY_ERROR_NONE,
  LUMINARY_ERROR_UNKNOWN,               // Unknown
  LUMINARY_ERROR_ARGUMENT_NULL,         // Non-optional argument was NULL.
  LUMINARY_ERROR_NOT_IMPLEMENTED,       // Encountered code path that was not implemented.
  LUMINARY_ERROR_INVALID_API_ARGUMENT,  // Argument given to an API function was invalid.
  LUMINARY_ERROR_MEMORY_LEAK,           // Action would cause a memory leak.
  LUMINARY_ERROR_OUT_OF_MEMORY,         // Insufficient memory for action.
  LUMINARY_ERROR_C_STD,                 // Error in C standard library.
  LUMINARY_ERROR_API_EXCEPTION,         // API function was used in a non-compliant way.
  LUMINARY_ERROR_CUDA,                  // Error in CUDA library.
  LUMINARY_ERROR_OPTIX,                 // Error in OptiX library.
  LUMINARY_ERROR_PREVIOUS_ERROR,        // Error due to Luminary being in an unstable state caused by a previous error.
  LUMINARY_ERROR_DEBUG_ASSERT,          // Error due to a debug condition being violated.
  LUMINARY_ERROR_MISSING_DATA,          // Error due to embedded data missing.
  LUMINARY_ERROR_INVALID_DEVICE,        // Error due to specifying an invalid device.
  LUMINARY_ERROR_KIND_COUNT,
} typedef LuminaryErrorKind;

struct LuminaryError {
  LuminaryErrorKind kind;
  const char* message;
  LuminaryStackTrace* trace;
  uint64_t host_id;
} typedef LuminaryError;

LUMINARY_API void luminary_get_error_details(LuminaryResult result, const LuminaryError** error);

#endif /* LUMINARY_API_ERROR_H */
