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

/*
 * No error.
 */
#define LUMINARY_SUCCESS (0ull)

/*
 * Non-optional argument was NULL.
 */
#define LUMINARY_ERROR_ARGUMENT_NULL (1ull)

/*
 * Encountered code path that was not implemented.
 */
#define LUMINARY_ERROR_NOT_IMPLEMENTED (2ull)

/*
 * Argument given to an API function was invalid.
 */
#define LUMINARY_ERROR_INVALID_API_ARGUMENT (3ull)

/*
 * Action would cause a memory leak.
 */
#define LUMINARY_ERROR_MEMORY_LEAK (4ull)

/*
 * Insufficient memory for action.
 */
#define LUMINARY_ERROR_OUT_OF_MEMORY (5ull)

/*
 * Error in C standard library.
 */
#define LUMINARY_ERROR_C_STD (6ull)

/*
 * API function was used in a non-compliant way.
 */
#define LUMINARY_ERROR_API_EXCEPTION (7ull)

/*
 * Error in CUDA library.
 */
#define LUMINARY_ERROR_CUDA (8ull)

/*
 * Error in OptiX library.
 */
#define LUMINARY_ERROR_OPTIX (9ull)

/*
 * Error due to Luminary being in an unstable state caused by a previous error.
 */
#define LUMINARY_ERROR_PREVIOUS_ERROR (10ull)

/*
 * Error due to a debug condition being violated.
 */
#define LUMINARY_ERROR_DEBUG_ASSERT (11ull)

/*
 * Error due to embedded data missing.
 */
#define LUMINARY_ERROR_MISSING_DATA (12ull)

/*
 * Error was propagated from an internal function returning an error.
 */
#define LUMINARY_ERROR_PROPAGATED (0x8000000000000000ull)

LUMINARY_API const char* luminary_result_to_string(LuminaryResult result);

#endif /* LUMINARY_API_ERROR_H */
