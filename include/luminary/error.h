/*
  Copyright (C) 2021-2024 Max Jenke

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
#define LUMINARY_SUCCESS (0x0000000000000000ull)

/*
 * Error was propagated from an internal function returning an error.
 */
#define LUMINARY_ERROR_PROPAGATED (0x0000000000000001ull)

/*
 * Non-optional argument was NULL.
 */
#define LUMINARY_ERROR_ARGUMENT_NULL (0x0000000000000002ull)

/*
 * Encountered code path that was not implemented.
 */
#define LUMINARY_ERROR_NOT_IMPLEMENTED (0x0000000000000004ull)

/*
 * Argument given to an API function was invalid.
 */
#define LUMINARY_ERROR_INVALID_API_ARGUMENT (0x0000000000000008ull)

/*
 * Action would cause a memory leak.
 */
#define LUMINARY_ERROR_MEMORY_LEAK (0x0000000000000010ull)

/*
 * Insufficient memory for action.
 */
#define LUMINARY_ERROR_OUT_OF_MEMORY (0x0000000000000020ull)

/*
 * Error in C standard library.
 */
#define LUMINARY_ERROR_C_STD (0x0000000000000040ull)

/*
 * API function was used in a non-compliant way.
 */
#define LUMINARY_ERROR_API_EXCEPTION (0x0000000000000080ull)

/*
 * Error in CUDA library.
 */
#define LUMINARY_ERROR_CUDA (0x0000000000000100ull)

/*
 * Error in OptiX library.
 */
#define LUMINARY_ERROR_OPTIX (0x0000000000000200ull)

/*
 * Error due to Luminary being in an unstable state caused by a previous error.
 */
#define LUMINARY_ERROR_PREVIOUS_ERROR (0x0000000000000400ull)

LUMINARY_API const char* luminary_result_to_string(LuminaryResult result);

#endif /* LUMINARY_API_ERROR_H */
