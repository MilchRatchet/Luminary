/*
  Copyright (c) 2021-2024, MilchRatchet

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
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

LUMINARY_API const char* luminary_result_to_string(LuminaryResult result);

#endif /* LUMINARY_API_ERROR_H */
