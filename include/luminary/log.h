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

#ifndef LOG_H
#define LOG_H

#include <luminary/api_utils.h>

#define log_message(fmt, ...) luminary_print_log("[%s:%d] " fmt, __func__, __LINE__, ##__VA_ARGS__)
#define info_message(fmt, ...) luminary_print_info("[%s:%d] " fmt, __func__, __LINE__, ##__VA_ARGS__)
#define warn_message(fmt, ...) luminary_print_warn("[%s:%d] " fmt, __func__, __LINE__, ##__VA_ARGS__)
#define error_message(fmt, ...) luminary_print_error("[%s:%d] " fmt, __func__, __LINE__, ##__VA_ARGS__)
#define crash_message(fmt, ...) luminary_print_crash("[%s:%d] " fmt, __func__, __LINE__, ##__VA_ARGS__)

#if __cplusplus
extern "C" {
#endif

/*
 * Writes a message to log only.
 * @param format Message which may contain format specifiers.
 * @param ... Additional arguments to replace the format speciefiers.
 */
LUMINARY_API void luminary_print_log(const char* format, ...);

/*
 * Writes a message.
 * @param format Message which may contain format specifiers.
 * @param ... Additional arguments to replace the format speciefiers.
 */
LUMINARY_API void luminary_print_info(const char* format, ...);

/*
 * Writes a message without a newline. This message will be overwritten by any following message.
 * @param format Message which may contain format specifiers.
 * @param ... Additional arguments to replace the format speciefiers.
 */
LUMINARY_API void luminary_print_info_inline(const char* format, ...);

/*
 * Writes a message in yellow text.
 * @param format Message which may contain format specifiers.
 * @param ... Additional arguments to replace the format speciefiers.
 */
LUMINARY_API void luminary_print_warn(const char* format, ...);

/*
 * Writes a message in red text.
 * @param format Message which may contain format specifiers.
 * @param ... Additional arguments to replace the format speciefiers.
 */
LUMINARY_API void luminary_print_error(const char* format, ...);

/*
 * Writes a message in purple text and terminates the program.
 * @param format Message which may contain format specifiers.
 * @param ... Additional arguments to replace the format speciefiers.
 */
LUMINARY_API void luminary_print_crash(const char* format, ...);

LUMINARY_API void luminary_write_log();

#if __cplusplus
}
#endif

#endif /* LOG_H */
