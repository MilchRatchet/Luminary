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

#ifndef LOG_H
#define LOG_H

#include <luminary/api_utils.h>

#define log_message(fmt, ...) luminary_print_log("[%s:%d] " fmt, __func__, __LINE__, ##__VA_ARGS__)
#define info_message(fmt, ...)                                             \
  {                                                                        \
    luminary_print_info(false, fmt, ##__VA_ARGS__);                        \
    luminary_print_log("[%s:%d] " fmt, __func__, __LINE__, ##__VA_ARGS__); \
  }
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
LUMINARY_API void luminary_print_info(bool log, const char* format, ...);

/*
 * Writes a message without a newline. This message will be overwritten by any following message.
 * @param format Message which may contain format specifiers.
 * @param ... Additional arguments to replace the format speciefiers.
 */
LUMINARY_API void luminary_print_info_inline(bool log, const char* format, ...);

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
