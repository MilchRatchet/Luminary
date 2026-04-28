#ifndef MANDARIN_DUCK_UTILS_H
#define MANDARIN_DUCK_UTILS_H

#define LUMINARY_INCLUDE_EXTRA_UTILS

#include <luminary/luminary.h>

#define LUM_FAILURE_HANDLE(__lum_command)                                                                                          \
  {                                                                                                                                \
    LuminaryResult __lum_result = (__lum_command);                                                                                 \
    if (__lum_result != LUMINARY_SUCCESS) {                                                                                        \
      const LuminaryError* __lum_error;                                                                                            \
      luminary_get_error_details(__lum_result, &__lum_error);                                                                      \
                                                                                                                                   \
      if (__lum_error != (const LuminaryError*) 0) {                                                                               \
        error_message("Luminary API returned error code: %s.", luminary_strings_error_kind[__lum_error->kind]);                    \
        error_message("Error message: %s.", (__lum_error->message) ? __lum_error->message : "None");                               \
        const LuminaryStackTrace* __lum_stacktrace = __lum_error->trace;                                                           \
        while (__lum_stacktrace != (LuminaryStackTrace*) 0) {                                                                      \
          error_message("\tat %s in %s:%u", __lum_stacktrace->function_name, __lum_stacktrace->file_name, __lum_stacktrace->line); \
          __lum_stacktrace = __lum_stacktrace->caller;                                                                             \
        }                                                                                                                          \
      }                                                                                                                            \
      else {                                                                                                                       \
        error_message("Luminary API returned unspecified error.");                                                                 \
      }                                                                                                                            \
                                                                                                                                   \
      crash_message("Luminary API ran into unrecoverable error.");                                                                 \
    }                                                                                                                              \
  }

#define MD_CHECK_NULL_ARGUMENT(__md_argument)     \
  if (!(__md_argument)) {                         \
    crash_message("%s is NULL.", #__md_argument); \
  }

#define MD_UNUSED(__macro_x) ((void) (__macro_x))

#define MD_COLOR_WHITE 0xFFFBF4DB
#define MD_COLOR_GRAY 0xFFFDE7BB
#define MD_COLOR_DARKGRAY 0xFF898581
#define MD_COLOR_ACCENT_1 0xFFEB5B00
#define MD_COLOR_ACCENT_2 0xFFAA5486
#define MD_COLOR_ACCENT_LIGHT_1 0xFFF0BB78
#define MD_COLOR_ACCENT_LIGHT_2 0xFFEFB6C8
#define MD_COLOR_BORDER 0xFF111111
#define MD_COLOR_BLACK 0xFF000000
#define MD_COLOR_WINDOW_BACKGROUND 0xFF111928

#endif /* MANDARIN_DUCK_UTILS_H */
