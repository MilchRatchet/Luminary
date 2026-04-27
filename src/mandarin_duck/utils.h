#ifndef MANDARIN_DUCK_UTILS_H
#define MANDARIN_DUCK_UTILS_H

#define LUMINARY_INCLUDE_EXTRA_UTILS

#include <luminary/luminary.h>

#define LUM_FAILURE_HANDLE(__lum_command)                             \
  {                                                                   \
    LuminaryResult __lum_result = (__lum_command);                    \
    if (__lum_result != LUMINARY_SUCCESS) {                           \
      crash_message("Luminary API returned error: %u", __lum_result); \
    }                                                                 \
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
