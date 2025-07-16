#ifndef MANDARIN_DUCK_UTILS_H
#define MANDARIN_DUCK_UTILS_H

#define LUMINARY_INCLUDE_EXTRA_UTILS

#include <luminary/luminary.h>

#define LUM_FAILURE_HANDLE(command)                                                           \
  {                                                                                           \
    LuminaryResult __lum_res = command;                                                       \
    if (__lum_res != LUMINARY_SUCCESS) {                                                      \
      crash_message("Luminary API returned error: %s", luminary_result_to_string(__lum_res)); \
    }                                                                                         \
  }

#define MD_CHECK_NULL_ARGUMENT(arg)     \
  if (!(arg)) {                         \
    crash_message("%s is NULL.", #arg); \
  }

#define MD_UNUSED(__macro_x) ((void) (__macro_x))

#define MD_COLOR_WHITE 0xFFFBF4DB
#define MD_COLOR_GRAY 0xFFFDE7BB
#define MD_COLOR_ACCENT_1 0xFFEB5B00
#define MD_COLOR_ACCENT_2 0xFFAA5486
#define MD_COLOR_ACCENT_LIGHT_1 0xFFF0BB78
#define MD_COLOR_ACCENT_LIGHT_2 0xFFEFB6C8
#define MD_COLOR_BORDER 0xFF111111
#define MD_COLOR_BLACK 0xFF000000
#define MD_COLOR_WINDOW_BACKGROUND 0xFF111928
#define MD_COLOR_DISPLAY_BACKGROUND 0xFF081018

#endif /* MANDARIN_DUCK_UTILS_H */
