#ifndef MANDARIN_DUCK_WINDOWS_DEBUG_WINDOW_H
#define MANDARIN_DUCK_WINDOWS_DEBUG_WINDOW_H

#include "debug_window_types.h"
#include "window.h"

const char* window_debug_type_get_name(WindowDebugType type);
void window_debug_create(Window** window, WindowDebugType type);

#endif /* MANDARIN_DUCK_WINDOWS_DEBUG_WINDOW_H */
