#ifndef MANDARIN_DUCK_WINDOWS_SUBWINDOW_TOOLTIP_H
#define MANDARIN_DUCK_WINDOWS_SUBWINDOW_TOOLTIP_H

#include "utils.h"
#include "window.h"

void subwindow_tooltip_create(Window* window, const char* text, uint32_t x, uint32_t y);
void subwindow_tooltip_create_stack_trace(
  Window* window, Display* display, const char* text, uint32_t allocation_index, float wheel_motion, uint32_t x, uint32_t y);
void subwindow_tooltip_destroy(Window* window);

#endif /* MANDARIN_DUCK_WINDOWS_SUBWINDOW_TOOLTIP_H */
