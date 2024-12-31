#ifndef MANDARIN_DUCK_WINDOWS_SUBWINDOW_DROPDOWN_H
#define MANDARIN_DUCK_WINDOWS_SUBWINDOW_DROPDOWN_H

#include "utils.h"
#include "window.h"

void subwindow_dropdown_add_string(Window* window, const char* string);
void subwindow_dropdown_create(Window* window, uint32_t selected_index, uint32_t width, uint32_t x, uint32_t y);

#endif /* MANDARIN_DUCK_WINDOWS_SUBWINDOW_DROPDOWN_H */
