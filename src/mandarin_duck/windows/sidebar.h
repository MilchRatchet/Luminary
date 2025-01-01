#ifndef MANDARIN_DUCK_WINDOWS_SIDEBAR_H
#define MANDARIN_DUCK_WINDOWS_SIDEBAR_H

#include "entity_properties.h"
#include "utils.h"
#include "window.h"

void window_sidebar_create(Window** window);
void window_sidebar_register_window_id(Window* window, WindowEntityPropertiesType type, uint32_t window_id);

#endif /* MANDARIN_DUCK_WINDOWS_SIDEBAR_H */
