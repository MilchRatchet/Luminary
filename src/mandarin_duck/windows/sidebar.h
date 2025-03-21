#ifndef MANDARIN_DUCK_WINDOWS_SIDEBAR_H
#define MANDARIN_DUCK_WINDOWS_SIDEBAR_H

#include "entity_properties.h"
#include "utils.h"
#include "window.h"

enum WindowSidebarType {
  WINDOW_SIDEBAR_TYPE_ENTITY_PROPERTIES,
  WINDOW_SIDEBAR_TYPE_MOUSE_MODES,
  WINDOW_SIDEBAR_TYPE_COUNT
} typedef WindowSidebarType;

void window_sidebar_create(Window** window, WindowSidebarType type);
void window_sidebar_register_window_id(Window* window, WindowEntityPropertiesType type, uint32_t window_id);

#endif /* MANDARIN_DUCK_WINDOWS_SIDEBAR_H */
