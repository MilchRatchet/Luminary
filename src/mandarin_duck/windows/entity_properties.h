#ifndef MANDARIN_DUCK_WINDOWS_ENTITY_PROPERTIES_H
#define MANDARIN_DUCK_WINDOWS_ENTITY_PROPERTIES_H

#include "utils.h"
#include "window.h"

enum WindowEntityPropertiesType {
  WINDOW_ENTITY_PROPERTIES_TYPE_CAMERA,
  WINDOW_ENTITY_PROPERTIES_TYPE_COUNT
} typedef WindowEntityPropertiesType;

void window_entity_properties_create(Window** window);

#endif /* MANDARIN_DUCK_WINDOWS_ENTITY_PROPERTIES_H */
