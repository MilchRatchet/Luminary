#ifndef MANDARIN_DUCK_ELEMENT_H
#define MANDARIN_DUCK_ELEMENT_H

#include "utils.h"

enum ElementType {
  ELEMENT_TYPE_CONTAINER,
  ELEMENT_TYPE_BUTTON,
  ELEMENT_TYPE_TEXT,
  ELEMENT_TYPE_CHECK_BOX,
  ELEMENT_TYPE_COLOR,
  ELEMENT_TYPE_DROPDOWN,
  ELEMENT_TYPE_COUNT
} typedef ElementType;

struct WindowRenderContext typedef WindowRenderContext;

struct Element {
  ElementType type;
  void* data;
  void (*action_func)(Element* element, LuminaryHost* host, void* args);
  void (*render_func)(Element* element, WindowRenderContext* context, void* args);
  void (*destroy_func)(Element* element);
} typedef Element;

#endif /* MANDARIN_DUCK_ELEMENT_H */
