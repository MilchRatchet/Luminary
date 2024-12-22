#ifndef MANDARIN_DUCK_ELEMENT_H
#define MANDARIN_DUCK_ELEMENT_H

#include "utils.h"

#define ELEMENT_DATA_SECTION_SIZE 0x100

struct Display typedef Display;
struct Element typedef Element;

enum ElementType {
  ELEMENT_TYPE_BUTTON,
  ELEMENT_TYPE_TEXT,
  ELEMENT_TYPE_SLIDER,
  ELEMENT_TYPE_CHECK_BOX,
  ELEMENT_TYPE_COLOR,
  ELEMENT_TYPE_DROPDOWN,
  ELEMENT_TYPE_COUNT
} typedef ElementType;

struct Element {
  ElementType type;
  uint32_t x;
  uint32_t y;
  uint32_t width;
  uint32_t height;
  uint8_t data[ELEMENT_DATA_SECTION_SIZE];
  void (*render_func)(Element* element, Display* display);
} typedef Element;

#endif /* MANDARIN_DUCK_ELEMENT_H */
