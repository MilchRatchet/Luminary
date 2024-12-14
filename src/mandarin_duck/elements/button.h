#ifndef MANDARIN_DUCK_ELEMENTS_BUTTON_H
#define MANDARIN_DUCK_ELEMENTS_BUTTON_H

#include "utils.h"

enum ElementButtonShape { ELEMENT_BUTTON_SHAPE_CIRCLE = 0 } typedef ElementButtonShape;

struct ElementButtonData {
  ElementButtonShape shape;
  XRGB8 color;
} typedef ElementButtonData;

struct ElementButtonArgs {
  ElementButtonShape shape;
  uint32_t color;
} typedef ElementButtonArgs;

#endif /* MANDARIN_DUCK_ELEMENTS_BUTTON_H */
