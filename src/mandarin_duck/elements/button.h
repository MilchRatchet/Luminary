#ifndef MANDARIN_DUCK_ELEMENTS_BUTTON_H
#define MANDARIN_DUCK_ELEMENTS_BUTTON_H

#include <assert.h>

#include "elements_common.h"
#include "utils.h"

enum ElementButtonShape { ELEMENT_BUTTON_SHAPE_CIRCLE, ELEMENT_BUTTON_SHAPE_IMAGE, ELEMENT_BUTTON_SHAPE_COUNT } typedef ElementButtonShape;

enum ElementButtonImage {
  ELEMENT_BUTTON_IMAGE_CHECK,
  ELEMENT_BUTTON_IMAGE_SETTINGS,
  ELEMENT_BUTTON_IMAGE_CAMERA,
  ELEMENT_BUTTON_IMAGE_WAVES,
  ELEMENT_BUTTON_IMAGE_SUN,
  ELEMENT_BUTTON_IMAGE_CLOUD,
  ELEMENT_BUTTON_IMAGE_MIST,
  ELEMENT_BUTTON_IMAGE_PRECIPITATION,
  ELEMENT_BUTTON_IMAGE_MATERIAL,
  ELEMENT_BUTTON_IMAGE_INSTANCE,
  ELEMENT_BUTTON_IMAGE_MOVE,
  ELEMENT_BUTTON_IMAGE_SELECT,
  ELEMENT_BUTTON_IMAGE_FOCUS,
  ELEMENT_BUTTON_IMAGE_SYNC,
  ELEMENT_BUTTON_IMAGE_REGION,
  ELEMENT_BUTTON_IMAGE_ERROR,
  ELEMENT_BUTTON_IMAGE_COUNT
} typedef ElementButtonImage;

struct ElementButtonData {
  ElementButtonShape shape;
  ElementButtonImage image;
  uint32_t shape_size_id;
  uint32_t color;
  uint32_t hover_color;
  uint32_t press_color;
  bool is_hovered;
  bool is_down;
} typedef ElementButtonData;
static_assert(sizeof(ElementButtonData) <= ELEMENT_DATA_SECTION_SIZE, "Element data exceeds allocated size.");

struct ElementButtonArgs {
  ElementSize size;
  ElementButtonShape shape;
  ElementButtonImage image;
  uint32_t color;
  uint32_t hover_color;
  uint32_t press_color;
  const char* tooltip_text;
} typedef ElementButtonArgs;

bool element_button(Window* window, Display* display, const MouseState* mouse_state, ElementButtonArgs args);

#endif /* MANDARIN_DUCK_ELEMENTS_BUTTON_H */
