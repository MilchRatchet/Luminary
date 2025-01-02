#ifndef MANDARIN_DUCK_ELEMENTS_SLIDER_H
#define MANDARIN_DUCK_ELEMENTS_SLIDER_H

#include <assert.h>

#include "elements_common.h"
#include "utils.h"

struct KeyboardState typedef KeyboardState;

enum ElementSliderDataType {
  ELEMENT_SLIDER_DATA_TYPE_FLOAT,
  ELEMENT_SLIDER_DATA_TYPE_UINT,
  ELEMENT_SLIDER_DATA_TYPE_SINT,
  ELEMENT_SLIDER_DATA_TYPE_VECTOR,
  ELEMENT_SLIDER_DATA_TYPE_RGB
} typedef ElementSliderDataType;

struct ElementSliderData {
  ElementSliderDataType type;
  float data_float;
  uint32_t data_uint;
  int32_t data_sint;
  LuminaryVec3 data_vec3;
  ElementSize size;
  uint32_t color;
  uint32_t component_padding;
  uint32_t margins;
  bool center_x;
  bool center_y;
  bool string_edit_mode;
  char string[WINDOW_STATE_STRING_SIZE];
  uint32_t string_component_index;
} typedef ElementSliderData;
static_assert(sizeof(ElementSliderData) <= ELEMENT_DATA_SECTION_SIZE, "Element data exceeds allocated size.");

struct ElementSliderArgs {
  const char* identifier;
  ElementSliderDataType type;
  void* data_binding;
  float max;
  float min;
  float change_rate;
  ElementSize size;
  uint32_t color;
  uint32_t component_padding;
  uint32_t margins;
  bool center_x;
  bool center_y;
} typedef ElementSliderArgs;

bool element_slider(
  Window* window, Display* display, const MouseState* mouse_state, const KeyboardState* keyboard_state, ElementSliderArgs args);

#endif /* MANDARIN_DUCK_ELEMENTS_SLIDER_H */
